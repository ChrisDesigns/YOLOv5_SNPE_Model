

#
# Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

#
# modified and adapted by Chris Lenart for academic use and support for YOLOv5 
#

'''
Helper script to download artifacts to run YOLOv5 model with SNPE SDK.
'''
from google.protobuf.json_format import MessageToDict
import onnx
import os
import subprocess
import hashlib
import argparse
import sys

COCO_VAL2007_ARCHIVE_CHECKSUM = '442b8da7639aecaf257c1dceb8ba8c80'
YOLO_V5_ONNX_FILENAME = "yolov5s.onnx"
YOLO_V5_DLC_FILENAME = "yolov5s.dlc"
YOLO_V5_DLC_QUANTIZED_FILENAME = "yolov5s_quantized.dlc"

def wget(download_dir, file_url):
    cmd = ['wget', '-N', '--directory-prefix={}'.format(download_dir), file_url]
    subprocess.call(cmd)

def generateMd5(path):
    checksum = hashlib.md5()
    with open(path, 'rb') as data_file:
        while True:
            block = data_file.read(checksum.block_size)
            if not block:
                break
            checksum.update(block)
    return checksum.hexdigest()

def checkResource(data_dir, filename, md5):
    filepath = os.path.join(data_dir, filename)
    if not os.path.isfile(filepath):
        raise RuntimeError(filename + ' not found at the location specified by ' + data_dir + '. Re-run with download option.')
    else:
        checksum = generateMd5(filepath)
        if not checksum == md5:
            raise RuntimeError('Checksum of ' + filename + ' : ' + checksum + ' does not match checksum of file ' + md5)

def setup_assets(model_dir, download, runtime, enable_htp):

    if download:
        print("Downloading COOC Val2007 Images...")
        wget(model_dir, 'http://images.cocodataset.org/zips/val2017.zip')
        try:
            checkResource(model_dir, 'val2017.zip', COCO_VAL2007_ARCHIVE_CHECKSUM)
        except Exception as err:
            sys.stderr.write('ERROR: %s\n' % str(err))
            sys.exit(0)

    if 'SNPE_ROOT' not in os.environ:
        raise RuntimeError('SNPE_ROOT not setup.  Please run the SDK env setup script.')
    snpe_root = os.path.abspath(os.environ['SNPE_ROOT'])
    if not os.path.isdir(snpe_root):
        raise RuntimeError('SNPE_ROOT (%s) is not a dir' % snpe_root)


    # yolov5s.dlc -> /dlc/
    print('Creating DLC...')
    dlc_dir = os.path.join(model_dir, 'dlc')
    if not os.path.isdir(dlc_dir):
        os.makedirs(dlc_dir)
    # Could use https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt to convert weights
    # But is too much overhead, so we store the models
    onnx_dir = os.path.join(model_dir, 'onnx')
    model_path = os.path.join(onnx_dir, YOLO_V5_ONNX_FILENAME)
    model = onnx.load(model_path)
    graphInput = MessageToDict(model.graph.input[0])
    name = graphInput.get("name") 
    input_shape = [d.get("dimValue") for d in graphInput.get("type").get("tensorType").get("shape").get("dim")]
    cmd = ['snpe-onnx-to-dlc',
           '--input_network', model_path,
           '--input_dim', name.translate({}), ','.join(input_shape),
           '--output_path', os.path.join(dlc_dir, YOLO_V5_DLC_FILENAME)]
    subprocess.call(cmd)

    # val2017.zip images -> /data/
    print('Getting COCO val2017 data...')
    data_dir = os.path.join(model_dir, 'data')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    cmd = ['unzip', '-j', 'val2017.zip', 'val2017/*', '-d', data_dir]
    subprocess.call(cmd)


    print('Create SNPE YOLOv5 input')
    inception_scripts_dir = os.path.join(snpe_root, 'models', 'inception_v3', 'scripts')
    create_raws_script = os.path.join(inception_scripts_dir, 'create_inceptionv3_raws.py')
    data_cropped_dir = os.path.join(data_dir, 'cropped')
    if not os.path.isdir(data_cropped_dir):
        os.makedirs(data_cropped_dir)
    cmd = ['python', create_raws_script,
           '-i', data_dir,
           '-d', data_cropped_dir,
           '-s', "640"]
    subprocess.call(cmd)



    print('Create file lists')
    create_file_list_script = os.path.join(inception_scripts_dir, 'create_file_list.py')
     # list of images -> raw_list.txt  ( /home/admin/yolov5_model_snpe/data/image.raw )
    cmd = ['python', create_file_list_script,
           '-i', data_cropped_dir,
           '-o', os.path.join(data_cropped_dir, 'raw_list.txt'),
           '-e', '*.raw']
    subprocess.call(cmd)
    # list of images -> target_raw_list.txt ( data/image.raw )
    cmd = ['python', create_file_list_script,
           '-i', data_cropped_dir,
           '-o', os.path.join(data_dir, 'target_raw_list.txt'),
           '-e', '*.raw',
           '-r']
    subprocess.call(cmd)

    
    if ('dsp' == runtime or 'aip' == runtime or 'all' == runtime):
        cmd = ['snpe-dlc-quantize',
               '--input_dlc', os.path.join(dlc_dir, YOLO_V5_DLC_FILENAME),
               '--input_list', os.path.join(data_cropped_dir, 'raw_list.txt'),
               '--output_dlc', os.path.join(dlc_dir, YOLO_V5_DLC_QUANTIZED_FILENAME)]
        if (enable_htp):
            cmd.append('--enable_htp')
        subprocess.call(cmd)

    print('Setup YOLOv5 completed.')

def getArgs():

    parser = argparse.ArgumentParser(
        prog=__file__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=
        '''Prepares the YOLOv5 assets for Engine Inferences.''')

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('-a', '--assets_dir', type=str, required=True,
                        help='directory containing the YOLOv5 assets')
    optional.add_argument('-d', '--download', action="store_true", required=False,
                        help='Download YOLOv5 assets to YOLOv5 assets directory')
    optional.add_argument('-r', '--runtime', type=str, required=False,
                        help='Choose a runtime to set up tutorial for. Choices: cpu, gpu, dsp, aip, all. \'all\'')
    optional.add_argument('-l', '--htp', action="store_true", required=False,
                        help='Offline prepare quantized model for htp')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = getArgs()

    try:
        setup_assets(args.assets_dir, args.download, args.runtime, args.htp)
    except Exception as err:
        sys.stderr.write('ERROR: %s\n' % str(err))