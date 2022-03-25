# YOLOv5_SNPE_Model
A setup toolkit to support the YOLOv5 model(s) on Snapdragon Neural Processing Engine (SNPE)

## Requirements:
- [Qualcomm Neural Processing SDK](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)
- ONNX
- Python 3.6 (hard requirement by SNPE itself)


## Details
This project attempts to mirror the structure of SNPE SDK's `model` folder and to provide support for the YOLOv5 model(s). 
The outputs of the model(s) can be found in the `dlc` folder. Which are the yolov5[x].dlc file, the 32 bit floating point Intermediate Representation (IR), and yolov5[x]_quantized.dlc file, the int8 IR, which defaultly builds for a DSP runtime.
To learn more about the specifics of the quantization method, I'd advise reading the [SNPE documentation](https://developer.qualcomm.com/sites/default/files/docs/snpe/quantized_models.html).
