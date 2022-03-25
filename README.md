# YOLOv5_SNPE_Model
A setup toolkit to support the YOLOv5 model on Snapdragon Neural Processing Engine (SNPE)

## Requirements:
- [Qualcomm Neural Processing SDK](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)
- ONNX
- Python 3.6 (hard requirement by SNPE itself)


## Details
This project is attempt to mirror the structure of  model's folder in the SNPE SDK, to provide support to the YOLOv5 model. 
The output of the models can be found in the `dlc` folder. Where as the yolov5[x].dlc is the 32 bit floating point Intermediate Representation (IR) and yolov5[x]_quantized.dlc by default the is int8 IR for DSP runtime.
To learn more about the specifics of the quantization method, I'd advise reading the [SNPE documentation](https://developer.qualcomm.com/sites/default/files/docs/snpe/quantized_models.html).
