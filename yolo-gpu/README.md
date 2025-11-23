# YOLOv4-P5 (GPU)
YOLOv4-P5 (object detection reference application), based on [this repository](https://github.com/WongKinYiu/ScaledYOLOv4), optimised for GPU.

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| PyTorch | Vision | YOLOv4-P5 | COCO 2017 | Object detection | <p style="text-align: center;">❌ | <p style="text-align: center;">✅ <br> GPU required | [Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036) |


## Instructions summary
1. Install the system and Python requirements (see Environment setup)

2. Download the COCO 2017 dataset (See Dataset setup)


## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Install the Python requirements:
```bash
pip3 install -r requirements.txt
```

## Dataset setup

### COCO 2017
Download the COCO 2017 dataset from [the source](http://images.cocodataset.org/zips/) or [via kaggle](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset), or via the script we provide:
```bash
bash utils/download_coco_dataset.sh
```

Additionally, also download  and unzip the labels:
```bash
curl -L https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip -o coco2017labels.zip && unzip -q coco2017labels.zip -d '<dataset path>' && rm coco2017labels.zip
```

Disk space required: 26G

```bash
.
├── LICENSE
├── README.txt
├── annotations
├── images
├── labels
├── test-dev2017.txt
├── train2017.cache
├── train2017.txt
├── val2017.cache
└── val2017.txt

3 directories, 7 files
```

## Custom inference

### Inference with pre-trained weights
To download the pretrained weights, run the following commands:
```bash
mkdir weights
cd weights
curl https://gc-demo-resources.s3.us-west-1.amazonaws.com/yolov4_p5_reference_weights.tar.gz -o yolov4_p5_reference_weights.tar.gz && tar -zxvf yolov4_p5_reference_weights.tar.gz && rm yolov4_p5_reference_weights.tar.gz
cd ..
```
These weights are derived from the a pre-trained model shared by the [YOLOv4's author](https://github.com/WongKinYiu/ScaledYOLOv4). We have post-processed these weights to remove the model description and leave a state_dict compatible with the model description.

To run:
```bash
python3 run.py --weights weights/yolov4_p5_reference_weights/yolov4-p5-sd.pt
```

### Inference without pre-trained weights

```console
python run.py
```
`run.py` will use the default config defined in `configs/inference-yolov4p5.yaml` which can be overridden by various arguments (`python run.py --help` for more info)

### Evaluation

To compute evaluation metrics run:
```bash
python run.py --weights '/path/to/your/pretrain_weights.pt' --obj-threshold 0.001 --class-conf-threshold 0.001 --benchmark
```
You can use the `--verbose` flag if you want to print the metrics per class.

</br></br>
