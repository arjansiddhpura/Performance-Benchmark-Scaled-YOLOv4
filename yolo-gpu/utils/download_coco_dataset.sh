# Copyright (c) 2021 Graphcore Ltd. All rights reserved.


#!/bin/bash
# Download and unzip the images for test, validation and training
training_file='train2017.zip'
validation_file='val2017.zip'
test_file='test2017.zip'
url=http://images.cocodataset.org/zips/

directory="$HOME/localdata/datasets/coco/images" # Unzip directory
mkdir -p "$directory" # Ensure directory exists

# Download, unzip and clean files
for image_file in $validation_file $test_file; do # Add $training_file to download training images too
    curl -L $url$image_file -o $image_file && unzip -q $image_file -d $directory && rm $image_file
done

# Download and unzip the labels
curl -L https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip -o coco2017labels.zip && unzip -q coco2017labels.zip -d "$HOME/localdata/datasets/coco/labels" && rm coco2017labels.zip