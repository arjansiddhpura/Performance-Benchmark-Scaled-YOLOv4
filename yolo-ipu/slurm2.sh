#!/bin/bash
#SBATCH --job-name=yolo-ipu-benchmark   # Job name
#SBATCH --output=logs/test_output_%j.log     # Output file (%j will be replaced by job ID)
#SBATCH --error=logs/test_error_%j.log       # Error file (%j will be replaced by job ID)
#SBATCH --partition=vipup               # Partition name
#SBATCH --gres=ipu:1                    # Number of IPUs to allocate
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --time=4-00:00:00               # Maximum runtime (HH:MM:SS)
# SBATCH --mem=8G                       # Memory allocation


# Activate virtual environment
source /csghome/gg281/.conda/envs/thesis-gpu/bin/activate

# Load necessary modules
export SDK_PATH="/local/poplar_sdk-ubuntu_20_04-3.4.0+1507-69d9d03fd8"
export VIRTUAL_ENV=/csghome/gg281/.conda/envs/thesis-gpu/bin/python3
export DATASETS_DIR=/localdata/datasets/
source $SDK_PATH/enable

# Load appropriate conda paths and activate the environment
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate thesis

# from yolo-ipu/
mpirun --np 1 --allow-run-as-root \
python3 run.py \
  --config configs/inference-yolov4p5.yaml \
  --data /csghome/gg281/localdata/datasets/ \
  --weights checkpoint/yolov4_p5_reference_weights/yolov4-p5-sd.pt \
  --mode test \
  --class-conf-threshold 0.001 \
  --obj-threshold 0.001 \
  --verbose
