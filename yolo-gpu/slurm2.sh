#!/bin/bash
#SBATCH --job-name=yolo-gpu-benchmark   # Job name
#SBATCH --output=logs/output_%j.log     # Output file (%j will be replaced by job ID)
#SBATCH --error=logs/error_%j.log       # Error file (%j will be replaced by job ID)
#SBATCH --partition=rivulet             # Partition name
#SBATCH --gres=gpu:1                    # Number of GPUs to allocate
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --time=4-00:00:00               # Maximum runtime (HH:MM:SS)
# SBATCH --mem=8G                       # Memory allocation

# Load CUDA and check versions
spack env activate cuda
spack load cuda@11.8.0
nvcc --version


echo "------------------------------------------------------------------------------------------"
echo "   1. VALIDATION EVALUATION   "
echo "------------------------------------------------------------------------------------------"
# Run the benchmark
python3 run.py \
  --config configs/inference-yolov4p5.yaml \
  --data /csghome/gg281/localdata/datasets/ \
  --weights checkpoint/yolov4_p5_reference_weights/yolov4-p5-sd.pt \
  --mode test \
  --class-conf-threshold 0.001 \
  --obj-threshold 0.001 \
  --verbose
