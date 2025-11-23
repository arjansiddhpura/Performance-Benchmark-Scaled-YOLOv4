#!/bin/bash
#SBATCH --job-name=yolo-gpu-benchmark   # Job name
#SBATCH --output=logs/output_%j.log     # Output file (%j will be replaced by job ID)
#SBATCH --error=logs/error_%j.log       # Error file (%j will be replaced by job ID)
#SBATCH --partition=rivulet             # Partition name
#SBATCH --gres=gpu:1                    # Number of GPUs to allocate
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --time=4-00:00:00               # Maximum runtime (HH:MM:SS)
#SBATCH --exclusive
# SBATCH --mem=8G                       # Memory allocation


# Load CUDA and check versions
spack env activate cuda
spack load cuda@11.8.0
nvcc --version

echo "------------------------------------------------------------------------------------------"
echo "   1. EFFECT OF IMAGE SIZE (HALF PRECISION)   "
echo "------------------------------------------------------------------------------------------"
BATCH_SIZE=1
PRECISION="half" # Set precision to half

# Loop through image sizes from 32 to 1280 in steps of 32
for IMG_SIZE in $(seq 32 32 1280)
do
  # Run the benchmark with the current parameters
  python3 run.py \
    --device-iterations 1 \
    --micro-batch-size ${BATCH_SIZE} \
    --image-size ${IMG_SIZE} \
    --precision ${PRECISION} \
    --mode test_inference \
    --benchmark \
    --no-eval
done

echo "------------------------------------------------------------------------------------------"
echo "   2. EFFECT OF IMAGE SIZE (FULL PRECISION)   "
echo "------------------------------------------------------------------------------------------"
BATCH_SIZE=1
PRECISION="single" # Set precision to single

# Loop through image sizes from 32 to 1280 in steps of 32
for IMG_SIZE in $(seq 32 32 1280)
do
  # Run the benchmark with the current parameters
  python3 run.py \
    --device-iterations 1 \
    --micro-batch-size ${BATCH_SIZE} \
    --image-size ${IMG_SIZE} \
    --precision ${PRECISION} \
    --mode test_inference \
    --benchmark \
    --no-eval
done

echo "------------------------------------------------------------------------------------------"
echo "   3. EFFECT OF BATCH SIZE (HALF PRECISION)   "
echo "------------------------------------------------------------------------------------------"
IMAGE_SIZES="128 224 320 416 512 640 896"
BATCH_SIZES="1 2 4 8 16 32"
PRECISION="half" # Set precision to half

# Outer loop for image sizes
for IMG_SIZE in ${IMAGE_SIZES}
do
  # Inner loop for batch sizes
  for BATCH_SIZE in ${BATCH_SIZES}
  do
    # Run the benchmark command
    python3 run.py \
      --device-iterations 1 \
      --micro-batch-size ${BATCH_SIZE} \
      --image-size ${IMG_SIZE} \
      --precision ${PRECISION} \
      --mode test_inference \
      --benchmark \
      --no-eval
  done
done

echo "------------------------------------------------------------------------------------------"
echo "   4. EFFECT OF BATCH SIZE (FULL PRECISION)   "
echo "------------------------------------------------------------------------------------------"
IMAGE_SIZES="128 224 320 416 512 640 896"
BATCH_SIZES="1 2 4 8 16 32"
PRECISION="single" # Set precision to single

# Outer loop for image sizes
for IMG_SIZE in ${IMAGE_SIZES}
do
  # Inner loop for batch sizes
  for BATCH_SIZE in ${BATCH_SIZES}
  do
    #Run the benchmark command
    python3 run.py \
      --device-iterations 1 \
      --micro-batch-size ${BATCH_SIZE} \
      --image-size ${IMG_SIZE} \
      --precision ${PRECISION} \
      --mode test_inference \
      --benchmark \
      --no-eval
  done
done

echo "------------------------------------------------------------------------------------------"
echo "   5. TEST PROFILING RUN (HALF PRECISION)   "
echo "------------------------------------------------------------------------------------------"
IMG_SIZE=896
BATCH_SIZE=1
PRECISION="half"

# Profiling with nsys
nsys profile --output=profiles/profile_${IMG_SIZE}_${BATCH_SIZE} \
  python3 run.py \
    --device-iterations 1 \
    --micro-batch-size ${BATCH_SIZE} \
    --image-size ${IMG_SIZE} \
    --precision ${PRECISION} \
    --mode test_inference \
    --benchmark \
    --no-eval

echo "All experiments completed."