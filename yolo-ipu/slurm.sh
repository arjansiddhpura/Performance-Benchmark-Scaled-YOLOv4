#!/bin/bash
#SBATCH --job-name=yolo-ipu-benchmark   # Job name
#SBATCH --output=logs/output_%j.log     # Output file (%j will be replaced by job ID)
#SBATCH --error=logs/error_%j.log       # Error file (%j will be replaced by job ID)
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

echo "------------------------------------------------------------------------------------------"
echo "   1. EFFECT OF IMAGE SIZE (HALF PRECISION)   "
echo "------------------------------------------------------------------------------------------"
BATCH_SIZE=1
PRECISION="half" # Set precision to half

# Loop through image sizes from 32 to 1280 in steps of 32
for IMG_SIZE in $(seq 32 32 1280)
do
  # Run the benchmark with the current parameters
  mpirun --np 1 \
         --allow-run-as-root \
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
  mpirun --np 1 \
         --allow-run-as-root \
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
IMAGE_SIZES="128 224 320"
BATCH_SIZES="1 2 4 8 16 32"
PRECISION="half" # Set precision to half

# Outer loop for image sizes
for IMG_SIZE in ${IMAGE_SIZES}
do
  # Inner loop for batch sizes
  for BATCH_SIZE in ${BATCH_SIZES}
  do
    # Run the benchmark
    mpirun --np 1 \
           --allow-run-as-root \
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
IMAGE_SIZES="128 224 320"
BATCH_SIZES="1 2 4 8 16 32"
PRECISION="single" # Set precision to single

# Outer loop for image sizes
for IMG_SIZE in ${IMAGE_SIZES}
do
  # Inner loop for batch sizes
  for BATCH_SIZE in ${BATCH_SIZES}
  do
    # Run the benchmark
    mpirun --np 1 \
           --allow-run-as-root \
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
echo "   5. PROFILING RUN   "
echo "------------------------------------------------------------------------------------------"

export POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true"}'
export PVTI_OPTIONS='{"enable":"true"}'

# Run the benchmark with profiling 
mpirun --np 1 \
        --allow-run-as-root \
        python3 run.py \
          --device-iterations 1 \
          --micro-batch-size 1 \
          --image-size 1088 \
          --precision half \
          --mode test_inference \
          --benchmark \
          --no-eval \
          --profile-dir ./profile

echo "All experiments completed."