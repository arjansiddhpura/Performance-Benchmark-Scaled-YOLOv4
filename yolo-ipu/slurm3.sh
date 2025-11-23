#!/bin/bash
#SBATCH --job-name=yolo-ipu-sweep           # Job name
#SBATCH --output=logs/output_%j.log         # Output file
#SBATCH --error=logs/error_%j.log           # Error file
#SBATCH --partition=vipup                   # Partition name
#SBATCH --gres=ipu:1                        # Number of IPUs to allocate
#SBATCH --cpus-per-task=1
#SBATCH --time=4-00:00:00                   # Maximum runtime
# SBATCH --mem=8G

# --- Setup ---
echo "Setting up environment..."
source /csghome/gg281/.conda/envs/thesis-gpu/bin/activate
export SDK_PATH="/local/poplar_sdk-ubuntu_20_04-3.4.0+1507-69d9d03fd8"
export VIRTUAL_ENV=/csghome/gg281/.conda/envs/thesis-gpu/bin/python3
export DATASETS_DIR=/localdata/datasets/
source $SDK_PATH/enable
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate thesis
echo "Setup complete."

# --- Experiment Parameters ---
# Define the image and batch sizes to test
IMG_SIZES=(224 320 416 512 640 896)
BATCH_SIZES=(1 2 4 8 16 32 64 128)
PRECISION="half" # Set precision

echo "------------------------------------------------------------------------------------------"
echo "   Starting Automatic Batch Size Sweep (Precision: ${PRECISION})   "
echo "------------------------------------------------------------------------------------------"
echo "" > logs/results.log # Clear previous results log

# --- Main Loop ---
# Outer loop for image sizes
for IMG_SIZE in "${IMG_SIZES[@]}"; do
  
  echo ">>> Testing Image Size: ${IMG_SIZE}x${IMG_SIZE}"
  max_successful_bs=0 # Reset for each new image size

  # Inner loop for batch sizes
  for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    
    echo "  - Trying Batch Size: ${BATCH_SIZE}..."
    
    # Run the benchmark command
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

    # Check the exit code of the last command ($?)
    # A non-zero exit code means the script failed (e.g., OOM error)
    if [ $? -ne 0 ]; then
      echo "  - FAILURE detected at Batch Size: ${BATCH_SIZE}. Stopping tests for this image size."
      break # Exit the inner (batch size) loop
    else
      echo "  - SUCCESS with Batch Size: ${BATCH_SIZE}."
      max_successful_bs=${BATCH_SIZE} # Update the last known good batch size
    fi
  done

  # Log the final result for the current image size
  RESULT_MSG="RESULT for ${IMG_SIZE}x${IMG_SIZE}: Max successful batch size = ${max_successful_bs}"
  echo "${RESULT_MSG}"
  echo "${RESULT_MSG}" >> logs/results.log # Append to a clean results file
  echo "-----------------------------------------------------------------"
done


PRECISION="single" # Set precision

echo "------------------------------------------------------------------------------------------"
echo "   Starting Automatic Batch Size Sweep (Precision: ${PRECISION})   "
echo "------------------------------------------------------------------------------------------"
echo "" > logs/results.log # Clear previous results log

# --- Main Loop ---
# Outer loop for image sizes
for IMG_SIZE in "${IMG_SIZES[@]}"; do
  
  echo ">>> Testing Image Size: ${IMG_SIZE}x${IMG_SIZE}"
  max_successful_bs=0 # Reset for each new image size

  # Inner loop for batch sizes
  for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    
    echo "  - Trying Batch Size: ${BATCH_SIZE}..."
    
    # Run the benchmark command
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

    # Check the exit code of the last command ($?)
    # A non-zero exit code means the script failed (e.g., OOM error)
    if [ $? -ne 0 ]; then
      echo "  - FAILURE detected at Batch Size: ${BATCH_SIZE}. Stopping tests for this image size."
      break # Exit the inner (batch size) loop
    else
      echo "  - SUCCESS with Batch Size: ${BATCH_SIZE}."
      max_successful_bs=${BATCH_SIZE} # Update the last known good batch size
    fi
  done

  # Log the final result for the current image size
  RESULT_MSG="RESULT for ${IMG_SIZE}x${IMG_SIZE}: Max successful batch size = ${max_successful_bs}"
  echo "${RESULT_MSG}"
  echo "${RESULT_MSG}" >> logs/results.log # Append to a clean results file
  echo "-----------------------------------------------------------------"
done

echo "All tests completed. Final results are in logs/results.log"