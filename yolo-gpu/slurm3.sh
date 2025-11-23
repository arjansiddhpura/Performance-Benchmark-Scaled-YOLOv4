#!/bin/bash
#SBATCH --job-name=yolo-gpu-sweep           # Job name
#SBATCH --output=logs/gpu_output_%j.log     # Standard output file
#SBATCH --error=logs/gpu_error_%j.log       # Standard error file
#SBATCH --partition=rivulet                 # Partition name (use your GPU partition)
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --cpus-per-task=1                   # Request CPU cores
#SBATCH --time=4-00:00:00                   # Maximum runtime
#SBATCH --exclusive                         # Exclusive node allocation
# SBATCH --mem=8G                           # Memory per node

# --- Setup ---
echo "Setting up GPU environment..."
# Activate your shell package manager (e.g., spack or module)
spack env activate cuda
spack load cuda@11.8.0
nvcc --version

# Activate your conda environment
# Ensure your conda is initialized for bash scripts
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate thesis-gpu # Use your specific GPU environment name
echo "Setup complete."

# --- Experiment Parameters ---
IMG_SIZES=(128 224 320 416 512 640 896)
BATCH_SIZES=(1 2 4 8 16 32 64 128)
PRECISIONS=("half" "single")
LOG_FILE="logs/gpu_results.log"

# Clear previous results log
echo "GPU Memory Limit Sweep Results" > ${LOG_FILE}
echo "--------------------------------" >> ${LOG_FILE}
echo "" >> ${LOG_FILE}

# --- Main Loop ---
# Loop over precision settings first
for PRECISION in "${PRECISIONS[@]}"; do

  echo "=========================================================================================="
  echo "   Starting Automatic Batch Size Sweep (Precision: ${PRECISION})   "
  echo "=========================================================================================="
  echo "Precision: ${PRECISION}" >> ${LOG_FILE}

  # Outer loop for image sizes
  for IMG_SIZE in "${IMG_SIZES[@]}"; do
    
    echo ">>> Testing Image Size: ${IMG_SIZE}x${IMG_SIZE}"
    max_successful_bs=0 # Reset for each new image size

    # Inner loop for batch sizes
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
      
      echo "  - Trying Batch Size: ${BATCH_SIZE}..."
      
      # Run the command directly with python3 (no mpirun needed for GPU)
      python3 run.py \
        --device-iterations 1 \
        --micro-batch-size ${BATCH_SIZE} \
        --image-size ${IMG_SIZE} \
        --precision ${PRECISION} \
        --mode test_inference \
        --benchmark \
        --no-eval

      # Check the exit code of the last command ($?)
      # A non-zero exit code usually indicates an Out-of-Memory (OOM) error on GPUs
      if [ $? -ne 0 ]; then
        echo "  - ❌ FAILURE (likely OOM) detected at Batch Size: ${BATCH_SIZE}."
        echo "  - Stopping tests for this image size."
        break # Exit the inner (batch size) loop
      else
        echo "  - ✅ SUCCESS with Batch Size: ${BATCH_SIZE}."
        max_successful_bs=${BATCH_SIZE} # Update the last known good batch size
      fi
    done

    # Log the final result for the current image size and precision
    RESULT_MSG="Image ${IMG_SIZE}x${IMG_SIZE}: Max successful batch size = ${max_successful_bs}"
    echo "${RESULT_MSG}"
    echo "${RESULT_MSG}" >> ${LOG_FILE}
    echo "-----------------------------------------------------------------"
  done
  echo "" >> ${LOG_FILE} # Add a newline between precision results
done

echo "All tests completed. Final results are in ${LOG_FILE}"