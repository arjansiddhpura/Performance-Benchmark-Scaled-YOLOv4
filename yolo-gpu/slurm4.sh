#!/bin/bash
#SBATCH --job-name=yolo-gpu-timing              # Job name
#SBATCH --output=logs/gpu_timing_%j.log         # Standard output file
#SBATCH --error=logs/gpu_timing_error_%j.log    # Standard error file
#SBATCH --partition=rivulet                     # Partition name (use your GPU partition)
#SBATCH --gres=gpu:1                            # Request 1 GPU
#SBATCH --cpus-per-task=1                       # Request CPU cores
#SBATCH --time=1-00:00:00                       # Maximum runtime
#SBATCH --exclusive                             # Exclusive node allocation
# SBATCH --mem=8G                               # Memory per node

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
# Fixed parameters for this timing experiment
readonly PRECISION="half"
readonly BATCH_SIZE=1

# Array of image sizes to iterate through
readonly IMG_SIZES=(128 224 320 416 512 640 896 1024)
readonly LOG_FILE="logs/gpu_timing_results.log"

# --- Main Logic ---
echo "=========================================================================================="
echo "   Starting Inference Timing Sweep"
echo "   Precision: ${PRECISION}"
echo "   Batch Size: ${BATCH_SIZE}"
echo "=========================================================================================="

# Create a clean log file with a header
echo "Inference Wall-Clock Time Results (in seconds)" > ${LOG_FILE}
echo "Precision: ${PRECISION}, Batch Size: ${BATCH_SIZE}" >> ${LOG_FILE}
echo "------------------------------------------------" >> ${LOG_FILE}
echo "Image Size | Time (s)" >> ${LOG_FILE}
echo "-----------|----------" >> ${LOG_FILE}

# Loop over all specified image sizes
for IMG_SIZE in "${IMG_SIZES[@]}"; do

  echo ">>> Timing for image size: ${IMG_SIZE}x${IMG_SIZE}"
  
  # Capture start time with high precision (seconds.nanoseconds)
  start_time=$(date +%s.%N)

  # Run the inference script
  python3 run.py \
    --device-iterations 1 \
    --micro-batch-size ${BATCH_SIZE} \
    --image-size ${IMG_SIZE} \
    --precision ${PRECISION} \
    --mode test_inference \
    --benchmark \
    --no-eval

  # Capture end time
  end_time=$(date +%s.%N)

  # Calculate the duration using 'bc' for floating-point arithmetic
  duration=$(echo "$end_time - $start_time" | bc)

  # Format the result for logging
  RESULT_MSG="${IMG_SIZE}x${IMG_SIZE} | ${duration}"
  
  # Print to console and append to the log file
  echo "  - Total time: ${duration} seconds."
  echo "${RESULT_MSG}" >> ${LOG_FILE}

done

echo ""
echo "All timing tests completed."
echo "Final results are in ${LOG_FILE}"