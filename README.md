# Feasibility and Performance Analysis of Scaled-YOLOv4 Object Detection on Graphcore IPU

This repository contains the source code and data.
The project presents a comprehensive performance benchmark of the Scaled-YOLOv4-P5 object detection model on a Graphcore GC200 IPU against a comparable NVIDIA A30 GPU. It investigates performance trade-offs by analyzing inference latency and throughput while varying image size, batch size, and floating-point precision.

## Repository Structure

The repository is organized into the following directories:

-   **`yolo-gpu/`**: Contains the implementation and benchmarking scripts for the NVIDIA GPU.

    -   See [yolo-gpu/README.md](yolo-gpu/README.md) for setup and usage instructions.

-   **`yolo-ipu/`**: Contains the implementation and benchmarking scripts for the Graphcore IPU.

    -   See [yolo-ipu/README.md](yolo-ipu/README.md) for setup and usage instructions (requires Poplar SDK).

-   **`data/`**: Contains the raw CSV data collected during the evaluation and validation phases.

    -   Includes compilation times, runtime metrics, and accuracy results for different batch sizes and precisions.

-   **`paper/`**: Contains the LaTeX source code for the paper/thesis.
    -   `Abstract_submission.tex`: The abstract of the paper.
    -   `paper3.tex`: The main paper content.
    -   `plots.tex`, `tables.tex`: Supplementary LaTeX files for figures and tables.

## Key Findings

As detailed in the abstract:

-   **Latency**: The IPU excels in low-latency scenarios (e.g., ~4x faster than GPU at batch size 1 for 896px images).
-   **Throughput & Memory**: The GPU scales better for high-throughput tasks due to larger HBM2 memory, whereas the IPU is constrained by its on-chip SRAM (limiting batch sizes).
-   **Compilation**: The IPU's Ahead-of-Time (AOT) compilation introduces significant overhead compared to the GPU's Just-in-Time (JIT) execution.

## Author

**Arjan Siddhpura and Kazem Shekofteh**  
_Hardware and Artificial Intelligence (HAWAII) Lab,  
Heidelberg University_
