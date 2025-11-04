# Benchmark Public Cloud Results

This repository contains scripts for analyzing and visualizing benchmark results for Tinfoil model inference services collected using [tinfoilsh/benchmark_public_cloud](https://github.com/tinfoilsh/benchmark_public_cloud).

## Overview

The repository includes Python scripts that generate comparative charts and reports for different performance metrics:
- Inter-token latency (ITL)
- Request latency (TTFT and E2E)
- Throughput (input/output tokens per second)
- Network-adjusted latency (E2E latency with simulated network conditions)

Each script processes JSON result files and produces PDF reports with visual comparisons between models running with and without confidential computing (CC vs No-CC).

## Usage

1. Place your benchmark results in the `results_gcp` directory
2. Run the analysis scripts:
   ```bash
   python plot_itl.py                 # For inter-token latency analysis
   python plot_latency.py             # For request latency analysis
   python plot_throughput.py          # For throughput analysis
   python plot_latency_network.py     # For network-adjusted latency analysis
   ```

Each script will generate a corresponding PDF report with charts and summary statistics.