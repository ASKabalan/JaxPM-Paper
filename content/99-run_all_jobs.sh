#!/bin/bash

# =============================================================================
# Run Gradient Stability & Memory Benchmarking
# =============================================================================
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm 01-gradient-stability.py -t grad
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm 01-gradient-stability.py -t mem
# =============================================================================
# Run Benchmarking
# =============================================================================
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm 02-jaxpm-benchmarks.py -m 64 128 256 512 -b 128.0 256.0 512.0 1024.0 -a REVERSE RECURSIVE -n 10
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm 03-pmwd-benchmarks.py  -m 64 128 256 512 -b 128.0 256.0 512.0 1024.0 -n 10
