#!/bin/bash

# =============================================================================
# Run Gradient Stability & Memory Benchmarking
# =============================================================================
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm 01-gradient-stability.py -t grad
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm 01-gradient-stability.py -t mem
# =============================================================================
# Run Benchmarking
# =============================================================================
# Single GPU
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm 02-jaxpm-benchmarks.py -m 64 128 256 512 -b 128.0 256.0 512.0 1024.0 -a REVERSE RECURSIVE -n 10
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm 03-pmwd-benchmarks.py  -m 64 128 256 512 -b 128.0 256.0 512.0 1024.0 -n 10
# Single Node Multi-GPU
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:4 --tasks-per-node=4 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm 02-jaxpm-benchmarks.py -m 64 128 256 512 -b 128.0 256.0 512.0 1024.0 -a REVERSE RECURSIVE -n 10 -p 1 4
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm 02-jaxpm-benchmarks.py -m 64 128 256 512 -b 128.0 256.0 512.0 1024.0 -a REVERSE RECURSIVE -n 10 -p 1 8
# Multi Node Multi-GPU
sbatch --account=tkc@a100 --nodes=2 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm 02-jaxpm-benchmarks.py -m 64 128 256 512 -b 128.0 256.0 512.0 1024.0 -a REVERSE RECURSIVE -n 10 -p 2 8
sbatch --account=tkc@a100 --nodes=4 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm 02-jaxpm-benchmarks.py -m 64 128 256 512 -b 128.0 256.0 512.0 1024.0 -a REVERSE RECURSIVE -n 10 -p 4 8
sbatch --account=tkc@a100 --nodes=8 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm 02-jaxpm-benchmarks.py -m 64 128 256 512 -b 128.0 256.0 512.0 1024.0 -a REVERSE RECURSIVE -n 10 -p 8 8