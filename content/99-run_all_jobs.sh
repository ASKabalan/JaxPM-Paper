#!/bin/bash

# =============================================================================
# Run Gradient Stability & Memory Benchmarking
# =============================================================================
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=GRAD_STABILITY 99-slurm_runner.slurm 01-gradient-stability.py -t grad
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=MEM_STABILITY 99-slurm_runner.slurm 01-gradient-stability.py -t mem
# =============================================================================
# Run Benchmarking
# =============================================================================
# for steps in 10 20 40 50 90
for steps in 10 20 40
do
    # Single GPU
    sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=JPM_SINGLE_GPU_$steps 99-slurm_runner.slurm 02-jaxpm-benchmarks.py -m 64 128 256 512 -b 128.0 256.0 512.0 1024.0 -a REVERSE RECURSIVE -n $steps
    sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=PMWD_SINGLE_GPU_$steps 99-slurm_runner.slurm 03-pmwd-benchmarks.py  -m 64 128 256 512 -b 128.0 256.0 512.0 1024.0 -n $steps
    # Single Node Multi-GPU
    sbatch --account=tkc@a100 --nodes=1 --gres=gpu:4 --tasks-per-node=4 -C a100 --job-name=JPM_4_GPU_$steps 99-slurm_runner.slurm 02-jaxpm-benchmarks.py -m 64 128 256 512 -b 128.0 256.0 512.0 1024.0 -a REVERSE RECURSIVE -n $steps -p 1 4 -i 5
    sbatch --account=tkc@a100 --nodes=1 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=JPM_8_GPU_$steps 99-slurm_runner.slurm 02-jaxpm-benchmarks.py -m 64 128 256 512 -b 128.0 256.0 512.0 1024.0 -a REVERSE RECURSIVE -n $steps -p 1 8 -i 5
    # Multi Node Multi-GPU
    sbatch --account=tkc@a100 --nodes=2 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=JPM_16_GPU_$steps 99-slurm_runner.slurm 02-jaxpm-benchmarks.py -m 64 128 256 512 1024 -b 128.0 256.0 512.0 1024.0 2048.0 -a REVERSE RECURSIVE -n $steps -p 2 8 -i 5
    sbatch --account=tkc@a100 --nodes=4 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=JPM_32_GPU_$steps 99-slurm_runner.slurm 02-jaxpm-benchmarks.py -m 64 128 256 512 1024 -b 128.0 256.0 512.0 1024.0 2048.0 -a REVERSE RECURSIVE -n $steps -p 4 8 -i 5
    sbatch --account=tkc@a100 --nodes=8 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=JPM_64_GPU_$steps 99-slurm_runner.slurm 02-jaxpm-benchmarks.py -m 64 128 256 512 1024 -b 128.0 256.0 512.0 1024.0 2048.0 -a REVERSE RECURSIVE -n $steps -p 8 8 -i 5
done

# =============================================================================
# Run Inference with Numpyro
# =============================================================================
# Single GPU
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm  04-forward_model.py --box_shape 64 64 128 --box_size 800. 800. 4000. --field_size 9.6 --field_npix 64 --density_plane_width 400 --density_plane_npix 64 --density_plane_smoothing 0.1 -u 100 -c 200 -B 10 -S NUTS
# Single Node Multi-GPU (single process)
sbatch --account=tkc@a100 --nodes=1 --gres=gpu:8 --tasks-per-node=1 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm  04-forward_model.py --box_shape 256 256 512 --box_size 800. 800. 4000. --field_size 38.4 --field_npix 256 --density_plane_width 400 --density_plane_npix 256 --density_plane_smoothing 0.1 -u 100 -c 200 -B 10 -S NUTS -p 1 4
# Multi Node Multi-GPU
sbatch --account=tkc@a100 --nodes=4 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm  04-forward_model.py --box_shape 512 512 1024 --box_size 800. 800. 4000. --field_size 76.8 --field_npix 512 --density_plane_width 400 --density_plane_npix 512 --density_plane_smoothing 0.1 -u 100 -c 200 -B 10 -S NUTS -p 2 8


