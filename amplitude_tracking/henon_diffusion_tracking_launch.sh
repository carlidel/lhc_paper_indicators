#!/bin/bash

# List of long commands
commands=(
    "python3 henon_diffusion_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lhc_injection_0.h5\" --omega_x 0.28 --omega_y 0.31 --epsilon 0.0 --r_max 0.5 --gpu 0"
    "python3 henon_diffusion_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lhc_injection_8.h5\" --omega_x 0.28 --omega_y 0.31 --epsilon 8.0 --r_max 0.5 --gpu 1"
    "python3 henon_diffusion_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lhc_injection_16.h5\" --omega_x 0.28 --omega_y 0.31 --epsilon 16.0 --r_max 0.5 --gpu 2"
    "python3 henon_diffusion_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lhc_injection_32.h5\" --omega_x 0.28 --omega_y 0.31 --epsilon 32.0 --r_max 0.5 --gpu 3"
    "python3 henon_diffusion_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lhc_injection_64.h5\" --omega_x 0.28 --omega_y 0.31 --epsilon 64.0 --r_max 0.5 --gpu 0"
    "python3 henon_diffusion_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lhc_injection_4.h5\" --omega_x 0.28 --omega_y 0.31 --epsilon 4.0 --r_max 0.5 --gpu 1"
    "python3 henon_diffusion_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lhc_injection_24.h5\" --omega_x 0.28 --omega_y 0.31 --epsilon 24.0 --r_max 0.5 --gpu 2"
    "python3 henon_diffusion_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lhc_injection_42.h5\" --omega_x 0.28 --omega_y 0.31 --epsilon 42.0 --r_max 0.5 --gpu 3"
    "python3 henon_diffusion_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lhc_top_0.h5\" --omega_x 0.31 --omega_y 0.32 --epsilon 0.0 --r_max 0.22 --gpu 0"
    "python3 henon_diffusion_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lhc_top_8.h5\" --omega_x 0.31 --omega_y 0.32 --epsilon 8.0 --r_max 0.22 --gpu 1"
    "python3 henon_diffusion_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lhc_top_16.h5\" --omega_x 0.31 --omega_y 0.32 --epsilon 16.0 --r_max 0.22 --gpu 2"
    "python3 henon_diffusion_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lhc_top_32.h5\" --omega_x 0.31 --omega_y 0.32 --epsilon 32.0 --r_max 0.22 --gpu 3"
    "python3 henon_diffusion_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lhc_top_64.h5\" --omega_x 0.31 --omega_y 0.32 --epsilon 64.0 --r_max 0.22 --gpu 0"
    "python3 henon_diffusion_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lhc_top_4.h5\" --omega_x 0.31 --omega_y 0.32 --epsilon 4.0 --r_max 0.22 --gpu 1"
    "python3 henon_diffusion_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lhc_top_24.h5\" --omega_x 0.31 --omega_y 0.32 --epsilon 24.0 --r_max 0.22 --gpu 2"
    "python3 henon_diffusion_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lhc_top_42.h5\" --omega_x 0.31 --omega_y 0.32 --epsilon 42.0 --r_max 0.22 --gpu 3"
)

# Number of commands to execute in each batch
batch_size=4

# Calculate the number of batches
num_batches=$(( ${#commands[@]} / batch_size ))

# Loop through batches
for ((batch=0; batch<num_batches; batch++)); do
    start_idx=$((batch * batch_size))

    # Execute commands in the current batch in the background
    for ((i=0; i<batch_size; i++)); do
        idx=$((start_idx + i))
        eval "${commands[$idx]}" &
    done

    # Wait for background processes to finish
    wait
done