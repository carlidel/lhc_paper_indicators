#!/bin/bash

# List of long commands
commands=(
    "python3 henon_lyapunov_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lyap_default_track_0.h5\" --epsilon 0.0 --gpu 1"
    "python3 henon_lyapunov_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lyap_default_track_8.h5\" --epsilon 8.0 --gpu 1"
    "python3 henon_lyapunov_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lyap_default_track_16.h5\" --epsilon 16.0 --gpu 2"
    "python3 henon_lyapunov_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lyap_default_track_32.h5\" --epsilon 32.0 --gpu 3"
    "python3 henon_lyapunov_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lyap_default_track_64.h5\" --epsilon 64.0 --gpu 1"
    "python3 henon_lyapunov_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lyap_default_track_4.h5\" --epsilon 4.0 --gpu 1"
    "python3 henon_lyapunov_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lyap_default_track_24.h5\" --epsilon 24.0 --gpu 2"
    "python3 henon_lyapunov_tracking.py --output \"/home/HPC/camontan/lhc_paper_indicators/data/henon_diffusion/lyap_default_track_42.h5\" --epsilon 42.0 --gpu 3"
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