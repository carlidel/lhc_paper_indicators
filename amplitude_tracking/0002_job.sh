#!/bin/bash

python3 xsuite_code/run_sim_normed.py --env tracking_files/configs/local_env.json --mask tracking_files/configs/mask_worst.json --tracking amplitude_tracking/config/tracking_1e5.json --kind coordinates --output amplitude_4 --zeta avg --particles amplitude_tracking/config/particles.json