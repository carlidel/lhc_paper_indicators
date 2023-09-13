import argparse
import datetime
import os

import h5py
import henon_map_cpp as hm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
from tqdm import tqdm

if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(description="Henon map tracking")
    # add arguments to the parser
    parser.add_argument(
        "--omega_x",
        type=float,
        default=0.168,
        help="Frequency of the x plane",
    )
    parser.add_argument(
        "--omega_y",
        type=float,
        default=0.201,
        help="Frequency of the x plane",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=32.0,
        help="Strength of the modulation",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.0,
        help="Strength of the octupolar kick",
    )

    parser.add_argument(
        "--r_min",
        type=float,
        default=0.1,
        help="Minimum value of the radius",
    )
    parser.add_argument(
        "--r_max",
        type=float,
        default=0.5,
        help="Maximum value of the radius",
    )
    parser.add_argument(
        "--r_samples",
        type=int,
        default=100,
        help="Number of samples for the radius",
    )
    parser.add_argument(
        "--angle_min",
        type=float,
        default=0.0,
        help="Minimum value of the angle",
    )
    parser.add_argument(
        "--angle_max",
        type=float,
        default=0.5 * np.pi,
        help="Maximum value of the angle",
    )
    parser.add_argument(
        "--angle_samples",
        type=int,
        default=50,
        help="Number of samples for the angle",
    )

    parser.add_argument(
        "--turns",
        type=int,
        default=int(1e5),
        help="Number of turns",
    )
    parser.add_argument(
        "--turns_long_term",
        type=int,
        default=int(1e5),
        help="Number of turns for the long term tracking",
    )
    parser.add_argument(
        "--turn_mod_sample",
        type=int,
        default=100,
        help="module to consider for the sampling",
    )

    parser.add_argument(
        "--part_samples",
        type=int,
        default=int(1e4),
        help="Number of samples for the particles",
    )
    parser.add_argument(
        "--std_I",
        type=float,
        default=0.001,
        help="Standard deviation of the initial action",
    )
    parser.add_argument(
        "--std_angle",
        type=float,
        default=0.001,
        help="Standard deviation of the angle",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="HDF5 output file",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to be used as CUDA_VISIBLE_DEVICES",
    )

    # parse the arguments
    args = parser.parse_args()

    # set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    r_list = np.linspace(args.r_min, args.r_max, args.r_samples)
    angle_list = np.linspace(args.angle_min, args.angle_max, args.angle_samples)

    rr, aa = np.meshgrid(r_list, angle_list)
    rr = rr.flatten()
    aa = aa.flatten()

    # if the output file already exists, delete it
    if os.path.exists(args.output):
        os.remove(args.output)

    # save the parameters in the output file
    with h5py.File(args.output, "a") as f:
        f.attrs["omega_x"] = args.omega_x
        f.attrs["omega_y"] = args.omega_y
        f.attrs["epsilon"] = args.epsilon
        f.attrs["mu"] = args.mu

        f.attrs["r_min"] = args.r_min
        f.attrs["r_max"] = args.r_max
        f.attrs["r_samples"] = args.r_samples
        f.attrs["angle_min"] = args.angle_min
        f.attrs["angle_max"] = args.angle_max
        f.attrs["angle_samples"] = args.angle_samples

        f.attrs["turns"] = args.turns
        f.attrs["turns_long_term"] = args.turns_long_term
        f.attrs["turn_mod_sample"] = args.turn_mod_sample

        f.attrs["part_samples"] = args.part_samples
        f.attrs["std_I"] = args.std_I
        f.attrs["std_angle"] = args.std_angle

        f.create_dataset("r_list", data=r_list)
        f.create_dataset("angle_list", data=angle_list)
        f.create_dataset("radiuses", data=rr)
        f.create_dataset("angles", data=aa)

    tracker = hm.henon_tracker(
        args.turns_long_term, args.omega_x, args.omega_y, "sps", epsilon=args.epsilon
    )

    for i, (r, a) in enumerate(zip(tqdm(rr), aa)):
        spread_I = np.random.normal(0, args.std_I, args.part_samples)
        spread_angles_x = np.random.normal(0, args.std_angle, args.part_samples)
        spread_angles_y = np.random.normal(0, args.std_angle, args.part_samples)

        x = np.sqrt(r**2 + spread_I) * np.cos(a) * np.cos(spread_angles_x)
        px = np.sqrt(r**2 + spread_I) * np.cos(a) * np.sin(spread_angles_x)
        y = np.sqrt(r**2 + spread_I) * np.sin(a) * np.cos(spread_angles_y)
        py = np.sqrt(r**2 + spread_I) * np.sin(a) * np.sin(spread_angles_y)

        x = x.flatten()
        px = px.flatten()
        y = y.flatten()
        py = py.flatten()

        particles = hm.particles(x, px, y, py, force_CPU=False)

        I_std = np.ones(args.turns // args.turn_mod_sample + 1) * np.nan
        angle_x_std = np.ones(args.turns // args.turn_mod_sample + 1) * np.nan
        angle_y_std = np.ones(args.turns // args.turn_mod_sample + 1) * np.nan
        time_samples = np.ones(args.turns // args.turn_mod_sample + 1) * np.nan

        I_std[0] = particles.get_action_std()
        angle_x_std[0] = particles.get_angle_x_std()
        angle_y_std[0] = particles.get_angle_y_std()
        time_samples[0] = 0

        for idx, j in enumerate(range(1, args.turns + 1, args.turn_mod_sample)):
            tracker.track(particles, args.turn_mod_sample, mu=args.mu, barrier=1.0)

            I_std[idx + 1] = particles.get_action_std()
            angle_x_std[idx + 1] = particles.get_angle_x_std()
            angle_y_std[idx + 1] = particles.get_angle_y_std()
            time_samples[idx + 1] = j

        delta_turns = -args.turns + args.turns_long_term
        if delta_turns > 0:
            tracker.track(particles, delta_turns, mu=args.mu, barrier=1.0)

        steps = particles.get_steps()

        with h5py.File(args.output, "a") as f:
            f.create_dataset(
                f"track_data/{i}/I_std",
                data=I_std,
                compression="gzip",
                compression_opts=9,
            )
            f.create_dataset(
                f"track_data/{i}/angle_x_std",
                data=angle_x_std,
                compression="gzip",
                compression_opts=9,
            )
            f.create_dataset(
                f"track_data/{i}/angle_y_std",
                data=angle_y_std,
                compression="gzip",
                compression_opts=9,
            )
            f.create_dataset(
                f"track_data/{i}/time_samples",
                data=time_samples,
                compression="gzip",
                compression_opts=9,
            )
            f.create_dataset(
                f"track_data/{i}/steps",
                data=steps,
                compression="gzip",
                compression_opts=9,
            )
            f.create_dataset(f"track_data/{i}/r", data=r)
            f.create_dataset(f"track_data/{i}/a", data=a)
