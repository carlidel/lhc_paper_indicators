import argparse
import os

import h5py
import henon_map_cpp as hm
import numpy as np
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
        default=int(1e6),
        help="Number of turns",
    )
    parser.add_argument(
        "--turns_long_term",
        type=int,
        default=int(1e6),
        help="Number of turns for the long term tracking",
    )
    # parser.add_argument(
    #     "--turn_mod_sample",
    #     type=int,
    #     default=100,
    #     help="module to consider for the sampling",
    # )

    # parser.add_argument(
    #     "--part_samples",
    #     type=int,
    #     default=int(1e4),
    #     help="Number of samples for the particles",
    # )
    # parser.add_argument(
    #     "--std_I",
    #     type=float,
    #     default=0.001,
    #     help="Standard deviation of the initial action",
    # )
    # parser.add_argument(
    #     "--std_angle",
    #     type=float,
    #     default=0.001,
    #     help="Standard deviation of the angle",
    # )

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
        # f.attrs["turn_mod_sample"] = args.turn_mod_sample

        # f.attrs["part_samples"] = args.part_samples
        # f.attrs["std_I"] = args.std_I
        # f.attrs["std_angle"] = args.std_angle

        f.create_dataset("r_list", data=r_list)
        f.create_dataset("angle_list", data=angle_list)
        f.create_dataset("radiuses", data=rr)
        f.create_dataset("angles", data=aa)

    tracker = hm.henon_tracker(
        args.turns_long_term, args.omega_x, args.omega_y, "sps", epsilon=args.epsilon
    )

    x = (rr) * np.cos(aa)
    px = np.zeros_like(x)
    y = (rr) * np.sin(aa)
    py = np.zeros_like(y)

    x = x.flatten()
    px = px.flatten()
    y = y.flatten()
    py = py.flatten()

    particles = hm.particles(x, px, y, py, force_CPU=False)

    matrices = hm.matrix_4d_vector(x.size, force_cpu=False)
    construct = hm.lyapunov_birkhoff_construct_multi(x.size, [args.turns])
    vectors_x = hm.vector_4d(np.array([[1.0, 0.0, 0.0, 0.0] for i in range(x.size)]))

    for i in tqdm(range(1, args.turns + 1)):
        vectors_x.normalize()
        matrices.set_with_tracker(tracker, particles, args.mu)
        vectors_x.multiply(matrices)
        construct.add(vectors_x)
        tracker.track(particles, 1, args.mu, 1.0)

    steps = particles.get_steps()

    raw_lyap = construct.get_values_raw()
    b_lyap = construct.get_values_b()

    with h5py.File(args.output, "a") as f:
        f.create_dataset("track_data/lyapunov_x", data=raw_lyap[0], compression="gzip")
        f.create_dataset("track_data/lyapunov_b_x", data=b_lyap[0], compression="gzip")
        f.create_dataset(
            "track_data/steps",
            data=steps,
            compression="gzip",
            compression_opts=9,
        )
        f.create_dataset("track_data/r", data=rr)
        f.create_dataset("track_data/a", data=aa)
