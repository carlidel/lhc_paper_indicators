import itertools
import os
import pickle
import sys

import h5py
import numpy as np
from numba import njit

# from clustering_scripts import *
from tqdm import tqdm


@njit
def simple_gali(gali_matrix, normalize):
    if np.any(np.isnan(gali_matrix)):
        return np.nan
    else:
        if normalize:
            for i in len(gali_matrix.shape[0]):
                gali_matrix[i] /= np.sum(gali_matrix[i])
        _, s, _ = np.linalg.svd(gali_matrix)
        return np.prod(s)


@njit
def gali(gali_matrix, normalize=False):
    gali_matrix = np.transpose(gali_matrix, (2, 0, 1))
    gali = []
    for m in gali_matrix:
        gali.append(simple_gali(m, normalize=normalize))
    gali = np.asarray(gali)
    return gali


coord_list = ["x", "px", "y", "py", "zeta", "pzeta"]

gali_2_list = list(itertools.combinations(coord_list, 2))
gali_3_list = list(itertools.combinations(coord_list, 3))
gali_4_list = list(itertools.combinations(coord_list, 4))
gali_5_list = list(itertools.combinations(coord_list, 5))
gali_6_list = list(itertools.combinations(coord_list, 6))

all_gali = gali_2_list + gali_3_list + gali_4_list + gali_5_list + gali_6_list


def gali_hdf5_converter(input_file, output_file, times):
    log_disp_file = h5py.File(input_file, "r")
    gali_out_file = h5py.File(output_file, "w")

    print("loading data")
    dataset_dict = {
        f"{a}_{b}_{c}": log_disp_file[f"disp/{a}_norm/normed_direction/{b}_norm/{c}"][:]
        for a, b, c in tqdm(list(itertools.product(coord_list, coord_list, times)))
    }
    print("evaluating gali")
    for i, combo in enumerate(tqdm(all_gali)):
        for t in tqdm(times):
            dataset_name = f"gali{len(combo)}/{'_'.join(combo)}/{t}"
            # check if dataset exists, if so, skip
            if dataset_name in gali_out_file:
                continue

            gali_matrix = np.asarray(
                [[dataset_dict[f"{a}_{x}_{t}"] for x in coord_list] for a in combo]
            )
            disp = gali(gali_matrix)

            gali_out_file.create_dataset(dataset_name, data=disp)


tr_only_coord_list = ["x", "px", "y", "py"]
tr_only_gali_2_list = list(itertools.combinations(tr_only_coord_list, 2))
tr_only_gali_3_list = list(itertools.combinations(tr_only_coord_list, 3))
tr_only_gali_4_list = list(itertools.combinations(tr_only_coord_list, 4))

tr_only_gali = tr_only_gali_2_list + tr_only_gali_3_list + tr_only_gali_4_list


def tr_only_gali_hdf5_converter(input_file, output_file, times):
    log_disp_file = h5py.File(input_file, "r")
    gali_out_file = h5py.File(output_file, "w")

    print("loading data")
    dataset_dict = {
        f"{a}_{b}_{c}": log_disp_file[f"disp/{a}_norm/normed_direction/{b}_norm/{c}"][:]
        for a, b, c in tqdm(
            list(itertools.product(tr_only_coord_list, tr_only_coord_list, times))
        )
    }
    print("evaluating gali")
    for i, combo in enumerate(tqdm(tr_only_gali)):
        for t in tqdm(times):
            dataset_name = f"gali{len(combo)}/{'_'.join(combo)}/{t}"
            # check if dataset exists, if so, skip
            if dataset_name in gali_out_file:
                continue

            gali_matrix = np.asarray(
                [
                    [dataset_dict[f"{a}_{x}_{t}"] for x in tr_only_coord_list]
                    for a in combo
                ]
            )
            disp = gali(gali_matrix, normalize=True)

            gali_out_file.create_dataset(dataset_name, data=disp)


if __name__ == "__main__":
    OUTDIR = "../data/"

    lattice_list = ["b1_worst", "b1_best"]
    extent_list = [
        np.array([0.0, 14.5, 0.0, 15.5]),
        np.array([0.0, 15.0, 0.0, 17.5]),
    ]
    lattice_name_list = ["Worst", "Best"]
    zeta_list = ["zeta_min", "zeta_avg", "zeta_max"]
    zeta_name_list = ["0.0", "0.0065", "0.0130"]

    l_list = []
    l_name_list = []
    e_list = []
    z_list = []
    z_name_list = []
    stability_list = []
    mask_list = []

    for (lattice, extent, l_name), (zeta, z_name) in itertools.product(
        zip(lattice_list, extent_list, lattice_name_list),
        zip(zeta_list, zeta_name_list),
    ):
        l_list.append(lattice)
        l_name_list.append(l_name)
        e_list.append(extent)
        z_list.append(zeta)
        z_name_list.append(z_name)
        # stability_file = h5py.File(
        #     os.path.join(OUTDIR, f"stability_{lattice}_zeta_{zeta}.h5"), "r"
        # )
        log_disp_file = h5py.File(
            os.path.join(OUTDIR, f"log_displacement_{lattice}_{zeta}.h5"), "r"
        )
        t = 100000
        # stability = stability_file["stability"][:]
        # stability_list.append(stability)
        # mask = np.log10(stability) == 5
        # mask_list.append(mask)

        times = np.array(
            sorted(int(j) for j in log_disp_file["disp/x_norm/log_disp"].keys())
        )

    for i, (lattice, zeta) in tqdm(
        enumerate(itertools.product(lattice_list, zeta_list))
    ):
        gali_hdf5_converter(
            os.path.join(OUTDIR, f"log_displacement_{lattice}_{zeta}.h5"),
            os.path.join(OUTDIR, f"gali_{lattice}_{zeta}.h5"),
            times,
        )
