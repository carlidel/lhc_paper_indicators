"""Create initial conditions for the amplitude tracking model.
"""

import json
import os
import pickle

import numpy as np

FROM_I = 10.0
TO_I = 15.0
I_SAMPLES = 21
ANGLE_SAMPLES = 500
# LOCAL_DISPLACEMENTS = ["x", "px", "y", "py"]
LOCAL_DISPLACEMENTS = []
DISPLACEMENT_AMPLITUDE = 1e-8

# main
if __name__ == "__main__":
    I = np.linspace(FROM_I, TO_I, I_SAMPLES)
    angles = np.linspace(0.0, 2.0 * np.pi, ANGLE_SAMPLES)

    x = np.zeros((I_SAMPLES, ANGLE_SAMPLES))
    px = np.zeros((I_SAMPLES, ANGLE_SAMPLES))
    y = np.zeros((I_SAMPLES, ANGLE_SAMPLES))
    py = np.zeros((I_SAMPLES, ANGLE_SAMPLES))
    zeta = np.zeros((I_SAMPLES, ANGLE_SAMPLES))
    pzeta = np.zeros((I_SAMPLES, ANGLE_SAMPLES))

    for i in range(I_SAMPLES):
        for j in range(ANGLE_SAMPLES):
            x[i, j] = I[i] * np.cos(angles[j])
            y[i, j] = I[i] * np.sin(angles[j])

    amplitude = np.sqrt(x**2 + px**2 + y**2 + py**2)

    x = x.flatten()
    px = px.flatten()
    y = y.flatten()
    py = py.flatten()
    zeta = zeta.flatten()
    pzeta = pzeta.flatten()
    amplitude = amplitude.flatten()

    # concatenate local displacements
    if len(LOCAL_DISPLACEMENTS) > 0:
        x_base = x.copy()
        px_base = px.copy()
        y_base = y.copy()
        py_base = py.copy()
        zeta_base = zeta.copy()
        pzeta_base = pzeta.copy()
        amplitude_base = amplitude.copy()

        for local_displacement in LOCAL_DISPLACEMENTS:
            x = np.concatenate(
                (
                    x,
                    x_base
                    + (DISPLACEMENT_AMPLITUDE if local_displacement == "x" else 0.0),
                )
            )
            px = np.concatenate(
                (
                    px,
                    px_base
                    + (DISPLACEMENT_AMPLITUDE if local_displacement == "px" else 0.0),
                )
            )
            y = np.concatenate(
                (
                    y,
                    y_base
                    + (DISPLACEMENT_AMPLITUDE if local_displacement == "y" else 0.0),
                )
            )
            py = np.concatenate(
                (
                    py,
                    py_base
                    + (DISPLACEMENT_AMPLITUDE if local_displacement == "py" else 0.0),
                )
            )
            zeta = np.concatenate(
                (
                    zeta,
                    zeta_base
                    + (DISPLACEMENT_AMPLITUDE if local_displacement == "zeta" else 0.0),
                )
            )
            pzeta = np.concatenate(
                (
                    pzeta,
                    pzeta_base
                    + (
                        DISPLACEMENT_AMPLITUDE if local_displacement == "pzeta" else 0.0
                    ),
                )
            )

            amplitude = np.concatenate((amplitude, amplitude_base))

    # save initial conditions
    initial_conditions = {
        "x_norm": x.tolist(),
        "px_norm": px.tolist(),
        "y_norm": y.tolist(),
        "py_norm": py.tolist(),
        "zeta_norm": zeta.tolist(),
        "pzeta_norm": pzeta.tolist(),
    }
    appendix = {
        "I": I.tolist(),
        "angles": angles.tolist(),
        "amplitude": amplitude.tolist(),
    }

    with open(
        "/home/HPC/camontan/lhc_paper_indicators/amplitude_tracking/config/particles.json",
        "w",
    ) as f:
        json.dump(initial_conditions, f, indent=4)

    with open(
        "/home/HPC/camontan/lhc_paper_indicators/amplitude_tracking/config/appendix.json",
        "w",
    ) as f:
        json.dump(appendix, f, indent=4)
