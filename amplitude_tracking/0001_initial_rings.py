"""Create initial conditions for the amplitude tracking model.
"""

import json
import os
import pickle

import numpy as np

FROM_I = 5.0
TO_I = 15.0
I_SAMPLES = 21
ANGLE_SAMPLES = 50
RING_SAMPLES_X = 10
RING_SAMPLES_Y = 10

# main
if __name__ == "__main__":
    I = np.linspace(FROM_I, TO_I, I_SAMPLES)
    angles = np.linspace(0.0, 0.5 * np.pi, ANGLE_SAMPLES)
    angles_x = np.linspace(0.0, 2.0 * np.pi, RING_SAMPLES_X)
    angles_y = np.linspace(0.0, 2.0 * np.pi, RING_SAMPLES_X)

    x = np.zeros((I_SAMPLES, ANGLE_SAMPLES, RING_SAMPLES_X, RING_SAMPLES_Y))
    px = np.zeros((I_SAMPLES, ANGLE_SAMPLES, RING_SAMPLES_X, RING_SAMPLES_Y))
    y = np.zeros((I_SAMPLES, ANGLE_SAMPLES, RING_SAMPLES_X, RING_SAMPLES_Y))
    py = np.zeros((I_SAMPLES, ANGLE_SAMPLES, RING_SAMPLES_X, RING_SAMPLES_Y))
    zeta = np.zeros((I_SAMPLES, ANGLE_SAMPLES, RING_SAMPLES_X, RING_SAMPLES_Y))
    pzeta = np.zeros((I_SAMPLES, ANGLE_SAMPLES, RING_SAMPLES_X, RING_SAMPLES_Y))

    amplitude_x = np.zeros((I_SAMPLES, ANGLE_SAMPLES, RING_SAMPLES_X, RING_SAMPLES_Y))
    amplitude_y = np.zeros((I_SAMPLES, ANGLE_SAMPLES, RING_SAMPLES_X, RING_SAMPLES_Y))

    for i in range(I_SAMPLES):
        for j in range(ANGLE_SAMPLES):
            for k in range(RING_SAMPLES_X):
                for l in range(RING_SAMPLES_Y):
                    x[i, j, k, l] = I[i] * np.cos(angles[j]) * np.cos(angles_x[k])
                    px[i, j, k, l] = I[i] * np.cos(angles[j]) * np.sin(angles_x[k])
                    y[i, j, k, l] = I[i] * np.sin(angles[j]) * np.cos(angles_y[l])
                    py[i, j, k, l] = I[i] * np.sin(angles[j]) * np.sin(angles_y[l])

                    amplitude_x[i, j, k, l] = I[i] * np.cos(angles[j])
                    amplitude_y[i, j, k, l] = I[i] * np.sin(angles[j])

    amplitude = np.sqrt(x**2 + px**2 + y**2 + py**2)

    x = x.flatten()
    px = px.flatten()
    y = y.flatten()
    py = py.flatten()
    zeta = zeta.flatten()
    pzeta = pzeta.flatten()
    amplitude_x = amplitude_x.flatten()
    amplitude_y = amplitude_y.flatten()
    amplitude = amplitude.flatten()

    print("x.shape", x.shape)

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
        "amplitude_x": amplitude_x.tolist(),
        "amplitude_y": amplitude_y.tolist(),
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
