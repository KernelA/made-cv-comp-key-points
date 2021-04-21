"""Original code: https://www.kaggle.com/sergemsu/kalman-faces
"""
import pathlib
import math
import os
import logging


import configargparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import interpolate
from matplotlib import pyplot as plt

import face_landmarks
from cli_utils import is_dir

FACE_ZONES = [0, 64, 128, 273, 337, 401, 464, 527, 587, 714, 841, 873, 905, 937, 969]


def angle_variation(ps):
    dps = np.diff(ps, axis=0)
    angles = []
    for i in range(len(dps) - 1):
        e1, e2 = dps[i], dps[i + 1]
        x = np.clip(e1.dot(e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 0.00001), -1, 1)
        angle = math.degrees(math.acos(x))
        angles.append(angle)

    return np.mean(angles)


def kalman(x_observ, Q=1e-5, R=0.0001):
    n = len(x_observ)
    x_hat = np.zeros(n)      # a posteri estimate of x
    P = np.zeros(n)          # a posteri error estimate
    x_hatminus = np.zeros(n)  # a priori estimate of x
    P_minus = np.zeros(n)    # a priori error estimate
    K = np.zeros(n)          # gain or blending factor

    # intial guesses
    x_hat[0] = x_observ[0]
    P[0] = 1.0

    for k in range(1, n):
        # time update
        x_hatminus[k] = x_hat[k-1]
        P_minus[k] = P[k-1]+Q

        # measurement update
        K[k] = P_minus[k] / (P_minus[k] + R)
        x_hat[k] = x_hatminus[k] + K[k]*(x_observ[k] - x_hatminus[k])
        P[k] = (1 - K[k])*P_minus[k]

    return x_hat


def bidir_kalman(x_observ, Q=1e-5, R=0.0003, iters=2):
    n = len(x_observ)

    for _ in range(iters):
        x_forw = kalman(x_observ, Q, R)
        x_back = np.flip(kalman(np.flip(x_observ), Q, R))

        k = 7
        x_full = np.zeros(n+k)
        x_full[:k] = x_forw[:k]
        x_full[k:n] = 0.5 * (x_forw[k:] + x_back[:n-k])
        x_full[n:] = x_back[n-k:]

        f = interpolate.interp1d(np.linspace(0, n+k, n+k), x_full, kind='quadratic')
        x_observ = f(np.linspace(0, n + k, n))

    return x_observ


def fix_landmars(landmark):
    landmark = landmark.astype(float)

    for i in range(1, len(FACE_ZONES)):
        zone = landmark[FACE_ZONES[i-1]:FACE_ZONES[i]]
        x_filt = bidir_kalman(zone[:, 0], iters=1)
        y_filt = bidir_kalman(zone[:, 1], iters=1)
        np.copyto(zone, np.array([x_filt, y_filt]).T)

    return landmark


def fix_faces(image_dir):
    logger = logging.getLogger("kp.main")
    image_dir_pathlib = pathlib.Path(image_dir)

    face_landmarks_path = image_dir_pathlib.parent / "landmarks.csv"

    face_landmarks = pd.read_csv(face_landmarks_path, sep='\t',
                                 engine="c", index_col="file_name")

    image_names = face_landmarks.index.tolist()

    stats = [0] * len(image_names)

    for i, image_name in tqdm(enumerate(image_names), desc="Calculate landmarks stat", total=len(image_names)):
        img_landmark = face_landmarks.loc[image_name].to_numpy().reshape(-1, 2)
        stat1 = angle_variation(img_landmark[:64, :])
        stat2 = angle_variation(img_landmark[64: 128, :])
        stats[i] = (image_name, stat1, stat2)

    stats = pd.DataFrame(data=stats, columns=["file", "stat1", "stat2"]).set_index("file")

    noisy = stats[(stats['stat1'] > 70) & (stats['stat2'] > 70)].index

    logger.info("Detect %d noisy annotations", len(noisy))

    fixed_landmarks = face_landmarks.copy(deep=True)

    is_first = False

    def save_debug_image(input_img_path, landmarks, fixed, out_path):
        image = plt.imread(input_img_path)
        fig = plt.figure()
        axes = fig.subplots(1, 2)
        axes[0].imshow(image)
        axes[0].set_label("Before")
        axes[0].plot(landmarks[:, 0], landmarks[:, 1], "go")
        axes[1].imshow(image)
        axes[1].set_label("After")
        axes[1].plot(fixed[:, 0], fixed[:, 1], "go")
        fig.savefig(out_path)

    for image_name in tqdm(noisy, desc="Fix landmarks", total=len(noisy)):
        landmark = face_landmarks.loc[image_name].to_numpy().reshape(-1, 2)
        fixed = fix_landmars(landmark)
        fixed_landmarks.loc[image_name] = np.round(fixed.reshape(-1)).astype(int)
        if not is_first:
            save_debug_image(image_dir_pathlib / image_name, landmark, fixed,
                             face_landmarks_path.with_name("fix_landmarks_debug.jpg"))
            is_first = True

    fixed_path = face_landmarks_path
    fixed_landmarks.to_csv(fixed_path.with_name(
        "fixed_landmarks.csv"), index=True, sep="\t", encoding="utf-8")

    logger.info("Save new landmarks to '%s'", fixed_path)


def main(args):
    fix_faces(args.image_dir)


if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add_argument("-c", is_config_file=True)
    parser.add_argument("--image_dir", required=True, type=is_dir)

    args = parser.parse_args()

    main(args)
