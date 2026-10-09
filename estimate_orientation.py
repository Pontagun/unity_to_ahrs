import sys

import numpy as np
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick, Mahony, EKF, Complementary, FKF, UKF

FILTER = {"madwick": Madgwick, "mahony": Mahony, "ekf": EKF, "fkf": FKF, "ukf": UKF}


def ned_to_unity(q_list):
    # return q_list
    return [np.array([w, -y, z, -x]) for w, x, y, z in q_list]


def quaternion_error(q1, q2):
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)

    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    # conj(q1) = [w1, -x1, -y1, -z1], Hamilton product with q2
    q_diff = np.stack([
        w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2,
        w1 * x2 - x1 * w2 - y1 * z2 + z1 * y2,
        w1 * y2 + x1 * z2 - y1 * w2 - z1 * x2,
        w1 * z2 - x1 * y2 + y1 * x2 - z1 * w2,
    ], axis=-1)

    # q and -q are the same rotation; pick the shortest path
    q_diff = np.where(q_diff[..., :1] < 0, -q_diff, q_diff)
    angle = 2 * np.arctan2(np.linalg.norm(q_diff[..., 1:], axis=-1), q_diff[..., 0])
    return angle


def filter_errors(q1, q2):
    if len(q1) != len(q2):
        return ValueError("len(q1) != len(q2)")

    errors = [quaternion_error(q1, q2) for q1, q2 in zip(q1, q2)]
    return errors

for i in range(1, 10):
    # in_path = sys.argv[1] if len(sys.argv) > 1 else "imu_sequence_example_normalized.csv"
    in_path = sys.argv[1] if len(sys.argv) > 1 else f"ned/rec0{i}_ned.csv"
    base = in_path.rsplit(".", 1)[0]
    data = np.genfromtxt(in_path, delimiter=",", names=True)
    t = data["time_s"]
    gyr = np.column_stack([data["gyr_x_rad_s"], data["gyr_y_rad_s"], data["gyr_z_rad_s"]])
    acc = np.column_stack([data["acc_x_m_s2"], data["acc_y_m_s2"], data["acc_z_m_s2"]])
    mag = np.column_stack([data["mag_x_nT"], data["mag_y_nT"], data["mag_z_nT"]])
    cam = np.column_stack([data["cam_qw"], data["cam_qx"], data["cam_qy"], data["cam_qz"]])
    # Run the filter (acc and mag are normalized internally, so units don't matter)
    fs = 1 / np.mean(np.diff(t))

    madgwick = Madgwick(gyr=gyr, acc=acc, mag=mag, frequency=fs)
    complementary = Complementary(gyr=gyr, acc=acc, mag=mag, frequency=fs)
    ekf = EKF(gyr=gyr, acc=acc, mag=mag, frequency=fs)
    fkf = FKF(gyr=gyr, acc=acc, mag=mag, frequency=fs)
    ukf = UKF(gyr=gyr, acc=acc, mag=mag, frequency=fs)

    madgwicks = ned_to_unity(list(madgwick.Q))
    complementarys = ned_to_unity(list(complementary.Q))
    ekfs = ned_to_unity(list(ekf.Q))
    fkfs = ned_to_unity(list(fkf.Q))
    ukfs = ned_to_unity(list(ukf.Q))
    cam = ned_to_unity(cam)

    distortion_rec = int(len(cam) / 2)

    madgwick_errors_not_b = filter_errors(madgwicks[:distortion_rec], cam[:distortion_rec])
    complementary_errors_not_b = filter_errors(complementarys[:distortion_rec], cam[:distortion_rec])
    ekf_errors_not_b = filter_errors(ekfs[:distortion_rec], cam[:distortion_rec])
    fkf_errors_not_b = filter_errors(fkfs[:distortion_rec], cam[:distortion_rec])
    ukf_errors_not_b = filter_errors(ukfs[:distortion_rec], cam[:distortion_rec])

    madgwick_errors_at_b = filter_errors(madgwicks[distortion_rec:], cam[distortion_rec:])
    complementary_errors_at_b = filter_errors(complementarys[distortion_rec:], cam[distortion_rec:])
    ekf_errors_at_b = filter_errors(ekfs[distortion_rec:], cam[distortion_rec:])
    fkf_errors_at_b = filter_errors(fkfs[distortion_rec:], cam[distortion_rec:])
    ukf_errors_at_b = filter_errors(ukfs[distortion_rec:], cam[distortion_rec:])



    with open("notb.txt", "a", encoding="utf-8") as f:
        f.write(f"{np.linalg.norm(madgwick_errors_not_b)}"
                f", {np.linalg.norm(complementary_errors_not_b)}"
                f", {np.linalg.norm(ekf_errors_not_b)}"
                f", {np.linalg.norm(fkf_errors_not_b)}"
                f", {np.linalg.norm(ukf_errors_not_b)}\n")

    with open("atb.txt", "a", encoding="utf-8") as fb:
        fb.write(f"{np.linalg.norm(madgwick_errors_at_b)}"
                f", {np.linalg.norm(complementary_errors_at_b)}"
                f", {np.linalg.norm(ekf_errors_at_b)}"
                f", {np.linalg.norm(fkf_errors_at_b)}"
                f", {np.linalg.norm(ukf_errors_at_b)}\n")

    # fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 7))
    # axes[0].plot(t, madgwick_errors)
    # axes[1].plot(t, complementary_errors)
    # axes[2].plot(t, ekf_errors)
    # axes[3].plot(t[100:], fkf_errors[100:])  # Need normalization.
    # axes[4].plot(t, ukf_errors)
    # #
    # plt.show()
