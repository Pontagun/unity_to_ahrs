"""
Estimate orientation from an IMU CSV with the Madgwick filter (ahrs library)
and plot roll / pitch / yaw.

Input columns: time_s, gyr_*_rad_s, acc_*_m_s2, mag_*_nT (NED body frame)

Usage: python estimate_orientation.py imu_sequence_example_1.csv
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from ahrs import QuaternionArray
from ahrs.filters import Madgwick, Mahony, EKF, Complementary, FKF, UKF


def ned_to_unity(q_list):
    # return q_list
    return [np.array([w, -y, z, -x]) for w, x, y, z in q_list]


def quat_diff(q1, q2):
    # Relative rotation from q1 to q2 (both [w, x, y, z]): q_d = conj(q1) * q2,
    # so q1 * q_d = q2. Also returns the rotation angle between them in radians.
    w1, x1, y1, z1 = np.asarray(q1, dtype=float) / np.linalg.norm(q1)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float) / np.linalg.norm(q2)
    x1, y1, z1 = -x1, -y1, -z1  # conjugate of q1
    q_d = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    # q and -q are the same rotation, so use |w| to get the shortest angle
    angle = 2 * np.arccos(np.clip(abs(q_d[0]), 0.0, 1.0))
    return q_d, angle


def quat_diff_series(qs1, qs2):
    # Angle between each pair of quaternions in two (N, 4) arrays, in degrees.
    # Returns an (N,) array, ready for plt.plot(t, ...).
    qs1, qs2 = np.asarray(qs1), np.asarray(qs2)
    if qs1.shape != qs2.shape:
        raise ValueError(f"shape mismatch: {qs1.shape} vs {qs2.shape}")
    return [quat_diff(q1, q2) for q1, q2 in zip(qs1, qs2)]

# in_path = sys.argv[1] if len(sys.argv) > 1 else "imu_sequence_example_normalized.csv"
in_path = sys.argv[1] if len(sys.argv) > 1 else "ned_unit/rec22_ned_normalized.csv"
base = in_path.rsplit(".", 1)[0]
data = np.genfromtxt(in_path, delimiter=",", names=True)
t = data["time_s"]
gyr = np.column_stack([data["gyr_x_rad_s"], data["gyr_y_rad_s"], data["gyr_z_rad_s"]])
acc = np.column_stack([data["acc_x_m_s2"], data["acc_y_m_s2"], data["acc_z_m_s2"]])
mag = np.column_stack([data["mag_x_nT"], data["mag_y_nT"], data["mag_z_nT"]])


# Run the filter (acc and mag are normalized internally, so units don't matter)
fs = 1 / np.mean(np.diff(t))
madgwick = Madgwick(gyr=gyr, acc=acc, mag=mag, frequency=fs)
complementary = Complementary(gyr=gyr, acc=acc, mag=mag, frequency=fs)
ekf = EKF(gyr=gyr, acc=acc, mag=mag, frequency=fs)
fkf = FKF(gyr=gyr, acc=acc, mag=mag, frequency=fs)
ukf = UKF(gyr=gyr, acc=acc, mag=mag, frequency=fs)


fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 7))

madgwick_l = list(madgwick.Q)
complementary_l = list(complementary.Q)
ekf_l = list(ekf.Q)
fkf_l = list(fkf.Q)
ukf_l = list(ukf.Q)

madgwick_l = ned_to_unity(madgwick_l)
complementary_l = ned_to_unity(complementary_l)
ekf_l = ned_to_unity(ekf_l)
fkf_l = ned_to_unity(fkf_l)
ukf_l = ned_to_unity(ukf_l)

cam = np.column_stack([data["cam_qw"], data["cam_qx"], data["cam_qy"], data["cam_qz"]])
cam = ned_to_unity(cam)
ekf_diff_cam = quat_diff_series(ekf_l, cam)
axes[0].plot(t, cam)
axes[1].plot(t, ekf_l)
axes[2].plot(t, [q[0] for q in ekf_diff_cam], min)
axes[3].plot(t, [a[1] for a in ekf_diff_cam])
# axes[3].plot(t[30:], fkf_l[30:]) # Need normalization.
# axes[4].plot(t, ukf_l)

for ax in axes:
    ax.set_ylim(0, 1)

plt.show()
