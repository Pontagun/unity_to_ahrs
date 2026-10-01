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

# in_path = sys.argv[1] if len(sys.argv) > 1 else "imu_sequence_example_normalized.csv"
in_path = sys.argv[1] if len(sys.argv) > 1 else "rec22_ned_normalized.csv"
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


fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 7))

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

axes[0].plot(t, madgwick_l)
axes[1].plot(t, cam)
axes[2].plot(t, ekf_l)
axes[3].plot(t[500:], fkf_l[500:]) # Need normalization.
axes[4].plot(t, ukf_l)

plt.show()
