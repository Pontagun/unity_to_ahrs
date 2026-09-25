import csv
import math
import sys

G = 9.80665          # g -> m/s^2
GAUSS_TO_NT = 1e5    # Gauss -> nT


def fmt(v):
    return f"{v:.6f}".rstrip("0").rstrip(".") if v != 0 else "0"


def normalize(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v] if n > 0 else v


in_path = "rec01.csv" #"sys.argv[1]
base = in_path.rsplit(".", 1)[0]

with open(in_path, newline="") as f:
    rows = [r for r in csv.DictReader(f) if r["Timestamp"].strip()]

t0 = float(rows[0]["Timestamp"])

with open(f"{base}_ned.csv", "w", newline="") as fp, \
     open(f"{base}_ned_normalized.csv", "w", newline="") as fn:
    wp, wn = csv.writer(fp), csv.writer(fn)
    wp.writerow(["time_s", "gyr_x_rad_s", "gyr_y_rad_s", "gyr_z_rad_s",
                 "acc_x_m_s2", "acc_y_m_s2", "acc_z_m_s2",
                 "mag_x_nT", "mag_y_nT", "mag_z_nT"])
    wn.writerow(["time_s", "gyr_x_rad_s", "gyr_y_rad_s", "gyr_z_rad_s",
                 "acc_x_norm", "acc_y_norm", "acc_z_norm",
                 "mag_x_norm", "mag_y_norm", "mag_z_norm"])

    for r in rows:
        v = {k.strip(): float(x) for k, x in r.items()}
        t = (v["Timestamp"] - t0) / 1000
        gyr = [-v["gyro_z"], -v["gyro_x"], v["gyro_y"]]
        acc = [-G * v["acc_z"], -G * v["acc_x"], G * v["acc_y"]]
        mag = [GAUSS_TO_NT * v["mag_z"], GAUSS_TO_NT * v["mag_x"], -GAUSS_TO_NT * v["mag_y"]]

        wp.writerow([fmt(x) for x in [t, *gyr, *acc, *mag]])
        wn.writerow([fmt(x) for x in [t, *gyr, *normalize(acc), *normalize(mag)]])

print(f"Converted {len(rows)} samples -> {base}_ned.csv, {base}_ned_normalized.csv")
