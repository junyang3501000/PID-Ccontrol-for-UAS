import csv
import os
from itertools import cycle

import matplotlib.pyplot as plt


CSV_FILE = "flight_log.csv"
SAVE_FIGURE = False
FIGURE_NAME = "flight_3d_trajectory.png"


def resolve_csv_file(preferred_name):
    """
    Pick the latest flight log.
    Supports both flight_log.csv and timestamped files such as
    flight_log_YYYYMMDD_HHMMSS.csv.
    """
    base, ext = os.path.splitext(preferred_name)
    prefix = f"{base}_"
    candidates = []

    try:
        if os.path.exists(preferred_name):
            candidates.append(preferred_name)
        for name in os.listdir("."):
            if name.endswith(ext) and name.startswith(prefix):
                candidates.append(name)
    except OSError:
        return preferred_name

    if not candidates:
        return preferred_name
    return max(candidates, key=lambda n: os.path.getmtime(n))


def load_flight_data(filename):
    data = {
        "Target_X": [],
        "Target_Y": [],
        "Target_Z": [],
        "Target_Yaw": [],
        "Pos_X": [],
        "Pos_Y": [],
        "Pos_Z": [],
    }

    try:
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    parsed = {}
                    for key in data:
                        if key not in row:
                            raise KeyError(key)
                        value = row[key].strip()
                        if value == "":
                            raise ValueError(key)
                        parsed[key] = float(value)
                    for key in data:
                        data[key].append(parsed[key])
                except (KeyError, ValueError):
                    continue
    except OSError:
        pass

    return data


def unique_targets(data):
    targets = []
    seen = set()
    n = min(
        len(data["Target_X"]),
        len(data["Target_Y"]),
        len(data["Target_Z"]),
        len(data["Target_Yaw"]),
    )

    for i in range(n):
        target = (
            data["Target_X"][i],
            data["Target_Y"][i],
            data["Target_Z"][i],
            data["Target_Yaw"][i],
        )
        rounded = tuple(round(v, 6) for v in target)
        if rounded in seen:
            continue
        seen.add(rounded)
        targets.append(target)
    return targets


def set_equal_3d_axes(ax, xs, ys, zs):
    """Keep 3D axes visually proportional instead of stretched."""
    if not xs or not ys or not zs:
        return

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)

    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    z_mid = (z_min + z_max) / 2.0
    radius = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    radius = max(radius, 0.5)

    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_zlim(max(0.0, z_mid - radius), z_mid + radius)


def plot_3d_trajectory(data, csv_name):
    xs = data["Pos_X"]
    ys = data["Pos_Y"]
    zs = data["Pos_Z"]

    if not xs:
        print(f"No valid position data found in {csv_name}")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.manager.set_window_title("3D Flight Trajectory")

    ax.plot(xs, ys, zs, color="tab:blue", linewidth=1.6, label="Actual trajectory")
    ax.scatter(xs[0], ys[0], zs[0], color="tab:green", s=60, marker="o", label="Start")
    ax.scatter(xs[-1], ys[-1], zs[-1], color="tab:red", s=60, marker="x", label="End")

    colors = cycle(
        [
            "tab:orange",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:olive",
            "tab:cyan",
            "gold",
            "black",
        ]
    )
    targets = unique_targets(data)
    for idx, (tx, ty, tz, tyaw) in enumerate(targets, start=1):
        color = next(colors)
        ax.scatter(tx, ty, tz, color=color, s=90, marker="*", label=f"Target {idx}")
        ax.text(tx, ty, tz, f" T{idx}", color=color)

    all_x = xs + [t[0] for t in targets]
    all_y = ys + [t[1] for t in targets]
    all_z = zs + [t[2] for t in targets]
    set_equal_3d_axes(ax, all_x, all_y, all_z)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"3D Flight Trajectory\n{csv_name}")
    ax.grid(True)
    ax.legend(loc="best")

    # Matplotlib 3D windows support mouse drag rotation and scroll-wheel zoom.
    if SAVE_FIGURE:
        fig.savefig(FIGURE_NAME, dpi=180, bbox_inches="tight")
        print(f"Saved figure: {FIGURE_NAME}")

    plt.show()


if __name__ == "__main__":
    csv_name = resolve_csv_file(CSV_FILE)
    print(f"Using CSV file: {csv_name}")
    flight_data = load_flight_data(csv_name)
    plot_3d_trajectory(flight_data, csv_name)
