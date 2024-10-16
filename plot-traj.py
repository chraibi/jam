"""Plotting script using matplotlib."""

import matplotlib.pyplot as plt
import pandas as pd
import sys

# Check if a file argument is provided
if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} txt-trajectory filename")
    sys.exit(1)

# Constants
MARKER_SIZE = 22
TEXT_SIZE = 16
LINE_WIDTH = 0.05
LENGTH = 200  # Warning: this value should match the one used in model.py

file_path = sys.argv[1]
print(f"Loading file: {file_path}")
data = pd.read_csv(
    file_path, sep=r"\s+", header=None, names=["id", "time", "x", "v"], comment="#"
)
print("Finished loading")

ids = data["id"].unique()

fig, ax = plt.subplots()
output_filename = file_path.replace(".txt", ".png")

for i in ids[::2]:
    person_data = data[data["id"] == i]
    time = person_data["time"]
    position = person_data["x"] % LENGTH

    abs_diff = position.diff().abs().dropna()
    abs_mean = abs_diff.mean()
    abs_std = abs_diff.std()

    if abs_std > 0.5 * abs_mean:
        change_points = abs_diff[abs_diff > abs_mean + 3 * abs_std].index
    else:
        change_points = []

    start_idx = 0
    for change_idx in change_points:
        plt.plot(
            position[start_idx:change_idx],
            time[start_idx:change_idx],
            "k.",
            ms=LINE_WIDTH,
            lw=LINE_WIDTH,
            rasterized=True,
        )
        start_idx = change_idx + 1

    # Plot remaining data
    plt.plot(
        position[start_idx:],
        time[start_idx:],
        "k.",
        ms=LINE_WIDTH,
        lw=LINE_WIDTH,
        rasterized=True,
    )


plt.xlabel(r"$x_n$", fontsize=MARKER_SIZE)
plt.ylabel(r"$t\; \rm{(s)}$", fontsize=MARKER_SIZE)
plt.xticks(fontsize=TEXT_SIZE)
plt.yticks(fontsize=TEXT_SIZE)

fig.tight_layout()

plt.savefig(output_filename)
print(f"Figure saved as: {output_filename}")
plt.show()
