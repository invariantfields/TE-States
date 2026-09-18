#!/usr/bin/env python3
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import re
import sys
import glob
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

if shutil.which("latex"):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    })
else:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
    })


# -------------------------------------------------------------------
# Fast Vectorized Batch Quantum TE Evaluation
# -------------------------------------------------------------------
def is_appt_batch(x_batch):
    N, D, _ = x_batch.shape
    purities = np.sum(x_batch.real**2 + x_batch.imag**2, axis=(1, 2))
    bound = 1.0 / (D - 1.0)
    is_te_mask = purities <= bound

    rem_idx = np.where(~is_te_mask)[0]
    if len(rem_idx) > 0:
        ex = np.linalg.eigvalsh(x_batch[rem_idx])
        ex = np.clip(ex, 0.0, None)
        cond = (ex[:, -1] - ex[:, 1]) ** 2 <= 4.0 * ex[:, 0] * ex[:, 3]
        is_te_mask[rem_idx] = cond
    return is_te_mask


def is_TE_dataset(states, dim=2):
    states = np.asarray(states)
    N, L = states.shape
    n = int(np.log2(L))
    k = n - n // 2
    n_rem = n - k

    te_flags = np.ones(N, dtype=bool)

    for _i in combinations(range(n), k):
        active = np.where(te_flags)[0]
        if len(active) == 0:
            break
        sub_states = states[active]
        per = [x for x in range(n) if x not in _i] + list(_i)

        reshaped = sub_states.reshape([len(active)] + [dim] * n)
        permuted = np.transpose(reshaped, [0] + [p + 1 for p in per])
        mat = permuted.reshape(len(active), dim**n_rem, dim**k)
        x_batch = mat @ np.swapaxes(mat.conj(), 1, 2)
        x_batch = (x_batch + np.swapaxes(x_batch.conj(), 1, 2)) / 2.0

        sub_flags = is_appt_batch(x_batch)
        te_flags[active] = sub_flags

    return te_flags


def parse_filename_info(filepath):
    filename = os.path.basename(filepath)
    m = re.search(
        r"^(\d+)_qbt_(\d+)_sds_(.*?)(\d+)_stps_(?:(\d+)_(?:chunk|chk)_)?(\d+)\.pkl$",
        filename,
    )
    if m:
        qbt, sds, mid, stps, fn_chunk, gap = m.groups()
        return {
            "qbt": int(qbt),
            "sds": int(sds),
            "stps": int(stps),
            "gap": int(gap),
            "filename": filename,
        }

    qbt = 4 if "4_qbt" in filename else 7
    gap = 0 if filename.endswith("_0.pkl") else (4 if filename.endswith("_4.pkl") else 7)
    return {"qbt": qbt, "sds": 2000, "stps": 200, "gap": gap, "filename": filename}


def load_initial_magic_data(data_dir="Initial_Magic_data"):
    files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    loaded_items = []
    for f in files:
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        info = parse_filename_info(f)
        loaded_items.append((data, info, f))

    # Sort globally by qubit count (n) then stabilizer gap (q)
    loaded_items.sort(key=lambda item: (item[1]["qbt"], item[1]["gap"]))
    return loaded_items


def plot_initial_magic_hist(loaded_data, output_path="Initial_Magic_plots/Initial_Magic_data_Hist.png"):
    fig_w, fig_h = 5.2, 4.8
    label_fontsize = 12
    tick_fontsize = 10.5

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(fig_w, fig_h),
        sharex=True,
        facecolor="white",
        gridspec_kw={"hspace": 0},
    )
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    x_labels = []
    init_centers, init_errs, init_counts = [], [], []
    te_centers, te_errs, te_counts = [], [], []
    non_te_centers, non_te_errs, non_counts = [], [], []

    for data, info, file_path in loaded_data:
        te_mask = is_TE_dataset(data["final_states"])

        init_vals = []
        te_vals = []
        non_te_vals = []

        for j, is_te_state in enumerate(te_mask):
            init_sre = data["sre"][j][0]
            final_sre = data["sre"][j][-1]
            init_vals.append(init_sre)
            if is_te_state:
                te_vals.append(final_sre)
            else:
                non_te_vals.append(final_sre)

        state_name = "Stab" if info["gap"] == 0 else "Haar"
        x_labels.append(f"$n={info['qbt']}, \\text{{{state_name}}}$")

        init_centers.append(np.mean(init_vals) if init_vals else 0.0)
        init_errs.append(np.std(init_vals) if len(init_vals) > 1 else 0.0)
        init_counts.append(len(init_vals))

        te_centers.append(np.mean(te_vals) if te_vals else 0.0)
        te_errs.append(np.std(te_vals) if len(te_vals) > 1 else 0.0)
        te_counts.append(len(te_vals))

        non_te_centers.append(np.mean(non_te_vals) if non_te_vals else 0.0)
        non_te_errs.append(np.std(non_te_vals) if len(non_te_vals) > 1 else 0.0)
        non_counts.append(len(non_te_vals))

    x = np.arange(len(x_labels))
    width = 0.25
    ekw = dict(elinewidth=0.8, capthick=0.8)

    # Colorblind-friendly palette
    color_init = "#009E73"  # Bluish Green
    color_te = "#0072B2"  # Blue
    color_non_te = "#D55E00"  # Vermillion

    # Subplot 1 (Top): SRE Values
    ax1.bar(
        x - width,
        init_centers,
        width,
        yerr=init_errs,
        capsize=3,
        label=r"$\text{Initial SRE}$",
        color=color_init,
        edgecolor="black",
        linewidth=0.5,
        error_kw=ekw,
    )
    ax1.bar(
        x,
        te_centers,
        width,
        yerr=te_errs,
        capsize=3,
        label=r"$\text{Final TE SRE}$",
        color=color_te,
        edgecolor="black",
        linewidth=0.5,
        error_kw=ekw,
    )
    ax1.bar(
        x + width,
        non_te_centers,
        width,
        yerr=non_te_errs,
        capsize=3,
        label=r"$\text{Final non-TE SRE}$",
        color=color_non_te,
        edgecolor="black",
        linewidth=0.5,
        error_kw=ekw,
    )
    ax1.set_ylabel(r"$\text{SRE } (S_2)$", fontsize=label_fontsize)
    ax1.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0", axis="y")
    ax1.tick_params(direction="in", top=True, right=True, which="both", labelsize=tick_fontsize)
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)
    ax1.set_ylim(0, 5.8)
    ax1.legend(loc="upper left", frameon=True, fontsize=tick_fontsize - 0.5)

    # Subplot 2 (Bottom): Upright Proportions
    te_props_list = []
    non_te_props_list = []
    for tc, nc in zip(te_counts, non_counts):
        tot = tc + nc
        te_props_list.append(tc / tot if tot > 0 else 0.0)
        non_te_props_list.append(nc / tot if tot > 0 else 0.0)

    w2 = 0.25
    ax2.bar(
        x - w2 / 2,
        te_props_list,
        w2,
        label=r"$\text{TE States}$",
        color=color_te,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )
    ax2.bar(
        x + w2 / 2,
        non_te_props_list,
        w2,
        label=r"$\text{non-TE States}$",
        color=color_non_te,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=tick_fontsize)
    ax2.tick_params(direction="in", top=True, right=True, which="both", labelsize=tick_fontsize)
    ax2.spines["top"].set_visible(True)
    ax2.spines["right"].set_visible(True)
    ax2.set_xlabel(r"$\text{Qubits } (n) \text{ and Initial State}$", fontsize=label_fontsize)
    ax2.set_ylabel(r"$\text{Proportion of States}$", fontsize=label_fontsize)
    ax2.set_ylim(0, 1.08)
    ax2.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax2.set_yticklabels([r"$0\%$", r"$25\%$", r"$50\%$", r"$75\%$", r"$100\%$"], fontsize=tick_fontsize)
    ax2.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0", axis="y")

    fig.subplots_adjust(hspace=0)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "Initial_Magic_data"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "Initial_Magic_plots"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Initial Magic dataset files from {target_dir}...")
    loaded_data = load_initial_magic_data(target_dir)
    if not loaded_data:
        print(f"No .pkl files found in {target_dir}/!")
        return

    print(f"Found {len(loaded_data)} files in {target_dir}/:")
    for _, info, f in loaded_data:
        print(f"  - {info['filename']} (n={info['qbt']}, q={info['gap']})")

    out_file = os.path.join(output_dir, f"{os.path.basename(os.path.normpath(target_dir))}_Hist.png")
    print(f"\n--- Generating Side-by-Side Bar Plot ({out_file}) ---")
    plot_initial_magic_hist(loaded_data, output_path=out_file)


if __name__ == "__main__":
    main()
