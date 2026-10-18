# TE-States: N-Qubit State Optimization & SRE Trajectory Generator

This repository accompanies the paper *Threshold entanglement sharing: quantum states with absolutely separable marginals* ([arXiv:2604.13169](https://arxiv.org/abs/2604.13169)). It contains three complementary parts:

1. **Numerical state search** (Python / JAX): generates **Stabilizer Rényi Entropy (SRE)** trajectories for $N$-qubit states initialized from random stabilizer-gap states and optimized via JAX L-BFGS over the Hildebrand separability criterion, then classifies the optimized states as TE / non-TE and plots how much magic they carry.
2. **Existence bounds** (`LP_bound/`, SageMath / Julia): exact rational linear programs (and an SDP relaxation) bounding the largest $k$ for which $k$-TE (TES) states can exist.
3. **Exploratory analysis** (`Absolute_Entanglement.ipynb`, `AAME.py`): SDP / polynomial-optimization bounds, direct searches for 7-qubit states with APPT 3-qubit marginals, and local-unitary analysis of the states found.

**Definition (from the paper).** A *threshold entanglement sharing* (TES) state is an $n$-qubit pure state $|\psi\rangle$ in which all $\lfloor n/2 \rfloor$-body marginals are absolutely biseparable. Throughout the code, plots and file names this is abbreviated **TE** (non-TE = not TES); $k$-TE denotes the same property for $k$-body marginals. In the numerics, membership is decided with absolutely-PPT (APPT) spectral conditions on every $\lfloor N/2 \rfloor$-qubit reduced state (purity bound plus the Hildebrand eigenvalue condition), implemented in `is_TE_dataset` / `is_appt_batch` (shared by the plotting scripts); see [Third-Party Software & Algorithmic References](#third-party-software--algorithmic-references) for the underlying criteria.

---

## Repository Layout

```text
TE-States/
├── Generate_States.py                 # marimo notebook / CLI: state optimization + SRE trajectories
├── generate_plots.py                  # standard plots from trajectory .pkl files
├── generate_plots_with_numbers.py     # same plots, annotated with mean ± std, proportions and N
├── initial_magic.py                   # initial-vs-final SRE bar plot for Initial_Magic_data/
├── Absolute_Entanglement.ipynb        # exploratory notebook: SDPs, APPT checks, direct 7-qubit state search
├── AAME.py                            # Lasserre (ncpol2sdpa) relaxation: max purity under the Hildebrand condition
├── pyproject.toml / uv.lock           # dependencies (pip or uv)
├── Initial_Magic_data/                # trajectory data for the Stab (q=0) vs Haar (q=n) comparison
├── Initial_Magic_plots/               # output of initial_magic.py
├── zip4/ zip41/ zip42/ zip7/          # trajectory data for the q-sweep runs (see table below)
├── plots/                             # output of generate_plots*.py
├── __marimo__/                        # exported HTML / session cache of the marimo notebook
└── LP_bound/                          # exact rational LP + SDP bounds (SageMath / Julia)
```

---

## Notebooks

- **[`Generate_States.py`](Generate_States.py)** [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/invariantfields/TE-States/blob/main/Generate_States.py): Unified marimo notebook with automatic hardware detection. Runs with **GPU acceleration** (CuPy + JAX CUDA) when a CUDA GPU is present, and **automatically falls back to CPU** (NumPy + JAX CPU) if no GPU is available.

---

## Recommended Setup (Virtual Environment)

It is strongly recommended to install all dependencies inside an isolated Python virtual environment (`.venv`) and execute all code from within it.

### Step 1: Clone the repository

```bash
git clone https://github.com/invariantfields/TE-States.git
cd TE-States
```

### Step 2: Create a virtual environment

Using standard Python `venv`:

```bash
python3 -m venv .venv
```

*(Alternatively, if using `uv`: `uv venv`)*

### Step 3: Activate the virtual environment

- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```

- **Windows (Command Prompt)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```

- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

*(Your terminal prompt should now show `(.venv)` at the beginning.)*

### Step 4: Install dependencies

Choose the command that matches your hardware setup:

```bash
# Option A: CPU-only mode (works anywhere without GPU setup)
pip install -e .

# Option B: GPU mode with CUDA 12 (RTX 30xx/40xx, A100, etc.)
pip install -e ".[cuda12]"

# Option C: GPU mode with CUDA 13
pip install -e ".[cuda13]"

# Option D: Include Jupyter for running the LP_bound notebooks
pip install -e ".[lp]"
```

*(To install extra development tools like `ruff`, add `,dev`, e.g., `pip install -e ".[cuda12,dev]"`.)*

A `uv.lock` file is also provided, so `uv sync` reproduces the exact locked environment.

---

## Running the Code

Ensure your virtual environment is activated (`(.venv)` showing in terminal):

### 1. Interactive Web UI (marimo)

```bash
marimo edit Generate_States.py
```
*Opens an interactive browser GUI with sliders and live controls for all simulation parameters.*

### 2. Terminal Script Mode (`python Generate_States.py`)

When running directly from the terminal, the script will **interactively prompt you for simulation parameters**:

```bash
python Generate_States.py
```

```text
=======================================================
 Simulation Parameter Selection (Script Mode)
 Press [Enter] to accept the default value shown.
=======================================================
 Qubits (N) [default 7]: 
 Num Seeds (Samples) [default 4000]: 
 Optimisation Steps [default 50]: 
 Chunk Size (step_mes) [default 1]: 
 Output Directory [default 'zip1']: 
```

> **Tip:** Pressing `[Enter]` on any prompt accepts the default parameter.

#### Command-Line Arguments (Non-Interactive / Batch / Remote Jobs)
You can also pass arguments directly via CLI flags to bypass prompts:

```bash
python Generate_States.py --qubits 7 --seeds 4000 --steps 50 --chunk-size 1 --output-dir zip1
```

| Flag | Meaning | Default |
| :--- | :--- | :---: |
| `--qubits` | Number of qubits $N$ | 7 |
| `--seeds` | Number of seed states (trajectories) per gap | 4000 |
| `--steps` | Total L-BFGS optimization steps | 50 |
| `--chunk-size` | Steps between SRE / purity snapshots (`step_mes`) | 1 |
| `--output-dir` | Directory for the `.pkl` outputs | `zip1` |

#### What the generator does
For every stabilizer gap $q = N - k$ with $k \in \{0, \dots, N\}$ stabilized qubits, it writes one file `{output_dir}/{N}_qbt_{seeds}_sds_ptmzng_jfr_{steps}_stps_{gap}.pkl` (existing files are skipped, so interrupted runs can be resumed). Each initial state is a Haar-random state projected onto the stabilizer code of $k$ random commuting Pauli generators (symplectic construction) acting on the first $k$ qubits and normalized, so $q=0$ gives a pure stabilizer state and $q=N$ a Haar-random state; it is then optimized with vmapped L-BFGS.

Each `.pkl` is a dictionary with the following entries:

| Key | Content |
| :--- | :--- |
| `average_purity` / `max_purity` | Mean / maximum marginal purity at each snapshot, per seed |
| `total_violation` | Total Hilbert-Schmidt violation of the Hildebrand criterion (the optimization objective) |
| `sre` | Stabilizer Rényi Entropy $S_2$ at every snapshot, per seed |
| `final_states` | Optimized state vectors (used by the plotting scripts to classify TE / non-TE) |

---

## Dataset Directories & Experiment Specifications

This repository contains trajectory dataset runs stored in `.pkl` pickle files across several experiment directories:

| Directory | Qubits ($N$) | Seeds (Samples) | Optimization Steps | Stabilizer Gaps ($q$) | Experiments Included |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Initial_Magic_data/`** | 4 & 7 | 2000 | 200 & 1500 | $q \in \{0, 4, 7\}$ | • `4q_2000sds_200stps_200chunk` (2 runs)<br>• `7q_2000sds_1500stps_1500chunk` (2 runs) |
| **`zip7/`** | 7 | 1000 & 2000 | 2500 | $q \in \{0, 1, 2, 3, 4, 5, 6, 7\}$ | • `7q_1000sds_2500stps_250chunk` (8 runs)<br>• `7q_2000sds_2500stps_250chunk` (8 runs) |
| **`zip42/`** | 4 | 4000 | 100 & 200 | $q \in \{0, 1, 2, 3, 4\}$ | • `4q_4000sds_100stps_100chunk` (5 runs)<br>• `4q_4000sds_200stps_200chunk` (5 runs) |
| **`zip4/`** | 4 | 4000 & 5000 | 100 & 40 | $q \in \{0, 1, 2, 3, 4\}$ | • `4q_4000sds_100stps_4chunk` (5 runs)<br>• `4q_5000sds_40stps_1chunk` (5 runs) |
| **`zip41/`** | 4 | 4000 | 100 | $q \in \{0, 1, 2, 3, 4\}$ | • `4q_4000sds_100stps_100chunk` (5 runs) |

### File Naming Scheme
Dataset files follow the convention:
`{N}_qbt_{seeds}_sds_{method}_{steps}_stps_{gap}.pkl` or `{N}_qbt_{seeds}_sds_{method}_{steps}_stps_{chunk}_chunk_{gap}.pkl`
- `{N}_qbt`: System qubit size $N$
- `{seeds}_sds`: Number of trajectory state samples
- `{steps}_stps`: Optimization iterations
- `{chunk}_chunk` (or `step_mes`): Snapshot chunk resolution size
- `{gap}`: Initial stabilizer gap $q = N - k$

---

## Generating Plots & Visualizations

We provide three plotting scripts. The first two take `<dataset_directory> [output_directory]`:

1. **Standard Plots (`generate_plots.py`)**: Groups dataset files by run parameters (qubits, seeds, steps, chunk size) and, for each group, writes `*_TE.png` / `*_NTE.png` (SRE trajectories of TE / non-TE final states for each $q$) and `*_Hist.png` (SRE histogram of TE vs non-TE states). Default directory: `zip7`; default output: `plots/`.
2. **Annotated Plots with Numbers (`generate_plots_with_numbers.py`)**: Same plots, with bar charts annotated with exact Magic (mean $\pm$ std), state proportions (%), and sample counts ($N$). Default directory: `Initial_Magic_data`.
3. **Initial Magic Plot (`initial_magic.py`)**: Side-by-side bars of initial SRE vs final TE / non-TE SRE, plus the TE / non-TE proportion, for `Initial_Magic_data/` (`q=0` is labelled *Stab*, `q=N` *Haar*). Output: `Initial_Magic_plots/Initial_Magic_data_Hist.png`.

Passing `Initial_Magic_data` (or any path containing "initial") to either `generate_plots*.py` script produces the same initial-magic plot as `initial_magic.py`.

**Usage:**
```bash
# Standard plots
python generate_plots.py <dataset_directory> [output_directory]

# Annotated plots with numerical values (mean ± std, N)
python generate_plots_with_numbers.py <dataset_directory> [output_directory]

# Initial-vs-final magic comparison
python initial_magic.py [data_directory] [output_directory]
```

**Examples:**
```bash
# Generate plots for dataset runs (saved to plots/)
python generate_plots.py zip7
python generate_plots_with_numbers.py zip7

# Initial magic comparison (saved to Initial_Magic_plots/)
python initial_magic.py
```

Plots are saved with explicit tags identifying all run parameters (e.g., `4q_4000sds_100stps_100chunk_Hist.png`). If a LaTeX installation is found, text is rendered with `usetex`; otherwise the scripts fall back to Matplotlib's Computer-Modern-style mathtext.

---

## LP Bounds (`LP_bound/`)

> **Note:** The `.sage` helpers and the `LPrational*` / `CheckLPsolutions` notebooks require a [SageMath](https://www.sagemath.org/) installation (typically installed via your system package manager or Conda) and the `lp` dependency extra (`pip install -e ".[lp]"`). `SDP.ipynb` and `Graphic_kTE.ipynb` use a **Julia** kernel instead (see below).

The `LP_bound/` directory contains scripts and Jupyter notebooks for computing and verifying exact rational linear programming (LP) bounds related to the existence of $k$-TE (threshold entanglement sharing) states, together with an SDP relaxation.

| File | Kernel | Purpose |
| :--- | :--- | :--- |
| `fun.sage` | SageMath | Core module: builds the standard-form LP constraints ($A, b, c$) using Krawtchouk polynomials (`Standard_form(n, k, kuni=False)`; `kuni=True` switches to $k$-uniform constraints). |
| `LPrational_TE.ipynb` | SageMath | Solves the exact primal and dual LPs over $\mathbb{Q}$ (PPL backend) for $k = \lfloor n/2 \rfloor$, checks strong duality ($c x^T = y b$), and collects the solutions per $n$. |
| `LPrational-kuni.ipynb` | SageMath | Same exact-LP machinery for the $k$-uniform constraints, for $n = 8, \dots, 40$. |
| `CheckLPsolutions.ipynb` | SageMath | Loads `TE_8_190.sobj` and verifies feasibility, strong duality and the APPT condition $a'_i < \tfrac{4}{3}\binom{n}{i}2^{-i}$ for $n = 8, \dots, 190$, proving compatibility with the existence of $(\lfloor n/2 \rfloor - 1)$-TE states. |
| `TE_8_190.sobj` | — | SageMath object holding the exact solutions `{n: [primal, dual, objective]}` for $n = 8, \dots, 190$. |
| `fun.jl` | Julia | Julia counterpart of the helpers (binomial / multinomial, Krawtchouk $K_w$, index maps `f4_fun` / `f2_fun`, `alphamatrices`). |
| `SDP.ipynb` | Julia | SDP relaxation (JuMP + Clarabel; Eq. (31) of [arXiv:2603.19901](https://arxiv.org/abs/2603.19901) for self-dual codes with the $m$-uniform conditions relaxed). |
| `Graphic_kTE.ipynb` | Julia | Plots the largest $k$ found for $k$-uniform vs $k$-TE states against $\lfloor n/2 \rfloor$, for $n = 8, \dots, 40$. |

The Julia notebooks additionally need `JuMP`, `Clarabel`, `Combinatorics`, `Nemo`, `HiGHS`, `Plots` and `LaTeXStrings` (the `SDP.ipynb` preamble also loads `ClusteredLowRankSolver`, `Infinity`, `COSMO`, `Gurobi`, `CDDLib` and `Mosek`).

---

## Exploratory Analysis (`Absolute_Entanglement.ipynb`, `AAME.py`)

> **Author:** these files were contributed by [Albert Rico](https://github.com/AlbertRico) ([Google Scholar](https://scholar.google.com/citations?user=Ry02HD4AAAAJ&hl=en)).

These files are research scratchpads that complement the main pipeline. They are independent of `Generate_States.py` and need extra packages, available through the `explore` extra in `pyproject.toml` (plus `qgeo`, which has to be installed manually):

```bash
# 1. Python dependencies (adds picos, cvxopt, mosek, ncpol2sdpa, sympy, torch, pandas, networkx, jupyter)
pip install -e ".[explore]"

# 2. qgeo is not on PyPI and has no installer: clone it and put it on PYTHONPATH
git clone https://gitlab.com/qgeo/qgeo.git ~/qgeo_lib
export PYTHONPATH="$HOME/qgeo_lib:$PYTHONPATH"
python -c "import qgeo as q; print(q.ket('01'))"   # sanity check
```

[`qgeo`](https://gitlab.com/qgeo/qgeo) (Felix Huber, Nikolai Wyderka) is only needed by the first cells of the notebook. MOSEK (used by `AAME.py` and imported by the notebook) needs a [licence](https://www.mosek.com/); academic licences are free.

### `AAME.py`
Lasserre-hierarchy relaxation ([`ncpol2sdpa`](https://github.com/peterwittek/ncpol2sdpa), level 3, MOSEK) over a sorted spectrum $x_0 \ge \dots \ge x_{n-1} \ge 0$ with $n = 2^4$ and $\sum_i x_i = 1$. It minimizes $-\sum_i x_i^2$ (i.e. maximizes the purity) subject to the squared Hildebrand condition $(x_0 - x_{n-2})^2 \le 4\,x_{n-1}x_{n-3}$, which gives an upper bound on the purity of an APPT spectrum.

```bash
python AAME.py
```

### `Absolute_Entanglement.ipynb`
Python 3 notebook, organized roughly in the following blocks (top to bottom):

| Cells | Content |
| :--- | :--- |
| 0 – 5 | Two-qubit warm-ups with `qgeo`: singlet / Werner states, CHSH value, and the maximal negativity reachable by rotating states of a given spectrum (random orthogonal sampling) — the absolute-separability picture for $2\times2$. |
| 6 – 15 | Analytic plots (no optimization). Cells 6 – 13 draw families of curves $m_k = 1 - k\,x^2$ against the negativity of the partial transpose, for $N = 10$ and $k = 1, \dots, N-1$ (viridis colour per $k$). Here $x$ solves a quadratic fixed by the partial-transpose norm $A$ for a spectrum with $k$ equal eigenvalues $x$; the successive cells are variants of the same plot (different branch of the square root, domain of $A$, masking of unphysical regions / dominated $k$, legend placement). Cells 14 – 15 plot the maximal negativity $\mathcal{N}_{\max}$ as a function of $m_k$ for $d = 6$, $k = 1, \dots, 7$, with dashed horizontal lines marking Schmidt-number thresholds ($SN \ge \dots$); the figure is saved as `n_max_intercept_colors.png`. *(Interpretation based on reading the code; see the paper for the exact definitions.)* |
| 16 – 21 | PICOS SDPs searching for "almost AME" states (swap-invariant diagonal ansatz with equal marginals and PPT / APPT constraints, $k = 2$ and $k = 4$ qubits per party), plus an iterative PPT-SDP construction and a gradient-descent attempt that did not converge. |
| 22 – 23, 33 – 35 | Direct search for 7-qubit states whose 35 three-qubit marginals all satisfy the Hildebrand APPT condition: a GME-based iteration (following [arXiv:2210.13475](https://arxiv.org/abs/2210.13475)) and an L-BFGS-B minimization of the summed Hildebrand violations. This finds states with **35/35 APPT marginals for $N=7$**; the same search at $N=8$ (4-qubit marginals) did not converge within the iteration budget. |
| 24 – 32 | Analysis of the optimized state: per-marginal purity, spectra and APPT margin, the LU-sparsification of the state with PyTorch, and LaTeX table output for the paper. |
| 36 – 49 | Structure hunting for a 7-qubit APPT state: a circulant graph state (offsets 1, 2, 4) and its APPT check (cells 36 – 38); **cell 39 exports `State_7.txt` and `State_4.txt` from the `zip7/` and `zip4/` data (see below)**; then the APPT check of that state, canonical forms via local unitaries, a Fano-plane ansatz, and searches over weight-3/4 basis states with roots-of-unity or optimized phases (cells 40 – 49). The circulant graph state passes the APPT check; the roots-of-unity and weight-3/4 searches did not find APPT states. |

> **Note on inputs/outputs:** the cells exchange states through plain-text files in the working directory, in one of two layouts: two columns (real, imaginary) for `as_search_state.txt` / `asep_marginal_state.txt`, or a single column of interleaved (real, imaginary) values for `final_state_{N}.txt` and `State_7.txt`.
>
> | File | Created by |
> | :--- | :--- |
> | `State_7.txt`, `State_4.txt` | Cell 39, from the optimization data in [`zip7/`](zip7) and [`zip4/`](zip4) (see below). `State_7.txt` is the input of the analysis cells 40 – 49. |
> | `as_search_state.txt`, `asep_marginal_state.txt`, `final_state_{N}.txt`, `Optimized_Phases.txt` | The search cells themselves; they appear after the corresponding cell is run. |
>
> **Where `State_7.txt` and `State_4.txt` come from.** The `.pkl` files written by `Generate_States.py` store the optimized states under the key `final_states` (a list with one complex vector of length `2**N` per seed). Cell 39 loads the $q = 0$ run (started from stabilizer states) of each dataset — `zip7/7_qbt_2000_sds_ptmzng_jfr_2500_stps_0.pkl` and `zip4/4_qbt_4000_sds_ptmzng_jfr_100_stps_0.pkl` — keeps the final states whose $\lfloor N/2 \rfloor$-qubit marginals all satisfy the Hildebrand APPT condition (35 three-qubit marginals for $N=7$, 6 two-qubit marginals for $N=4$), and writes the one with the lowest final SRE (`sre[:, -1]`) as $2\cdot 2^N$ interleaved values (seed 249, SRE 2.69, for $N=7$; seed 3031, SRE 0.53, for $N=4$). Change the paths in the last lines of the cell to use another run. Each file is regenerated automatically if it is missing. LaTeX is needed for the cells that set `text.usetex`.

---

## Attribution Notes

### Contributors
- **Linear Programming Bounds (`LP_bound/`)**: Developed by [**Gerard Anglès Munné**](https://scholar.google.com/citations?user=kEi4_G4AAAAJ) (exact rational LP formulations, dual certificates, and APPT verification over $\mathbb{Q}$ using SageMath).
- **Exploratory Analysis (`Absolute_Entanglement.ipynb`, `AAME.py`)**: Developed by [**Albert Rico**](https://scholar.google.com/citations?user=Ry02HD4AAAAJ&hl=en) ([@AlbertRico](https://github.com/AlbertRico)) (SDP and polynomial-optimization bounds, direct searches for 7-qubit states with APPT marginals, and local-unitary / structural analysis of the states found).
- **Quantum State Optimization, SRE Trajectories & Framework**: Developed by [**Naga Bhavya Teja Kothakonda**](https://scholar.google.com/citations?user=fuzvthUAAAAJ) ([@invariantfields](https://github.com/invariantfields)) (symplectic state generation, JAX L-BFGS optimization over the Hildebrand separability criterion, Fast Walsh-Hadamard Transform $S_2$ magic entropy computation, interactive marimo interface, and plotting suites).

### Third-Party Software & Algorithmic References

This repository adapts, implements, and builds upon the following open-source packages and foundational works:

- **Random Clifford / Stabilizer Generation**: The symplectic generator construction for random stabilizer states was derived from [`randstab`](https://github.com/qotlabs/randstab), implementing the Aaronson-Gottesman stabilizer tableau update rules ([*Phys. Rev. A* 70, 052328](https://doi.org/10.1103/PhysRevA.70.052328)).
- **Fast Hadamard Transform for Magic / SRE**: The vectorized Fast Walsh-Hadamard Transform (FWHT) implementation for computing Stabilizer Rényi Entropy ($S_2$) was adapted from the Julia package [`HadaMAG.jl`](https://github.com/bsc-quantic/HadaMAG.jl), developed by the Quantic group at the Barcelona Supercomputing Center (BSC-CNS) ([arXiv:2601.07824](https://arxiv.org/abs/2601.07824)).
- **Absolute Separability & APPT Criteria**: The separability evaluation combines the Gurvits-Barnum maximal purity ball bound ([*Phys. Rev. A* 66, 062311](https://doi.org/10.1103/PhysRevA.66.062311)) and the two-qubit spectral separability condition by R. Hildebrand ([*Phys. Rev. A* 76, 052325](https://doi.org/10.1103/PhysRevA.76.052325)).
- **Exact Rational LP Solver (`LP_bound/`)**: Primal and dual linear programs are solved over $\mathbb{Q}$ using [SageMath](https://www.sagemath.org/) coupled with the [Parma Polyhedra Library (PPL)](https://www.bugseng.com/ppl).
- **Exploratory Analysis (`Absolute_Entanglement.ipynb`, `AAME.py`)**: The two-qubit warm-ups use the [`qgeo`](https://gitlab.com/qgeo/qgeo) library for quantum information by Felix Huber and Nikolai Wyderka (BSD-3). The SDPs are modelled with [PICOS](https://picos-api.gitlab.io/picos/) and solved with CVXOPT / [MOSEK](https://www.mosek.com/); the polynomial-optimization relaxation in `AAME.py` uses [`ncpol2sdpa`](https://github.com/peterwittek/ncpol2sdpa).
- **Optimization & Hardware Acceleration**: Numerical trajectory optimization is powered by [JAX](https://github.com/google/jax) and [JAXopt](https://github.com/google/jaxopt) (vmapped L-BFGS), with GPU acceleration via [CuPy](https://cupy.dev/) and an interactive interface built using [marimo](https://marimo.io/).

### Citation

If you use this work, methods, or datasets, please cite our paper ([arXiv:2604.13169](https://arxiv.org/abs/2604.13169)):

```bibtex
@misc{rico2026thresholdentanglementsharingquantum,
      title={Threshold entanglement sharing: quantum states with absolutely separable marginals}, 
      author={Albert Rico and Jofre Abellanet-Vidal and Naga Bhavya Teja Kothakonda and Anna Sanpera and Gerard Anglès Munné},
      year={2026},
      eprint={2604.13169},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2604.13169}, 
}
```



