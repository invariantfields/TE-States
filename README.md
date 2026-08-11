# TE-States: N-Qubit State Optimization & SRE Trajectory Generator

This repository generates **Stabilizer Rényi Entropy (SRE)** trajectories for $N$-qubit quantum states initialized from random stabilizer-gap states and optimized via JAX L-BFGS over the Hildebrand separability criterion.

---

## Notebooks

- **[`Generate_States.py`](Generate_States.py)**: Unified marimo notebook with automatic hardware detection. Runs with **GPU acceleration** (CuPy + JAX CUDA) when a CUDA GPU is present, and **automatically falls back to CPU** (NumPy + JAX CPU) if no GPU is available.

---

## Recommended Setup (Virtual Environment)

It is strongly recommended to install all dependencies inside an isolated Python virtual environment (`.venv`) and execute all code from within it.

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/TE_states.git
cd TE_states
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

### Step 4: Upgrade pip & install dependencies

Choose the command that matches your hardware setup:

```bash
# Option A: CPU-only mode (works anywhere without GPU setup)
pip install -e .

# Option B: GPU mode with CUDA 12 (RTX 30xx/40xx, A100, etc.)
pip install -e ".[cuda12]"

# Option C: GPU mode with CUDA 13
pip install -e ".[cuda13]"

# Option D: Include dependencies for running LP_bound Jupyter notebooks
pip install -e ".[lp]"
```

*(To install extra development tools like `ruff`, add `,dev`, e.g., `pip install -e ".[dev]"`)*

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
```

> **Tip:** Pressing `[Enter]` on any prompt accepts the default parameter.

#### Command-Line Arguments (Non-Interactive / Batch / Remote Jobs)
You can also pass arguments directly via CLI flags to bypass prompts:

```bash
python Generate_States.py --qubits 7 --seeds 4000 --steps 50 --chunk-size 1
```

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

We provide two plotting scripts:

1. **Standard Plots (`generate_plots.py`)**: Groups dataset files by experiment parameters and outputs standard line charts and SRE histograms.
2. **Annotated Plots with Numbers (`generate_plots_with_numbers.py`)**: Extends `generate_plots.py` by annotating each bar chart with exact numerical values for Magic (mean $\pm$ std), state proportions (%), and sample counts ($N$).

**Usage:**
```bash
# Standard plots
python generate_plots.py <dataset_directory>

# Annotated plots with numerical values (mean ± std, N)
python generate_plots_with_numbers.py <dataset_directory>
```

**Examples:**
```bash
python generate_plots.py Initial_Magic_data
python generate_plots_with_numbers.py Initial_Magic_data
```

Plots are saved to the `plots/` directory with explicit tags identifying all run parameters (e.g., `4q_2000sds_200stps_200chunk_Hist.png`).


---

## LP Bounds (`LP_bound/`)

> **Note:** Executing the notebooks in this directory requires a [SageMath](https://www.sagemath.org/) installation (typically installed via your system package manager or Conda) and the `lp` dependency extra (`pip install -e ".[lp]"`).

The `LP_bound/` directory contains SageMath scripts and Jupyter notebooks for computing and verifying exact rational linear programming (LP) bounds related to the existence of $k$-TE (T-Entangled) states.

- **`fun.sage`**: Core SageMath module that builds the standard form LP constraint matrices ($A, b, c$) using Krawtchouk polynomials.
- **`LPrational.ipynb`**: Solves the exact primal and dual Mixed Integer Linear Programs (MILP) over the rationals ($\mathbb{Q}$), checks for strong duality ($c x^T = y b$), and saves the exact solutions to a SageMath object file (`.sobj`).
- **`LPrational_TE.ipynb` & `CheckLPsolutions.ipynb`**: Load the generated `.sobj` exact solutions and formally verify that they satisfy the APPT condition across all specified qubit sizes $N$, proving compatibility with the existence of $(\lfloor N/2 \rfloor - 1)$-TE states.
