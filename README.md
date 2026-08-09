# 🧪 TE-States: N-Qubit State Optimization & SRE Trajectory Generator

This repository generates **Stabilizer Rényi Entropy (SRE)** trajectories for $N$-qubit quantum states initialized from random stabilizer-gap states and optimized via JAX L-BFGS over the Hildebrand separability criterion.

---

## 📌 Notebooks

- **[`Generate_States.py`](Generate_States.py)**: Unified marimo notebook with automatic hardware detection. Runs with **GPU acceleration** (CuPy + JAX CUDA) when a CUDA GPU is present, and **automatically falls back to CPU** (NumPy + JAX CPU) if no GPU is available.

---

## 🚀 Recommended Setup (Virtual Environment)

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
```

*(To install extra development tools like `ruff`, add `,dev`, e.g., `pip install -e ".[dev]"`)*

---

## 💻 Running the Code

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
 📋 Simulation Parameter Selection (Script Mode)
 Press [Enter] to accept the default value shown.
=======================================================
 Qubits (N) [default 7]: 
 Num Seeds (Samples) [default 4000]: 
 Optimisation Steps [default 50]: 
 Chunk Size (step_mes) [default 1]: 
```

> 💡 **Tip:** Pressing `[Enter]` on any prompt accepts the default parameter.

#### Command-Line Arguments (Non-Interactive / Batch / Remote Jobs)
You can also pass arguments directly via CLI flags to bypass prompts:

```bash
python Generate_States.py --qubits 7 --seeds 4000 --steps 50 --chunk-size 1
```

---

## 📂 Dataset Directories & Experiment Specifications

This repository contains trajectory dataset runs stored in `.pkl` pickle files across several experiment directories:

| Directory | Qubits ($N$) | Seeds (Samples) | Optimization Steps | Stabilizer Gaps ($q$) | Experiments Included |
| :--- | :---: | :---: | :---: | :---: | :--- |
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

## 📊 Generating Plots & Visualizations

Use [`generate_plots.py`](generate_plots.py) to automatically group dataset files by experiment parameters (including chunk resolution) and output line charts and Stabilizer Rényi Entropy (SRE) histograms:

```bash
python generate_plots.py <dataset_directory>
```

**Usage Examples:**
```bash
python generate_plots.py zip7
python generate_plots.py zip42
```

Plots are saved to the `plots/` directory with explicit tags identifying all run parameters (e.g., `4q_4000sds_100stps_100chunk_TE.png`, `4q_4000sds_100stps_100chunk_Hist.png`).

