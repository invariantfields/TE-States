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
