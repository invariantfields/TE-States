# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy>=2.0.0",
#     "jaxopt>=0.8.5",
#     "scipy>=1.12.0",
#
#     # ── GPU dependency: optional (code automatically falls back to CPU if absent) ──
#     # CUDA 13: "cupy-cuda13x>=14.0.0", "jax[cuda13]>=0.4.30", "nvidia-curand-cu13"
#     # CUDA 12: "cupy-cuda12x>=13.0.0", "jax[cuda12]>=0.4.30", "nvidia-curand-cu12"
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    """
    Core imports, hardware detection, and JAX/XLA environment setup.

    Automatic GPU → CPU Fallback
    -----------------------------
    This cell checks if CuPy is installed and an active CUDA GPU is accessible:
    - If a GPU is available: sets USE_GPU = True, xp = cp, JAX_PLATFORMS = 'cuda,cpu'.
    - If no GPU is available or CuPy is missing: falls back seamlessly to CPU
      (USE_GPU = False, xp = np, JAX_PLATFORMS = 'cpu').
    """
    import os
    import sys
    import time
    import pickle
    import argparse
    from itertools import combinations
    from functools import reduce

    import numpy as np
    import jax
    import jax.numpy as jnp
    from jaxopt import LBFGS
    import marimo as mo

    # ---- Hardware Detection & Automatic Fallback ----------------------------
    USE_GPU = False
    cp = None

    try:
        import cupy as cp
        # Test if CUDA runtime is operational and a GPU device is present
        if cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0:
            _cuda_ver = cp.cuda.runtime.runtimeGetVersion()
            _cuda_major = _cuda_ver // 1000
            _cuda_minor = (_cuda_ver % 1000) // 10
            print(f"✅ GPU detected! CUDA runtime: {_cuda_major}.{_cuda_minor}")
            USE_GPU = True
        else:
            print("⚠️ CuPy installed, but no CUDA GPU device available. Falling back to CPU.")
    except (ImportError, Exception) as exc:
        print(f"⚠️ GPU acceleration unavailable ({exc}). Automatically falling back to CPU compute (NumPy + JAX CPU).")
        cp = None
        USE_GPU = False

    # Set array module alias `xp` for array allocations throughout the notebook
    xp = cp if USE_GPU else np

    # Configure JAX / XLA based on hardware availability
    if USE_GPU:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"

    # 64-bit precision is essential for the L-BFGS convergence tolerances used later
    jax.config.update("jax_enable_x64", True)
    return (
        LBFGS,
        USE_GPU,
        argparse,
        combinations,
        jax,
        jnp,
        mo,
        np,
        os,
        pickle,
        reduce,
        sys,
        time,
        xp,
    )


@app.cell
def _(USE_GPU, mo):
    mode_str = "🚀 GPU-accelerated (CuPy + JAX CUDA)" if USE_GPU else "💻 CPU fallback mode (NumPy + JAX CPU)"
    mo.md(rf"""
    # 🧪 N-Qubit State Optimisation & SRE Trajectory Generator

    This notebook generates *stabilizer Rényi entropy* (SRE) trajectories for
    $N$-qubit quantum states.  Starting from random **stabilizer-gap** initial
    states it minimises the total **Hilbert-Schmidt violation** of the
    Hildebrand separability criterion using JAX L-BFGS, while recording the
    exact SRE at every checkpoint via a Fast Walsh-Hadamard Transform (FWHT) kernel.

    > **Current Runtime Execution Mode:** {mode_str}

    ### Workflow at a glance
    1. **Generate** random almost-stabilizer initial states (symplectic circuit + Haar projection).
    2. **Optimise** each state batch with vmapped L-BFGS.
    3. **Measure** SRE at every `chunk_size` optimisation steps.
    4. **Save** full trajectories (purity, violation, SRE, final states) to disk.
    """)
    return


@app.cell
def _(mo):
    """
    UI widgets that control every numerical parameter of the simulation.

    - n_qubits_slider  : system size N (Hilbert space dimension = 2^N).
    - num_seeds_input  : how many independently-initialised trajectories to run.
    - num_steps_input  : total number of L-BFGS micro-iterations per trajectory.
    - chunk_size_input : how many L-BFGS steps to take between metric snapshots
                         (labelled `step_mes` internally).
    - output_dir_input : directory path where output .pkl files will be saved.
    - run_button       : gate that prevents accidental re-runs on parameter
                         changes in interactive mode.
    """
    n_qubits_slider = mo.ui.slider(
        start=2, stop=12, step=1, value=7,
        label="Qubits (N)",
    )
    num_seeds_input = mo.ui.number(
        start=1, stop=50_000, value=4_000, step=1,
        label="Num Seeds (Samples)",
    )
    num_steps_input = mo.ui.number(
        start=1, stop=10_000, value=50, step=1,
        label="Optimisation Steps",
    )
    chunk_size_input = mo.ui.number(
        start=1, stop=500, value=1, step=1,
        label="Chunk Size (step_mes)",
    )
    output_dir_input = mo.ui.text(
        value="zip1",
        label="Output Directory",
    )
    run_button = mo.ui.run_button(label="🚀 Run Data Generation")

    mo.hstack(
        [
            mo.vstack([n_qubits_slider, num_seeds_input]),
            mo.vstack([num_steps_input, chunk_size_input]),
            mo.vstack([output_dir_input, run_button]),
        ],
        align="center",
        justify="start",
        gap=3,
    )
    return (
        chunk_size_input,
        n_qubits_slider,
        num_seeds_input,
        num_steps_input,
        output_dir_input,
        run_button,
    )


@app.cell
def _(USE_GPU, np, xp):
    """
    Exact second-order stabilizer Rényi entropy (SRE) computed via FWHT.

    Background
    ----------
    The SRE of order α=2 is defined as

        M_2(|ψ⟩) = -log2( Σ_{P ∈ Pauli_n} Ξ(P)^4 / 2^n )

    where Ξ(P) = ⟨ψ|P|ψ⟩ is the *characteristic function* of the state.

    Hardware Abstraction
    --------------------
    Uses `xp` (CuPy when GPU is active, NumPy otherwise) to automatically
    run calculations on GPU or CPU without code duplication.
    """

    SRE_CACHE = {}

    def get_sre_cache(n_qubits: int):
        """
        Build (or retrieve from cache) the Hadamard kernel, XOR index table,
        and phase matrix for a given system size N.
        """
        if n_qubits not in SRE_CACHE:
            dim = 2 ** n_qubits

            # Build n-qubit Hadamard matrix H^{⊗n} via Kronecker products.
            H = xp.array([[1.0, 1.0], [1.0, -1.0]], dtype=xp.complex128)
            H_n = H
            for _ in range(n_qubits - 1):
                H_n = xp.kron(H_n, H)

            # XOR index table
            x_idx = xp.arange(dim, dtype=xp.int32)[:, None]
            b_idx = xp.arange(dim, dtype=xp.int32)[None, :]
            xor_indices = x_idx ^ b_idx

            # Phase factor i^{popcount(x & z)}
            x_u = np.arange(dim, dtype=np.uint32)[:, None]
            z_u = np.arange(dim, dtype=np.uint32)[None, :]
            xz = x_u & z_u
            xz -= (xz >> 1) & 0x55555555
            xz = (xz & 0x33333333) + ((xz >> 2) & 0x33333333)
            pop = (((xz + (xz >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24

            phase_table = np.array([1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j], dtype=np.complex128)
            phase_matrix = xp.array(phase_table[pop % 4]).T

            SRE_CACHE[n_qubits] = (H_n, xor_indices, phase_matrix)

        return SRE_CACHE[n_qubits]

    def compute_sre_batch(psi_batch, alpha: int = 2) -> np.ndarray:
        """
        Compute the exact α=2 SRE for a batch of state vectors.

        Automatically uses GPU (CuPy) or CPU (NumPy) depending on runtime hardware.
        """
        num_states, dim = psi_batch.shape
        n_qubits = int(np.log2(dim))
        H_n, xor_indices, phase_matrix = get_sre_cache(n_qubits)

        # Convert to target backend (CuPy or NumPy)
        psi_xp = xp.asarray(psi_batch, dtype=xp.complex128)
        norms = xp.linalg.norm(psi_xp, axis=1, keepdims=True)
        norms = xp.where(norms > 1e-12, norms, 1.0)
        psi_xp = psi_xp / norms

        # V[i, x, b] = ψ*_i[x] · ψ_i[x XOR b]
        psi_conj = xp.conj(psi_xp)[:, :, None]
        psi_xor  = psi_xp[:, xor_indices]
        V_batch  = psi_conj * psi_xor

        # FWHT along b-index
        Xi_complex = xp.matmul(H_n, V_batch)

        # Apply i^{popcount} phase
        Xi_pauli = xp.real(Xi_complex * phase_matrix[None, :, :])

        # Compute M_2 = -log2( Σ Ξ^4 / dim )
        xi_4_sum  = xp.sum(Xi_pauli ** 4, axis=(1, 2))
        sre_batch = -xp.log2(xi_4_sum / dim)

        # Return as CPU NumPy array
        if USE_GPU and hasattr(sre_batch, "get"):
            return sre_batch.get()
        return np.asarray(sre_batch)

    return (compute_sre_batch,)


@app.cell
def _(np):
    """
    Generate a random stabilizer state via a random Clifford circuit.
    """

    def generate_random_generators_symplectic(
        n_qubits: int, depth_multiplier: int = 10
    ) -> list[str]:
        """
        Build a random stabilizer group by applying a random sequence of
        Clifford gates (H, S, CNOT) to the computational-basis stabilizer |0…0⟩.
        """
        x_mat = np.zeros((n_qubits, n_qubits), dtype=int)
        z_mat = np.eye(n_qubits, dtype=int)
        r     = np.zeros(n_qubits, dtype=int)

        def apply_H(target):
            r[:] ^= x_mat[:, target] & z_mat[:, target]
            x_mat[:, target], z_mat[:, target] = (
                z_mat[:, target].copy(),
                x_mat[:, target].copy(),
            )

        def apply_S(target):
            r[:] ^= x_mat[:, target] & z_mat[:, target]
            z_mat[:, target] ^= x_mat[:, target]

        def apply_CNOT(control, target):
            r[:] ^= (x_mat[:, control] & z_mat[:, target]) & (
                x_mat[:, target] ^ z_mat[:, control] ^ 1
            )
            x_mat[:, target]  ^= x_mat[:, control]
            z_mat[:, control] ^= z_mat[:, target]

        num_gates = depth_multiplier * n_qubits ** 2
        for _ in range(num_gates):
            gate = np.random.choice(["H", "S", "CNOT"])
            if gate == "H":
                apply_H(np.random.randint(n_qubits))
            elif gate == "S":
                apply_S(np.random.randint(n_qubits))
            elif n_qubits >= 2:
                c, t = np.random.choice(n_qubits, 2, replace=False)
                apply_CNOT(c, t)

        generators = []
        for i in range(n_qubits):
            sign      = "-" if r[i] else "+"
            pauli_str = sign
            for j in range(n_qubits):
                x, z = x_mat[i, j], z_mat[i, j]
                if   x == 1 and z == 0: pauli_str += "X"
                elif x == 1 and z == 1: pauli_str += "Y"
                elif x == 0 and z == 1: pauli_str += "Z"
                else:                   pauli_str += "I"
            generators.append(pauli_str)

        return generators

    return (generate_random_generators_symplectic,)


@app.cell
def _(generate_random_generators_symplectic, reduce, xp):
    """
    Build Pauli matrices, projectors onto stabilizer subspaces, and random
    almost-stabilizer initial states using `xp` (GPU/CPU unified).
    """

    PAULI_MAP = {
        "I": xp.array([[1, 0], [0, 1]],   dtype=complex),
        "X": xp.array([[0, 1], [1, 0]],   dtype=complex),
        "Y": xp.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": xp.array([[1, 0], [0, -1]],  dtype=complex),
    }

    def pauli_string_to_matrix(pauli_str: str):
        sign      = -1 if pauli_str[0] == "-" else 1
        clean_str = pauli_str.lstrip("+-")
        matrices  = [PAULI_MAP[char] for char in clean_str]
        return sign * reduce(xp.kron, matrices)

    def build_projector_from_generators(generators: list[str]):
        n_qubits = len(generators[0].lstrip("+-"))
        dim      = 2 ** n_qubits
        projector = xp.eye(dim, dtype=complex)
        identity  = xp.eye(dim, dtype=complex)
        for gen in generators:
            g_matrix  = pauli_string_to_matrix(gen)
            projector = projector @ ((identity + g_matrix) / 2.0)
        return projector

    def haar_random_unitary(dim: int):
        z    = xp.random.randn(dim, dim) + 1j * xp.random.randn(dim, dim)
        q, r = xp.linalg.qr(z)
        d    = xp.diagonal(r)
        ph   = d / xp.abs(d)
        return q * ph

    def rand_Almost_Stab_state(n_qubits: int, almost_gap: int = 1):
        psi = xp.zeros(2 ** n_qubits, dtype=complex)
        psi[0] = 1.0
        psi = haar_random_unitary(2 ** n_qubits) @ psi

        if n_qubits - almost_gap == 0:
            return psi

        proj = xp.kron(
            build_projector_from_generators(
                generate_random_generators_symplectic(n_qubits - almost_gap)
            ),
            xp.eye(2 ** almost_gap, dtype=complex),
        )
        projected_psi = proj @ psi
        return projected_psi / xp.linalg.norm(projected_psi)

    return (rand_Almost_Stab_state,)


@app.cell
def _(
    LBFGS,
    combinations,
    compute_sre_batch,
    jax,
    jnp,
    np,
    pickle,
    rand_Almost_Stab_state,
    time,
):
    """
    Core optimisation routine: vmapped L-BFGS over a batch of states.
    Fully compatible with both GPU and CPU execution.
    """

    def run_jax_optimization(
        n_qubits: int,
        num_starts: int,
        num_loops: int,
        gap: int,
        step_mes: int = 1,
        filename: str = "jax_trajectory_data.pkl",
    ) -> dict:
        n     = n_qubits
        k     = n // 2
        n_dim = 2 ** n

        combos = list(combinations(range(n), k))
        perms_list = []
        for combo in combos:
            keep  = list(combo)
            trace = [i for i in range(n) if i not in keep]
            perms_list.append(tuple(trace + keep))

        @jax.jit
        def get_purity_and_violation(psi_vec):
            psi        = psi_vec[:n_dim] + 1j * psi_vec[n_dim:]
            psi        = psi / jnp.linalg.norm(psi)
            psi_tensor = psi.reshape((2,) * n)

            rhos = [
                psi_tensor.transpose(perm).reshape(-1, 2 ** k).conj().T
                @ psi_tensor.transpose(perm).reshape(-1, 2 ** k)
                for perm in perms_list
            ]
            batch_rho = jnp.stack(rhos, axis=0)
            eigenvalues = jnp.linalg.eigvalsh(batch_rho)

            purities    = jnp.sum(eigenvalues ** 2, axis=-1)
            avg_purity  = jnp.mean(purities)
            max_purity  = jnp.max(purities)

            rhs  = eigenvalues[:, 1] + 2 * jnp.sqrt(
                jnp.maximum(eigenvalues[:, 0] * eigenvalues[:, 2], 1e-15)
            )
            viols           = jnp.maximum(0.0, eigenvalues[:, -1] - rhs)
            total_violation = jnp.sum(viols ** 2)

            return avg_purity, max_purity, total_violation

        @jax.jit
        def objective(params):
            _, _, total_viol = get_purity_and_violation(params)
            return total_viol

        solver = LBFGS(fun=objective, maxiter=step_mes, tol=1e-11)

        print(f"Generating {num_starts} initial states (gap={gap})…")
        init_params_list = []
        for _ in range(num_starts):
            psi_init = rand_Almost_Stab_state(n, gap)
            psi_np   = psi_init.get() if hasattr(psi_init, "get") else np.array(psi_init)
            p        = np.concatenate([np.real(psi_np), np.imag(psi_np)])
            init_params_list.append(p)
        params_batch = jnp.array(init_params_list, dtype=jnp.float64)

        vmapped_metrics = jax.vmap(get_purity_and_violation)
        vmapped_run     = jax.vmap(solver.run)

        traj_avg_p = [[] for _ in range(num_starts)]
        traj_max_p = [[] for _ in range(num_starts)]
        traj_viol  = [[] for _ in range(num_starts)]
        traj_sre   = [[] for _ in range(num_starts)]

        start_time = time.time()
        avg_p, max_p, viol = vmapped_metrics(params_batch)
        avg_p_cpu, max_p_cpu, viol_cpu = (
            np.array(avg_p), np.array(max_p), np.array(viol)
        )
        states_complex = params_batch[:, :n_dim] + 1j * params_batch[:, n_dim:]
        sre_vals_cpu   = compute_sre_batch(states_complex, alpha=2)

        for i in range(num_starts):
            traj_avg_p[i].append(float(avg_p_cpu[i]))
            traj_max_p[i].append(float(max_p_cpu[i]))
            traj_viol[i].append(float(viol_cpu[i]))
            traj_sre[i].append(float(sre_vals_cpu[i]))

        num_chunks = max(1, num_loops // step_mes)

        for chunk_idx in range(1, num_chunks + 1):
            res          = vmapped_run(params_batch)
            params_batch = res.params

            avg_p, max_p, viol = vmapped_metrics(params_batch)
            avg_p_cpu, max_p_cpu, viol_cpu = (
                np.array(avg_p), np.array(max_p), np.array(viol)
            )
            states_complex = params_batch[:, :n_dim] + 1j * params_batch[:, n_dim:]
            sre_vals_cpu   = compute_sre_batch(states_complex, alpha=2)

            for i in range(num_starts):
                traj_avg_p[i].append(float(avg_p_cpu[i]))
                traj_max_p[i].append(float(max_p_cpu[i]))
                traj_viol[i].append(float(viol_cpu[i]))
                traj_sre[i].append(float(sre_vals_cpu[i]))

            elapsed = time.time() - start_time
            est_rem = (elapsed / chunk_idx) * (num_chunks - chunk_idx)
            print(
                f"  Step {chunk_idx * step_mes:5d}/{num_loops} | "
                f"Mean Violation: {float(np.mean(viol_cpu)):.2e} | "
                f"Mean SRE: {float(np.mean(sre_vals_cpu)):.4f} | "
                f"Elapsed: {elapsed:.1f}s | "
                f"Est. Remaining: {est_rem:.1f}s"
            )

        final_states = [
            np.array(params_batch[i, :n_dim] + 1j * params_batch[i, n_dim:])
            for i in range(num_starts)
        ]

        trajectories = {
            "average_purity":  traj_avg_p,
            "max_purity":      traj_max_p,
            "sre":             traj_sre,
            "total_violation": traj_viol,
            "final_states":    final_states,
        }

        with open(filename, "wb") as f:
            pickle.dump(trajectories, f)

        print(f"Saved trajectory data → {filename}")
        return trajectories

    return (run_jax_optimization,)


@app.cell
def _(os, run_jax_optimization):
    def gen_data(
        n_qubits: int,
        num_seeds: int,
        num_steps: int,
        step_mes: int,
        f_name: str,
    ) -> None:
        for _stab_qubits in range(0, n_qubits + 1):
            gap      = n_qubits - _stab_qubits
            out_file = f"{f_name}{num_steps}_stps_{gap}.pkl"

            if os.path.exists(out_file):
                print(f"Skipping gap={gap}: '{out_file}' already exists.")
                continue

            print(f"\nComputing {n_qubits}-qubit, gap={gap} "
                  f"({_stab_qubits}-qubit stabilized initial state).")

            run_jax_optimization(
                n_qubits  = n_qubits,
                num_starts= num_seeds,
                num_loops = num_steps,
                gap       = gap,
                step_mes  = step_mes,
                filename  = out_file,
            )

    return (gen_data,)


@app.cell
def _(
    argparse,
    chunk_size_input,
    gen_data,
    mo,
    n_qubits_slider,
    num_seeds_input,
    num_steps_input,
    output_dir_input,
    os,
    run_button,
    sys,
):
    is_script_mode = mo.app_meta().mode == "script"
    should_run     = is_script_mode or run_button.value

    if should_run:
        N          = n_qubits_slider.value
        seeds      = num_seeds_input.value
        steps      = num_steps_input.value
        step_mes   = chunk_size_input.value
        output_dir = output_dir_input.value or "zip1"

        # Interactive CLI prompting and argument parsing when run as a standalone script
        if is_script_mode:
            parser = argparse.ArgumentParser(description="7-Qubit State Optimization & SRE Trajectory Generator")
            parser.add_argument("--qubits", type=int, default=None, help="Number of qubits N (default: 7)")
            parser.add_argument("--seeds", type=int, default=None, help="Number of seed samples (default: 4000)")
            parser.add_argument("--steps", type=int, default=None, help="Number of optimization steps (default: 50)")
            parser.add_argument("--chunk-size", type=int, default=None, help="Snapshot chunk size / step_mes (default: 1)")
            parser.add_argument("--output-dir", type=str, default=None, help="Output directory to save .pkl files (default: zip1)")

            cli_args, _ = parser.parse_known_args()

            # CLI flags take highest precedence
            if cli_args.qubits is not None:     N = cli_args.qubits
            if cli_args.seeds is not None:      seeds = cli_args.seeds
            if cli_args.steps is not None:      steps = cli_args.steps
            if cli_args.chunk_size is not None: step_mes = cli_args.chunk_size
            if cli_args.output_dir is not None: output_dir = cli_args.output_dir

            # Interactive terminal prompts if run directly from terminal without flags
            if (cli_args.qubits is None and cli_args.seeds is None and
                cli_args.steps is None and cli_args.chunk_size is None and
                cli_args.output_dir is None and sys.stdin.isatty()):

                print("\n=======================================================")
                print(" 📋 Simulation Parameter Selection (Script Mode)")
                print(" Press [Enter] to accept the default value shown.")
                print("=======================================================")

                def _prompt_int(label, default_val):
                    try:
                        val_str = input(f" {label} [default {default_val}]: ").strip()
                        return int(val_str) if val_str else default_val
                    except Exception:
                        return default_val

                def _prompt_str(label, default_val):
                    try:
                        val_str = input(f" {label} [default '{default_val}']: ").strip()
                        return val_str if val_str else default_val
                    except Exception:
                        return default_val

                N          = _prompt_int("Qubits (N)", N)
                seeds      = _prompt_int("Num Seeds (Samples)", seeds)
                steps      = _prompt_int("Optimisation Steps", steps)
                step_mes   = _prompt_int("Chunk Size (step_mes)", step_mes)
                output_dir = _prompt_str("Output Directory", output_dir)

        os.makedirs(output_dir, exist_ok=True)
        prefix = os.path.join(output_dir, f"{N}_qbt_{seeds}_sds_ptmzng_jfr_")

        print("\n=======================================================")
        print(f"Running simulation: {N} Qubits | {seeds} Seeds | "
              f"{steps} Steps | Chunk size = {step_mes}")
        print(f"Output directory: {output_dir}")
        print(f"Output prefix: {prefix}")
        print("=======================================================")

        gen_data(
            n_qubits = N,
            num_seeds= seeds,
            num_steps= steps,
            step_mes = step_mes,
            f_name   = prefix,
        )

        status_md = mo.md(
            f"✅ **Simulation complete!** Data saved in directory `{output_dir}` with prefix `{prefix}`"
        )
    else:
        status_md = mo.md(
            "👆 Adjust the parameters above and click **Run Data Generation** to start."
        )

    status_md
    return


if __name__ == "__main__":
    app.run()
