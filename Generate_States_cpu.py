# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy>=2.0.0",
#     "jax>=0.4.30",
#     "jaxopt>=0.8.5",
#     "scipy>=1.12.0",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


# ---------------------------------------------------------------------------
# Cell 1 – Imports & JAX CPU Configuration
# ---------------------------------------------------------------------------
@app.cell
def _():
    """
    Core imports and CPU environment setup.

    Key choices:
    - JAX_PLATFORMS=cpu: forces JAX to run on CPU without requiring GPU/CUDA drivers.
    - jax_enable_x64: enables 64-bit floats throughout (required for
      numerically stable L-BFGS optimisation).
    """
    import os
    import time
    import pickle
    from itertools import combinations
    from functools import reduce

    import numpy as np
    import jax
    import jax.numpy as jnp
    from jaxopt import LBFGS
    import marimo as mo

    # Force JAX to use CPU backend
    os.environ["JAX_PLATFORMS"] = "cpu"

    # 64-bit precision is essential for the L-BFGS convergence tolerances used
    # later (tol=1e-11).
    jax.config.update("jax_enable_x64", True)

    return LBFGS, combinations, jax, jnp, mo, np, os, pickle, reduce, time


# ---------------------------------------------------------------------------
# Cell 2 – Notebook Title and Description
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
    # 🧪 N-Qubit State Optimisation & SRE Trajectory Generator (CPU Version)

    This notebook generates *stabilizer Rényi entropy* (SRE) trajectories for
    $N$-qubit quantum states using pure CPU computation (NumPy + JAX CPU).

    Starting from random **stabilizer-gap** initial states, it minimises the
    total **Hilbert-Schmidt violation** of the Hildebrand separability
    criterion using JAX L-BFGS, while recording the exact SRE at every
    checkpoint via a fast NumPy-based FWHT kernel.

    ### Workflow at a glance
    1. **Generate** random almost-stabilizer initial states (symplectic circuit + Haar projection).
    2. **Optimise** each state batch with vmapped L-BFGS on CPU.
    3. **Measure** SRE at every `chunk_size` optimisation steps.
    4. **Save** full trajectories (purity, violation, SRE, final states) to disk.
    """)
    return


# ---------------------------------------------------------------------------
# Cell 3 – Interactive Parameter Controls
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    """
    UI widgets that control every numerical parameter of the simulation.

    - n_qubits_slider  : system size N (Hilbert space dimension = 2^N).
    - num_seeds_input  : how many independently-initialised trajectories to run.
    - num_steps_input  : total number of L-BFGS micro-iterations per trajectory.
    - chunk_size_input : how many L-BFGS steps to take between metric snapshots
                         (labelled `step_mes` internally).
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
    run_button = mo.ui.run_button(label="🚀 Run Data Generation")

    mo.hstack(
        [
            mo.vstack([n_qubits_slider, num_seeds_input]),
            mo.vstack([num_steps_input, chunk_size_input]),
            run_button,
        ],
        align="center",
        justify="start",
        gap=3,
    )
    return chunk_size_input, n_qubits_slider, num_seeds_input, num_steps_input, run_button


# ---------------------------------------------------------------------------
# Cell 4 – CPU SRE Cache and Batch SRE Computation (NumPy / FWHT)
# ---------------------------------------------------------------------------
@app.cell
def _(np):
    """
    Exact second-order stabilizer Rényi entropy (SRE) computed on CPU using NumPy.

    Background
    ----------
    The SRE of order α=2 is defined as

        M_2(|ψ⟩) = -log2( Σ_{P ∈ Pauli_n} Ξ(P)^4 / 2^n )

    where Ξ(P) = ⟨ψ|P|ψ⟩ is the *characteristic function* of the state.

    Caching strategy
    ----------------
    The Hadamard kernel H_n, the XOR index table, and the i^{#Y} phase matrix
    are all stored in `SRE_CPU_CACHE` keyed by `n_qubits`.
    """

    SRE_CPU_CACHE = {}

    def get_cpu_sre_cache(n_qubits: int):
        """
        Build (or retrieve from cache) the three NumPy arrays required by the
        FWHT-based SRE kernel for a given number of qubits.
        """
        if n_qubits not in SRE_CPU_CACHE:
            dim = 2 ** n_qubits

            # Build n-qubit Hadamard matrix H^{⊗n} via Kronecker products.
            H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
            H_n = H
            for _ in range(n_qubits - 1):
                H_n = np.kron(H_n, H)

            # XOR index table
            x_idx = np.arange(dim, dtype=np.int32)[:, None]
            b_idx = np.arange(dim, dtype=np.int32)[None, :]
            xor_indices = x_idx ^ b_idx

            # Phase factor i^{popcount(x & z)}
            x_u = np.arange(dim, dtype=np.uint32)[:, None]
            z_u = np.arange(dim, dtype=np.uint32)[None, :]
            xz = x_u & z_u
            xz -= (xz >> 1) & 0x55555555
            xz = (xz & 0x33333333) + ((xz >> 2) & 0x33333333)
            pop = (((xz + (xz >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24

            phase_table = np.array([1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j], dtype=np.complex128)
            phase_matrix = phase_table[pop % 4].T

            SRE_CPU_CACHE[n_qubits] = (H_n, xor_indices, phase_matrix)

        return SRE_CPU_CACHE[n_qubits]

    def compute_sre_native_numpy_batch(psi_batch, alpha: int = 2) -> np.ndarray:
        """
        Compute the exact α=2 stabilizer Rényi entropy for a batch of states on CPU.

        Parameters
        ----------
        psi_batch : array-like of shape (num_states, 2^n_qubits), complex128
            Batch of state vectors.
        alpha : int
            SRE order (α=2).

        Returns
        -------
        sre_batch : np.ndarray of shape (num_states,), float64
        """
        psi_np = np.asarray(psi_batch, dtype=np.complex128)
        num_states, dim = psi_np.shape
        n_qubits = int(np.log2(dim))
        H_n, xor_indices, phase_matrix = get_cpu_sre_cache(n_qubits)

        # Normalise each state
        norms = np.linalg.norm(psi_np, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        psi_np = psi_np / norms

        # V[i, x, b] = ψ*_i[x] · ψ_i[x XOR b]
        psi_conj = np.conj(psi_np)[:, :, None]
        psi_xor  = psi_np[:, xor_indices]
        V_batch  = psi_conj * psi_xor

        # FWHT along b-index
        Xi_complex = np.matmul(H_n, V_batch)

        # Apply i^{popcount} phase
        Xi_pauli = np.real(Xi_complex * phase_matrix[None, :, :])

        # Compute M_2 = -log2( Σ Ξ^4 / dim )
        xi_4_sum  = np.sum(Xi_pauli ** 4, axis=(1, 2))
        sre_batch = -np.log2(xi_4_sum / dim)

        return sre_batch

    return (compute_sre_native_numpy_batch,)


# ---------------------------------------------------------------------------
# Cell 5 – Random Stabilizer-State Generator (Symplectic Clifford Circuit)
# ---------------------------------------------------------------------------
@app.cell
def _(np):
    """
    Generate a random stabilizer state via a random Clifford circuit.
    """

    def generate_random_generators_symplectic(
        n_qubits: int, depth_multiplier: int = 10
    ) -> list[str]:
        """
        Build a random stabilizer group using tableau operations.
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


# ---------------------------------------------------------------------------
# Cell 6 – Stabilizer Projector & Almost-Stabilizer Initial States (CPU)
# ---------------------------------------------------------------------------
@app.cell
def _(generate_random_generators_symplectic, np, reduce):
    """
    Build Pauli matrices, projectors, and random almost-stabilizer states on CPU.
    """

    PAULI_MAP = {
        "I": np.array([[1, 0], [0, 1]],   dtype=complex),
        "X": np.array([[0, 1], [1, 0]],   dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]],  dtype=complex),
    }

    def pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
        sign      = -1 if pauli_str[0] == "-" else 1
        clean_str = pauli_str.lstrip("+-")
        matrices  = [PAULI_MAP[char] for char in clean_str]
        return sign * reduce(np.kron, matrices)

    def build_projector_from_generators(generators: list[str]) -> np.ndarray:
        n_qubits = len(generators[0].lstrip("+-"))
        dim      = 2 ** n_qubits
        projector = np.eye(dim, dtype=complex)
        identity  = np.eye(dim, dtype=complex)
        for gen in generators:
            g_matrix  = pauli_string_to_matrix(gen)
            projector = projector @ ((identity + g_matrix) / 2.0)
        return projector

    def haar_random_unitary_cpu(dim: int) -> np.ndarray:
        z    = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        q, r = np.linalg.qr(z)
        d    = np.diagonal(r)
        ph   = d / np.abs(d)
        return q * ph

    def rand_Almost_Stab_state(n_qubits: int, almost_gap: int = 1) -> np.ndarray:
        psi = np.zeros(2 ** n_qubits, dtype=complex)
        psi[0] = 1.0
        psi = haar_random_unitary_cpu(2 ** n_qubits) @ psi

        if n_qubits - almost_gap == 0:
            return psi

        proj = np.kron(
            build_projector_from_generators(
                generate_random_generators_symplectic(n_qubits - almost_gap)
            ),
            np.eye(2 ** almost_gap, dtype=complex),
        )
        projected_psi = proj @ psi
        return projected_psi / np.linalg.norm(projected_psi)

    return (rand_Almost_Stab_state,)


# ---------------------------------------------------------------------------
# Cell 7 – JAX/L-BFGS CPU Batch Optimiser with SRE Trajectory Logging
# ---------------------------------------------------------------------------
@app.cell
def _(
    LBFGS,
    combinations,
    compute_sre_native_numpy_batch,
    jax,
    jnp,
    np,
    pickle,
    rand_Almost_Stab_state,
    time,
):
    """
    Core optimisation routine: vmapped L-BFGS over a batch of states on CPU.
    """

    def run_jax_cpu_optimization(
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
            psi_np   = np.array(psi_init)
            p = np.concatenate([np.real(psi_np), np.imag(psi_np)])
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
        sre_vals_cpu   = compute_sre_native_numpy_batch(states_complex, alpha=2)

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
            sre_vals_cpu   = compute_sre_native_numpy_batch(states_complex, alpha=2)

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

    return (run_jax_cpu_optimization,)


# ---------------------------------------------------------------------------
# Cell 8 – Utility: Pack Multiple .pkl Files into One Archive
# ---------------------------------------------------------------------------
@app.cell
def _(os, pickle):
    def pack_pkl_files(source_dir_or_files, output_archive_path: str) -> None:
        if isinstance(source_dir_or_files, str):
            if not os.path.exists(source_dir_or_files):
                print(f"Error: directory '{source_dir_or_files}' does not exist.")
                return
            file_paths = [
                os.path.join(source_dir_or_files, f)
                for f in sorted(os.listdir(source_dir_or_files))
                if f.endswith(".pkl")
            ]
        else:
            file_paths = list(source_dir_or_files)

        packed_data = {}
        for fpath in file_paths:
            fname = os.path.basename(fpath)
            try:
                with open(fpath, "rb") as f:
                    packed_data[fname] = pickle.load(f)
                print(f"  Packed: {fname}")
            except Exception as exc:
                print(f"  Error reading {fname}: {exc}")

        out_dir = os.path.dirname(output_archive_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        try:
            with open(output_archive_path, "wb") as f:
                pickle.dump(packed_data, f)
            print(f"\nArchive saved ({len(packed_data)} files) → {output_archive_path}")
        except Exception as exc:
            print(f"Error saving archive: {exc}")

    return (pack_pkl_files,)


# ---------------------------------------------------------------------------
# Cell 9 – Data Generation Driver: Iterate over Stabilizer Gaps
# ---------------------------------------------------------------------------
@app.cell
def _(os, pack_pkl_files, run_jax_cpu_optimization):
    def gen_data(
        n_qubits: int,
        num_seeds: int,
        num_steps: int,
        step_mes: int,
        f_name: str,
    ) -> None:
        for _stab_qubits in range(5, n_qubits + 1):
            gap      = n_qubits - _stab_qubits
            out_file = f"{f_name}{num_steps}_stps_{gap}.pkl"

            if os.path.exists(out_file):
                print(f"Skipping gap={gap}: '{out_file}' already exists.")
                continue

            print(f"\nComputing {n_qubits}-qubit, gap={gap} "
                  f"({_stab_qubits}-qubit stabilized initial state).")

            run_jax_cpu_optimization(
                n_qubits  = n_qubits,
                num_starts= num_seeds,
                num_loops = num_steps,
                gap       = gap,
                step_mes  = step_mes,
                filename  = out_file,
            )

            pack_pkl_files("zip1", "file1.pkl")

    return (gen_data,)


# ---------------------------------------------------------------------------
# Cell 10 – Orchestration: Read UI Values, Run Simulation, Display Status
# ---------------------------------------------------------------------------
@app.cell
def _(
    chunk_size_input,
    gen_data,
    mo,
    n_qubits_slider,
    num_seeds_input,
    num_steps_input,
    os,
    run_button,
):
    is_script_mode = mo.app_meta().mode == "script"
    should_run     = is_script_mode or run_button.value

    if should_run:
        os.makedirs("zip1", exist_ok=True)

        N        = n_qubits_slider.value
        seeds    = num_seeds_input.value
        steps    = num_steps_input.value
        step_mes = chunk_size_input.value
        prefix   = f"zip1/{N}_qbt_{seeds}_sds_ptmzng_jfr_"

        print("=======================================================")
        print(f"Running CPU simulation: {N} Qubits | {seeds} Seeds | "
              f"{steps} Steps | Chunk size = {step_mes}")
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
            f"✅ **Simulation complete!** Data saved under prefix `{prefix}`"
        )
    else:
        status_md = mo.md(
            "👆 Adjust the parameters above and click **Run Data Generation** to start."
        )

    status_md
    return


if __name__ == "__main__":
    app.run()
