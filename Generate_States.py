# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy>=2.0.0",
#     "jaxopt>=0.8.5",
#     "scipy>=1.12.0",
#
#     # ── GPU dependency: uncomment ONE block below based on your CUDA version ──
#
#     # CUDA 13 (RTX 40xx, H100, A100 with CUDA 13 toolkit)
#     "cupy-cuda13x>=14.0.0",
#     "jax[cuda13]>=0.4.30",
#
#     # CUDA 12 (RTX 30xx/40xx, A100 with CUDA 12 toolkit)
#     # "cupy-cuda12x>=13.0.0",
#     # "jax[cuda12]>=0.4.30",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", auto_download=["html"])


# ---------------------------------------------------------------------------
# Cell 1 – Imports & JAX/XLA configuration
# ---------------------------------------------------------------------------
@app.cell
def _():
    """
    Core imports and GPU/XLA environment setup.

    Key choices:
    - XLA_PYTHON_CLIENT_PREALLOCATE=false: prevents JAX from reserving the
      entire GPU VRAM up front, leaving room for CuPy allocations.
    - JAX_PLATFORMS=cuda,cpu: prefers CUDA, falls back to CPU gracefully.
    - jax_enable_x64: enables 64-bit floats throughout (required for
      numerically stable L-BFGS optimisation).

    CUDA version support
    --------------------
    Both CUDA 12 and CUDA 13 are supported.  The cell detects the active
    toolkit version at runtime via `cp.cuda.runtime.runtimeGetVersion()` and
    prints a confirmation.  No code changes are needed when switching between
    CUDA versions — just install the matching CuPy wheel:

        CUDA 12:  pip install "cupy-cuda12x>=13.0.0" "jax[cuda12]>=0.4.30"
        CUDA 13:  pip install "cupy-cuda13x>=14.0.0" "jax[cuda13]>=0.4.30"
    """
    import os
    import time
    import pickle
    from itertools import combinations
    from functools import reduce

    import numpy as np
    import cupy as cp
    import jax
    import jax.numpy as jnp
    from jaxopt import LBFGS
    import marimo as mo

    # ---- Detect and report active CUDA version ------------------------------
    _cuda_ver = cp.cuda.runtime.runtimeGetVersion()
    _cuda_major = _cuda_ver // 1000
    _cuda_minor = (_cuda_ver % 1000) // 10
    print(f"CuPy detected CUDA runtime: {_cuda_major}.{_cuda_minor}")

    if _cuda_major not in (12, 13):
        import warnings
        warnings.warn(
            f"Detected CUDA {_cuda_major}.{_cuda_minor}. "
            "Only CUDA 12 and 13 are tested. Proceeding, but results may vary."
        )

    # Allow JAX and CuPy to share GPU memory without one pre-consuming it all.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"

    # 64-bit precision is essential for the L-BFGS convergence tolerances used
    # later (tol=1e-11).
    jax.config.update("jax_enable_x64", True)

    return LBFGS, combinations, cp, jax, jnp, mo, np, os, pickle, reduce, time


# ---------------------------------------------------------------------------
# Cell 2 – Notebook title and description
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
    # 🧪 N-Qubit State Optimisation & SRE Trajectory Generator

    This notebook generates *stabilizer Rényi entropy* (SRE) trajectories for
    $N$-qubit quantum states.  Starting from random **stabilizer-gap** initial
    states it minimises the total **Hilbert-Schmidt violation** of the
    Hildebrand separability criterion using JAX L-BFGS, while recording the
    exact SRE at every checkpoint via a native CuPy / FWHT GPU kernel.

    ### Workflow at a glance
    1. **Generate** random almost-stabilizer initial states (symplectic circuit + Haar projection).
    2. **Optimise** each state batch with vmapped L-BFGS on GPU.
    3. **Measure** SRE at every `chunk_size` optimisation steps.
    4. **Save** full trajectories (purity, violation, SRE, final states) to disk.
    """)
    return


# ---------------------------------------------------------------------------
# Cell 3 – Interactive parameter controls
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    """
    UI widgets that control every numerical parameter of the simulation.

    - n_qubits_slider  : system size N (Hilbert space dimension = 2^N).
    - num_seeds_input  : how many independently-initialised trajectories to run.
    - num_steps_input  : total number of L-BFGS micro-iterations per trajectory.
    - chunk_size_input : how many L-BFGS steps to take between metric snapshots
                         (labelled `step_mes` internally).  Smaller values give
                         denser trajectories at higher I/O cost.
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
# Cell 4 – GPU SRE cache and batch SRE computation (CuPy / FWHT)
# ---------------------------------------------------------------------------
@app.cell
def _(cp, np):
    """
    Exact second-order stabilizer Rényi entropy (SRE) computed entirely on GPU.

    Background
    ----------
    The SRE of order α=2 is defined as

        M_2(|ψ⟩) = -log2( Σ_{P ∈ Pauli_n} Ξ(P)^4 / 2^n )

    where Ξ(P) = ⟨ψ|P|ψ⟩ is the *characteristic function* of the state.

    The characteristic function can be computed via a Fast Walsh-Hadamard
    Transform (FWHT) over the Weyl-Heisenberg phase space, which is what this
    cell implements.

    Caching strategy
    ----------------
    The Hadamard kernel H_n, the XOR index table, and the i^{#Y} phase matrix
    are all system-size-dependent but **state-independent**.  They are built
    once and stored in `SRE_GPU_CACHE` keyed by `n_qubits` so repeated calls
    with the same N pay no construction cost.
    """

    # Module-level cache: avoids re-computing H_n and phase tables each call.
    SRE_GPU_CACHE = {}

    def get_gpu_sre_cache(n_qubits: int):
        """
        Build (or retrieve from cache) the three GPU arrays required by the
        FWHT-based SRE kernel for a given number of qubits.

        Returns
        -------
        H_n              : (dim, dim) Hadamard matrix ⊗^n on GPU.
        xor_indices      : (dim, dim) int32 array where entry [x,b] = x XOR b.
        phase_matrix_gpu : (dim, dim) complex128 array encoding i^{popcount(x&z)}
                           for the Weyl-Heisenberg phase factor, transposed to
                           (z, b) layout used by the kernel.
        """
        if n_qubits not in SRE_GPU_CACHE:
            dim = 2 ** n_qubits

            # Build the n-qubit Hadamard matrix H^{⊗n} on the GPU via
            # successive Kronecker products.
            H = cp.array([[1.0, 1.0], [1.0, -1.0]], dtype=cp.complex128)
            H_n = H
            for _ in range(n_qubits - 1):
                H_n = cp.kron(H_n, H)

            # XOR index table: xor_indices[x, b] = x XOR b.
            # Used to map psi[b] -> psi[x XOR b] without explicit loop.
            x_idx = cp.arange(dim, dtype=cp.int32)[:, None]
            b_idx = cp.arange(dim, dtype=cp.int32)[None, :]
            xor_indices = x_idx ^ b_idx

            # Phase factor i^{popcount(x & z)} using a vectorised bit-popcount
            # (SWAR / "parallel bit-counting") algorithm on CPU, then moved to GPU.
            x_u = np.arange(dim, dtype=np.uint32)[:, None]
            z_u = np.arange(dim, dtype=np.uint32)[None, :]
            xz = x_u & z_u
            # SWAR popcount for 32-bit integers:
            xz -= (xz >> 1) & 0x55555555
            xz = (xz & 0x33333333) + ((xz >> 2) & 0x33333333)
            pop = (((xz + (xz >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24

            # Map popcount mod 4 to {1, i, -1, -i}.
            phase_table = np.array([1.0 + 0j, 0.0 + 1j, -1.0 + 0j, 0.0 - 1j], dtype=np.complex128)
            # Transpose to (z, b) layout expected by the batched kernel.
            phase_matrix_gpu = cp.array(phase_table[pop % 4]).T

            SRE_GPU_CACHE[n_qubits] = (H_n, xor_indices, phase_matrix_gpu)

        return SRE_GPU_CACHE[n_qubits]

    def compute_sre_native_cupy_batch(psi_batch, alpha: int = 2) -> np.ndarray:
        """
        Compute the exact α=2 stabilizer Rényi entropy for a batch of states.

        Parameters
        ----------
        psi_batch : array-like of shape (num_states, 2^n_qubits), complex128
            Batch of (possibly unnormalised) state vectors.  Accepts JAX,
            CuPy, or NumPy arrays – converted to CuPy internally.
        alpha : int
            SRE order.  Only α=2 is implemented (default).

        Returns
        -------
        sre_batch : np.ndarray of shape (num_states,), float64
            SRE values M_2(|ψ_i⟩) for each state in the batch.

        Algorithm
        ---------
        For each state ψ:
        1. Compute the Gram-like matrix V[x, b] = ψ*[x] · ψ[x XOR b].
        2. Apply H_n (FWHT) to get the characteristic function Ξ in the Pauli
           basis: Xi = H_n @ V (in the b-index).
        3. Multiply by the phase factor i^{popcount(x&z)} to get the real
           Pauli expectation values Xi_pauli.
        4. SRE = -log2( Σ Xi_pauli^4 / dim ).
        """
        num_states, dim = psi_batch.shape
        n_qubits = int(np.log2(dim))
        H_n, xor_indices, phase_matrix_gpu = get_gpu_sre_cache(n_qubits)

        # Move to GPU and normalise each state.
        psi_gpu = cp.asarray(psi_batch, dtype=cp.complex128)
        norms = cp.linalg.norm(psi_gpu, axis=1, keepdims=True)
        norms = cp.where(norms > 1e-12, norms, 1.0)
        psi_gpu = psi_gpu / norms

        # V[i, x, b] = ψ*_i[x] · ψ_i[x XOR b]  (vectorised over the batch).
        psi_conj = cp.conj(psi_gpu)[:, :, None]       # (batch, dim, 1)
        psi_xor  = psi_gpu[:, xor_indices]             # (batch, dim, dim)
        V_batch  = psi_conj * psi_xor                  # (batch, dim, dim)

        # FWHT along the b-index: Xi_complex[i] = H_n @ V_batch[i].
        Xi_complex = cp.matmul(H_n, V_batch)

        # Apply i^{popcount} phase to obtain real Pauli expectation values.
        Xi_pauli = cp.real(Xi_complex * phase_matrix_gpu[None, :, :])

        # Compute M_2 = -log2( Σ Ξ^4 / dim ).
        xi_4_sum  = cp.sum(Xi_pauli ** 4, axis=(1, 2))
        sre_batch = -cp.log2(xi_4_sum / dim)

        return cp.asnumpy(sre_batch)

    return (compute_sre_native_cupy_batch,)


# ---------------------------------------------------------------------------
# Cell 5 – Random stabilizer-state generator (symplectic Clifford circuit)
# ---------------------------------------------------------------------------
@app.cell
def _(np):
    """
    Generate a random stabilizer state via a random Clifford circuit.

    The tableau formalism (Aaronson & Gottesman 2004) is used to track the
    destabilizer-stabilizer tableau.  Only the stabilizer half is needed here
    so we store:

        x_mat[i, j]  - X-part of the i-th generator on qubit j
        z_mat[i, j]  - Z-part of the i-th generator on qubit j
        r[i]         - phase bit (0 = +1, 1 = -1) of the i-th generator
    """

    def generate_random_generators_symplectic(
        n_qubits: int, depth_multiplier: int = 10
    ) -> list[str]:
        """
        Build a random stabilizer group by applying a random sequence of
        Clifford gates (H, S, CNOT) to the computational-basis stabilizer |0…0⟩.

        Parameters
        ----------
        n_qubits       : number of qubits.
        depth_multiplier: total gates ≈ depth_multiplier * n_qubits².

        Returns
        -------
        generators : list of Pauli strings, e.g. ["+XIZI", "-ZZXI", …].
        """
        # Initialise tableau: generators are Z_0, Z_1, …, Z_{n-1}.
        x_mat = np.zeros((n_qubits, n_qubits), dtype=int)
        z_mat = np.eye(n_qubits, dtype=int)
        r     = np.zeros(n_qubits, dtype=int)

        # ---- Clifford gate update rules (in-place on the tableau) ----------

        def apply_H(target):
            """Hadamard: swaps X↔Z and flips phase when both are set."""
            r[:] ^= x_mat[:, target] & z_mat[:, target]
            x_mat[:, target], z_mat[:, target] = (
                z_mat[:, target].copy(),
                x_mat[:, target].copy(),
            )

        def apply_S(target):
            """Phase gate S: maps X→Y (Z unchanged), updates phase."""
            r[:] ^= x_mat[:, target] & z_mat[:, target]
            z_mat[:, target] ^= x_mat[:, target]

        def apply_CNOT(control, target):
            """CNOT: propagates X forward and Z backward through the circuit."""
            r[:] ^= (x_mat[:, control] & z_mat[:, target]) & (
                x_mat[:, target] ^ z_mat[:, control] ^ 1
            )
            x_mat[:, target]  ^= x_mat[:, control]
            z_mat[:, control] ^= z_mat[:, target]

        # ---- Random circuit -------------------------------------------------
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

        # ---- Convert tableau rows to Pauli strings -------------------------
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
# Cell 6 – Stabilizer projector & almost-stabilizer initial states
# ---------------------------------------------------------------------------
@app.cell
def _(cp, generate_random_generators_symplectic, reduce):
    """
    Build Pauli matrices, projectors onto stabilizer subspaces, and random
    *almost-stabilizer* (stabilizer-gap) initial states.

    An almost-stabilizer state with gap g is prepared by:
    1. Drawing a Haar-random unitary on the full 2^n space.
    2. Applying the state to |0⟩ to get a Haar-random state ψ.
    3. Projecting ψ onto the tensor product of a random (n-g)-qubit stabilizer
       subspace and a free g-qubit space, then re-normalising.

    This biases the initial state toward having low but non-zero SRE, providing
    a warm start for the violation-minimisation optimiser.
    """

    # Single-qubit Pauli matrices (on GPU).
    PAULI_MAP = {
        "I": cp.array([[1, 0], [0, 1]],   dtype=complex),
        "X": cp.array([[0, 1], [1, 0]],   dtype=complex),
        "Y": cp.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": cp.array([[1, 0], [0, -1]],  dtype=complex),
    }

    def pauli_string_to_matrix(pauli_str: str) -> cp.ndarray:
        """
        Convert a Pauli string (e.g. '+XZY') to its 2^n × 2^n matrix via
        successive Kronecker products.
        """
        sign      = -1 if pauli_str[0] == "-" else 1
        clean_str = pauli_str.lstrip("+-")
        matrices  = [PAULI_MAP[char] for char in clean_str]
        return sign * reduce(cp.kron, matrices)

    def build_projector_from_generators(generators: list[str]) -> cp.ndarray:
        """
        Build the stabilizer projector Π = ∏_g (I + g) / 2 for a list of
        commuting Pauli generators g.

        The result is the rank-1 projector onto the unique +1 eigenstate of all
        generators (assuming they form a maximal Abelian subgroup).
        """
        n_qubits = len(generators[0].lstrip("+-"))
        dim      = 2 ** n_qubits
        projector = cp.eye(dim, dtype=complex)
        identity  = cp.eye(dim, dtype=complex)
        for gen in generators:
            g_matrix  = pauli_string_to_matrix(gen)
            projector = projector @ ((identity + g_matrix) / 2.0)
        return projector

    def haar_random_unitary_gpu(dim: int) -> cp.ndarray:
        """
        Sample a Haar-random unitary matrix via QR decomposition of a
        Ginibre ensemble (complex Gaussian random matrix).
        """
        z    = cp.random.randn(dim, dim) + 1j * cp.random.randn(dim, dim)
        q, r = cp.linalg.qr(z)
        # Correct the QR phase ambiguity to ensure uniform Haar measure.
        d    = cp.diagonal(r)
        ph   = d / cp.abs(d)
        return q * ph

    def rand_Almost_Stab_state(n_qubits: int, almost_gap: int = 1) -> cp.ndarray:
        """
        Generate a random almost-stabilizer state with stabilizer gap `almost_gap`.

        Parameters
        ----------
        n_qubits   : total number of qubits N.
        almost_gap : number of 'free' (non-stabilized) qubits g ∈ {0, …, N}.
                     gap=0 → pure stabilizer state; gap=N → pure Haar-random.

        Returns
        -------
        Normalised state vector of length 2^N on the GPU.
        """
        # Start from a Haar-random state on the full space.
        psi = cp.zeros(2 ** n_qubits, dtype=complex)
        psi[0] = 1.0
        psi = haar_random_unitary_gpu(2 ** n_qubits) @ psi

        # If no stabilizer part is requested, return the Haar-random state as-is.
        if n_qubits - almost_gap == 0:
            return psi

        # Project onto the tensor product of a random (N-g)-qubit stabilizer
        # subspace and the free g-qubit space.
        proj = cp.kron(
            build_projector_from_generators(
                generate_random_generators_symplectic(n_qubits - almost_gap)
            ),
            cp.eye(2 ** almost_gap, dtype=complex),
        )
        projected_psi = proj @ psi
        return projected_psi / cp.linalg.norm(projected_psi)

    return (rand_Almost_Stab_state,)


# ---------------------------------------------------------------------------
# Cell 7 – JAX/L-BFGS batch optimiser with SRE trajectory logging
# ---------------------------------------------------------------------------
@app.cell
def _(
    LBFGS,
    combinations,
    compute_sre_native_cupy_batch,
    jax,
    jnp,
    np,
    pickle,
    rand_Almost_Stab_state,
    time,
):
    """
    Core optimisation routine: vmapped L-BFGS over a batch of states.

    Objective
    ---------
    Minimise the total squared Hildebrand violation across all bipartitions of
    equal size ⌊N/2⌋.  For a bipartition ρ_AB the violation is

        viol(ρ_AB) = max(0,  λ_max(ρ_AB) - λ_2(ρ_AB) - 2√(λ_1 λ_3))

    where λ_1 ≤ λ_2 ≤ λ_3 ≤ λ_max are the ordered eigenvalues of ρ_AB.
    A state with zero total violation across all bipartitions is entanglement-
    separable and necessarily a stabilizer state (SRE = 0).

    Parametrisation
    ---------------
    JAX operates on real parameter vectors of length 2·2^N, obtained by
    stacking the real and imaginary parts of the complex state vector.

    vmapping
    --------
    Both the metric computation and the solver step are vmapped over the
    batch dimension so all `num_starts` trajectories are optimised in parallel
    on the GPU.
    """

    def run_jax_gpu_optimization(
        n_qubits: int,
        num_starts: int,
        num_loops: int,
        gap: int,
        step_mes: int = 1,
        filename: str = "jax_trajectory_data.pkl",
    ) -> dict:
        """
        Run batched L-BFGS optimisation and record SRE trajectories.

        Parameters
        ----------
        n_qubits   : system size N.
        num_starts : number of parallel trajectories (batch size).
        num_loops  : total number of L-BFGS micro-iterations.
        gap        : stabilizer gap of the initial states.
        step_mes   : number of L-BFGS steps between metric snapshots.
        filename   : path to save the resulting trajectory dictionary.

        Returns
        -------
        trajectories : dict with keys
            'average_purity'  – list[list[float]]  mean purity over bipartitions
            'max_purity'      – list[list[float]]  max purity over bipartitions
            'sre'             – list[list[float]]  M_2 SRE per state
            'total_violation' – list[list[float]]  total squared violation
            'final_states'    – list[np.ndarray]   complex state vectors
        """
        n     = n_qubits
        k     = n // 2          # bipartition size
        n_dim = 2 ** n

        # Enumerate all equal bipartitions of size k.
        combos = list(combinations(range(n), k))

        # For each bipartition, build a permutation that moves the traced-out
        # indices to the front so reshape(-1, 2^k) gives the correct reduced
        # density matrix.
        perms_list = []
        for combo in combos:
            keep  = list(combo)
            trace = [i for i in range(n) if i not in keep]
            perms_list.append(tuple(trace + keep))

        # ---- JAX-jitted metric computation ----------------------------------

        @jax.jit
        def get_purity_and_violation(psi_vec):
            """
            From a real parameter vector (real | imag) compute:
            - average purity across bipartitions
            - maximum purity across bipartitions
            - total squared Hildebrand violation
            """
            # Reconstruct and normalise the complex state vector.
            psi        = psi_vec[:n_dim] + 1j * psi_vec[n_dim:]
            psi        = psi / jnp.linalg.norm(psi)
            psi_tensor = psi.reshape((2,) * n)

            # Compute all reduced density matrices ρ_AB = Tr_A |ψ⟩⟨ψ| for each bipartition.
            rhos = [
                psi_tensor.transpose(perm).reshape(-1, 2 ** k).conj().T
                @ psi_tensor.transpose(perm).reshape(-1, 2 ** k)
                for perm in perms_list
            ]
            batch_rho = jnp.stack(rhos, axis=0)           # (n_bipartitions, 2^k, 2^k)
            eigenvalues = jnp.linalg.eigvalsh(batch_rho)  # (n_bipartitions, 2^k)

            purities    = jnp.sum(eigenvalues ** 2, axis=-1)
            avg_purity  = jnp.mean(purities)
            max_purity  = jnp.max(purities)

            # Hildebrand violation per bipartition:
            #   max(0,  λ_max - λ_{max-1} - 2√(λ_min · λ_{max-2}))
            # (only defined for 2×2 blocks, i.e. k=1, but generalised here to
            # the three smallest eigenvalues of the reduced density matrix.)
            rhs  = eigenvalues[:, 1] + 2 * jnp.sqrt(
                jnp.maximum(eigenvalues[:, 0] * eigenvalues[:, 2], 1e-15)
            )
            viols           = jnp.maximum(0.0, eigenvalues[:, -1] - rhs)
            total_violation = jnp.sum(viols ** 2)

            return avg_purity, max_purity, total_violation

        @jax.jit
        def objective(params):
            """Scalar loss: total squared Hildebrand violation (to be minimised)."""
            _, _, total_viol = get_purity_and_violation(params)
            return total_viol

        # L-BFGS solver; `maxiter=step_mes` so one `.run()` call = one chunk.
        solver = LBFGS(fun=objective, maxiter=step_mes, tol=1e-11)

        # ---- Initialise batch of states ------------------------------------
        print(f"Generating {num_starts} initial states (gap={gap})…")
        init_params_list = []
        for _ in range(num_starts):
            psi_init = rand_Almost_Stab_state(n, gap)
            psi_np   = psi_init.get() if hasattr(psi_init, "get") else np.array(psi_init)
            # Stack real and imaginary parts into a flat real vector for JAX.
            p = np.concatenate([np.real(psi_np), np.imag(psi_np)])
            init_params_list.append(p)
        params_batch = jnp.array(init_params_list, dtype=jnp.float64)

        # vmap both the metric function and the solver over the batch axis.
        vmapped_metrics = jax.vmap(get_purity_and_violation)
        vmapped_run     = jax.vmap(solver.run)

        # Per-trajectory history lists (one entry per snapshot).
        traj_avg_p = [[] for _ in range(num_starts)]
        traj_max_p = [[] for _ in range(num_starts)]
        traj_viol  = [[] for _ in range(num_starts)]
        traj_sre   = [[] for _ in range(num_starts)]

        # ---- Step 0: record metrics for the initial states -----------------
        start_time = time.time()
        avg_p, max_p, viol = vmapped_metrics(params_batch)
        avg_p_cpu, max_p_cpu, viol_cpu = (
            np.array(avg_p), np.array(max_p), np.array(viol)
        )
        states_complex = params_batch[:, :n_dim] + 1j * params_batch[:, n_dim:]
        sre_vals_cpu   = compute_sre_native_cupy_batch(states_complex, alpha=2)

        for i in range(num_starts):
            traj_avg_p[i].append(float(avg_p_cpu[i]))
            traj_max_p[i].append(float(max_p_cpu[i]))
            traj_viol[i].append(float(viol_cpu[i]))
            traj_sre[i].append(float(sre_vals_cpu[i]))

        # ---- Optimisation loop (chunked for periodic SRE snapshots) --------
        num_chunks = max(1, num_loops // step_mes)

        for chunk_idx in range(1, num_chunks + 1):
            # Run `step_mes` L-BFGS steps for all trajectories simultaneously.
            res          = vmapped_run(params_batch)
            params_batch = res.params

            # Compute and record metrics for this snapshot.
            avg_p, max_p, viol = vmapped_metrics(params_batch)
            avg_p_cpu, max_p_cpu, viol_cpu = (
                np.array(avg_p), np.array(max_p), np.array(viol)
            )
            states_complex = params_batch[:, :n_dim] + 1j * params_batch[:, n_dim:]
            sre_vals_cpu   = compute_sre_native_cupy_batch(states_complex, alpha=2)

            for i in range(num_starts):
                traj_avg_p[i].append(float(avg_p_cpu[i]))
                traj_max_p[i].append(float(max_p_cpu[i]))
                traj_viol[i].append(float(viol_cpu[i]))
                traj_sre[i].append(float(sre_vals_cpu[i]))

            # Progress report every chunk.
            elapsed = time.time() - start_time
            est_rem = (elapsed / chunk_idx) * (num_chunks - chunk_idx)
            print(
                f"  Step {chunk_idx * step_mes:5d}/{num_loops} | "
                f"Mean Violation: {float(np.mean(viol_cpu)):.2e} | "
                f"Mean SRE: {float(np.mean(sre_vals_cpu)):.4f} | "
                f"Elapsed: {elapsed:.1f}s | "
                f"Est. Remaining: {est_rem:.1f}s"
            )

        # ---- Collect final states and save ---------------------------------
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

    return (run_jax_gpu_optimization,)


# ---------------------------------------------------------------------------
# Cell 8 – Utility: pack multiple .pkl files into one archive
# ---------------------------------------------------------------------------
@app.cell
def _(os, pickle):
    """
    Utility for combining per-gap output files into a single archive.

    The archive is itself a pickle whose top-level value is a dict mapping
    original filenames to their loaded contents.  This makes it easy to
    transfer a full run (all gap values) as a single file.
    """

    def pack_pkl_files(source_dir_or_files, output_archive_path: str) -> None:
        """
        Combine multiple .pkl files into one merged archive.

        Parameters
        ----------
        source_dir_or_files : str | list[str]
            Either a directory path (all .pkl files inside are packed in
            sorted order) or an explicit list of file paths.
        output_archive_path : str
            Destination path for the merged .pkl archive.
        """
        # Resolve the list of source files.
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

        # Load each file and accumulate into the merged dict.
        packed_data = {}
        for fpath in file_paths:
            fname = os.path.basename(fpath)
            try:
                with open(fpath, "rb") as f:
                    packed_data[fname] = pickle.load(f)
                print(f"  Packed: {fname}")
            except Exception as exc:
                print(f"  Error reading {fname}: {exc}")

        # Ensure the output directory exists.
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
# Cell 9 – Data generation driver: iterate over stabilizer gaps
# ---------------------------------------------------------------------------
@app.cell
def _(os, pack_pkl_files, run_jax_gpu_optimization):
    """
    Iterate over all stabilizer-gap values and call the optimiser for each one.

    Gap range: [5, N] (inclusive), so the 'almost-stabilizer' parameter
    ranges from gap=0 (pure stabilizer) down to gap=N-5.

    Each gap produces one .pkl file.  If a file already exists it is skipped,
    allowing interrupted runs to resume.  After each gap the individual files
    are merged into a running archive via `pack_pkl_files`.
    """

    def gen_data(
        n_qubits: int,
        num_seeds: int,
        num_steps: int,
        step_mes: int,
        f_name: str,
    ) -> None:
        """
        Run the optimiser for every stabilizer-gap value and pack results.

        Parameters
        ----------
        n_qubits  : system size N.
        num_seeds : number of parallel trajectories per gap.
        num_steps : total L-BFGS micro-iterations per gap.
        step_mes  : snapshot interval (chunk size).
        f_name    : filename prefix; individual files are named
                    <f_name><num_steps>_stps_<gap>.pkl.
        """
        for _stab_qubits in range(5, n_qubits + 1):
            gap      = n_qubits - _stab_qubits   # number of 'free' qubits
            out_file = f"{f_name}{num_steps}_stps_{gap}.pkl"

            if os.path.exists(out_file):
                print(f"Skipping gap={gap}: '{out_file}' already exists.")
                continue

            print(f"\nComputing {n_qubits}-qubit, gap={gap} "
                  f"({_stab_qubits}-qubit stabilized initial state).")

            run_jax_gpu_optimization(
                n_qubits  = n_qubits,
                num_starts= num_seeds,
                num_loops = num_steps,
                gap       = gap,
                step_mes  = step_mes,
                filename  = out_file,
            )

            # Update the merged archive after each gap completes.
            pack_pkl_files("zip1", "file1.pkl")

    return (gen_data,)


# ---------------------------------------------------------------------------
# Cell 10 – Orchestration: read UI values, run simulation, display status
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
    """
    Orchestration cell: reads the UI parameters, creates the output directory,
    and calls gen_data.

    In interactive mode execution is gated by `run_button` to prevent
    accidental restarts.  In script mode (e.g. `uv run Generate_States.py`)
    the simulation starts automatically using the default slider values.
    """
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
        print(f"Running simulation: {N} Qubits | {seeds} Seeds | "
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
