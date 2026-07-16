"""Direct Hermitian and non-Hermitian evolution through the tidal resonance."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from .models import (
    EvolutionConfig,
    EvolutionResult,
    PRIMARY_FINAL,
    PRIMARY_INITIAL,
)
from .spectrum import gamma_detweiler_M, omega_real_M
from .tides import (
    PRIMARY_DELTA_M,
    _tidal_eta_at_omega_M_unchecked,
    tidal_eta_at_omega_M,
)
from .timescales import gw_frequency_sweep_M2
from .validation import validate_alpha_chi, validate_q


def _spectrum_values(config: EvolutionConfig) -> tuple[complex, complex]:
    if config.omega_initial_M is not None or config.omega_final_M is not None:
        if config.omega_initial_M is None or config.omega_final_M is None:
            raise ValueError("both complex frequency overrides must be supplied")
        return complex(config.omega_initial_M), complex(config.omega_final_M)
    if config.spectrum_model == "hydrogenic_detweiler":
        initial = complex(
            omega_real_M(config.alpha, config.chi, PRIMARY_INITIAL),
            gamma_detweiler_M(config.alpha, config.chi, PRIMARY_INITIAL),
        )
        final = complex(
            omega_real_M(config.alpha, config.chi, PRIMARY_FINAL),
            gamma_detweiler_M(config.alpha, config.chi, PRIMARY_FINAL),
        )
        return initial, final
    if config.spectrum_model == "continued_fraction":
        from .relativity import quasibound_frequency_cf

        return (
            quasibound_frequency_cf(config.alpha, config.chi, PRIMARY_INITIAL),
            quasibound_frequency_cf(config.alpha, config.chi, PRIMARY_FINAL),
        )
    raise ValueError("unsupported spectrum_model")


def _initial_state(config: EvolutionConfig) -> np.ndarray:
    if config.initial_admixture < 0.0:
        raise ValueError("initial_admixture must be non-negative")
    state = np.asarray(
        [
            1.0 + 0.0j,
            config.initial_admixture * np.exp(1j * config.initial_phase),
        ],
        dtype=complex,
    )
    return state / np.linalg.norm(state)


class _EvolutionProblem:
    def __init__(
        self,
        config: EvolutionConfig,
        omega_initial: complex,
        omega_final: complex,
        delta_gamma: float,
    ) -> None:
        self.config = config
        self.energy_gap = omega_final.real - omega_initial.real
        self.delta_m = PRIMARY_DELTA_M
        self.omega_res = self.energy_gap / self.delta_m
        if self.omega_res <= 0.0:
            raise ValueError("the selected spectrum gives a non-positive resonance")
        self.delta_gamma = float(delta_gamma)
        self.sweep_res = abs(self.delta_m) * gw_frequency_sweep_M2(
            self.omega_res, config.q, order=config.chirp_order
        )
        self.sqrt_sweep = math.sqrt(self.sweep_res)
        self.eta_res = config.coupling_scale * tidal_eta_at_omega_M(
            config.alpha, config.chi, config.q, self.omega_res
        )
        raw_edge = config.edge_factor * max(
            self.eta_res, abs(self.delta_gamma), self.sqrt_sweep
        )
        self.detuning_edge = min(raw_edge, 0.8 * abs(self.energy_gap))
        if self.detuning_edge <= 0.0:
            raise ValueError("the integration window is empty")
        self.x_edge = self.detuning_edge / self.sqrt_sweep

    def omega_from_x(self, x_value: float) -> float:
        detuning = x_value * self.sqrt_sweep
        omega = (self.energy_gap - detuning) / self.delta_m
        if omega <= 0.0:
            raise ValueError("integration window reached non-positive orbital frequency")
        return float(omega)

    def dt_dx(self, x_value: float) -> float:
        omega = self.omega_from_x(x_value)
        delta_dot = abs(self.delta_m) * gw_frequency_sweep_M2(
            omega, self.config.q, order=self.config.chirp_order
        )
        return self.sqrt_sweep / delta_dot

    def generator(self, x_value: float, delta_gamma: float) -> np.ndarray:
        omega = self.omega_from_x(x_value)
        detuning = x_value * self.sqrt_sweep
        eta = self.config.coupling_scale * _tidal_eta_at_omega_M_unchecked(
            self.config.alpha, self.config.chi, self.config.q, omega
        )
        # A piecewise scalar gauge removes the large phase of the diabatic
        # state populated on each side of the crossing.  It leaves every
        # population and the non-Hermitian norm balance unchanged, while
        # avoiding millions of steps spent resolving an irrelevant phase.
        if x_value <= 0.0:
            hamiltonian = np.asarray(
                [[0.0, eta], [eta, -detuning - 1j * delta_gamma]],
                dtype=complex,
            )
        else:
            hamiltonian = np.asarray(
                [[detuning, eta], [eta, -1j * delta_gamma]],
                dtype=complex,
            )
        return -1j * hamiltonian * self.dt_dx(x_value)


def _loss_integrand(
    problem: _EvolutionProblem,
    x_value: float,
    state: np.ndarray,
    delta_gamma: float,
) -> float:
    return (
        2.0
        * delta_gamma
        * abs(state[1]) ** 2
        * problem.dt_dx(x_value)
    )


def _cf4_propagator(
    problem: _EvolutionProblem,
    x_value: float,
    step: float,
    delta_gamma: float,
) -> np.ndarray:
    """Fourth-order commutator-free exponential propagator."""
    root_three = math.sqrt(3.0)
    node_1 = 0.5 - root_three / 6.0
    node_2 = 0.5 + root_three / 6.0
    coefficient_1 = (3.0 - 2.0 * root_three) / 12.0
    coefficient_2 = (3.0 + 2.0 * root_three) / 12.0
    generator_1 = problem.generator(x_value + node_1 * step, delta_gamma)
    generator_2 = problem.generator(x_value + node_2 * step, delta_gamma)
    first = coefficient_1 * generator_1 + coefficient_2 * generator_2
    second = coefficient_2 * generator_1 + coefficient_1 * generator_2
    return expm(second * step) @ expm(first * step)


def _propagate_exponential(
    problem: _EvolutionProblem,
    initial_state: np.ndarray,
    delta_gamma: float,
    store_trajectory: bool,
) -> tuple[np.ndarray, float, int, int, dict[str, np.ndarray], bool]:
    x_start = -problem.x_edge
    x_stop = problem.x_edge
    x_value = x_start
    state = initial_state.copy()
    span = x_stop - x_start
    step = min(span / 128.0, 1.0)
    maximum_step = 100.0
    accepted = 0
    rejected = 0
    integrated_loss = 0.0
    xs = [x_value]
    states = [state.copy()]
    converged = True

    while x_value < x_stop:
        if accepted + rejected >= problem.config.max_steps:
            converged = False
            break
        step = min(step, maximum_step, x_stop - x_value)
        full_propagator = _cf4_propagator(
            problem, x_value, step, delta_gamma
        )
        full = full_propagator @ state

        first_propagator = _cf4_propagator(
            problem, x_value, 0.5 * step, delta_gamma
        )
        midpoint = first_propagator @ state
        second_propagator = _cf4_propagator(
            problem, x_value + 0.5 * step, 0.5 * step, delta_gamma
        )
        two_half_propagator = second_propagator @ first_propagator
        two_half = two_half_propagator @ state

        overlap = np.vdot(full, two_half)
        phase = overlap / abs(overlap) if abs(overlap) > 0.0 else 1.0 + 0.0j
        error = float(np.linalg.norm(two_half - phase * full))
        tolerance = problem.config.atol + problem.config.rtol * float(
            np.linalg.norm(two_half)
        )
        if error <= tolerance or abs(step) <= 1.0e-12:
            if delta_gamma > 0.0:
                f_start = _loss_integrand(
                    problem, x_value, state, delta_gamma
                )
                f_middle = _loss_integrand(
                    problem, x_value + 0.5 * step, midpoint, delta_gamma
                )
                f_stop = _loss_integrand(
                    problem, x_value + step, two_half, delta_gamma
                )
                integrated_loss += step * (f_start + 4.0 * f_middle + f_stop) / 6.0
            x_value += step
            state = two_half
            accepted += 1
            if store_trajectory:
                xs.append(x_value)
                states.append(state.copy())
            factor = 1.5 if error == 0.0 else min(1.5, 0.9 * (tolerance / error) ** (1.0 / 5.0))
            step *= max(1.05, factor)
        else:
            rejected += 1
            step *= max(0.1, 0.9 * (tolerance / error) ** (1.0 / 5.0))

    if not store_trajectory:
        xs = [x_start, x_value]
        states = [initial_state.copy(), state.copy()]
    state_array = np.asarray(states, dtype=complex)
    trajectory = {
        "x": np.asarray(xs, dtype=float),
        "state": state_array,
    }
    return state, integrated_loss, accepted, rejected, trajectory, converged


def _propagate_dop853(
    problem: _EvolutionProblem,
    initial_state: np.ndarray,
    delta_gamma: float,
) -> np.ndarray | None:
    # Direct Runge-Kutta is intentionally restricted to windows where resolving
    # the rapidly accumulating diabatic phase remains a useful cross-check.
    if problem.x_edge > 2_500.0:
        return None

    def derivative(x_value: float, real_state: np.ndarray) -> np.ndarray:
        complex_state = real_state[:2] + 1j * real_state[2:]
        result = problem.generator(x_value, delta_gamma) @ complex_state
        return np.concatenate((result.real, result.imag))

    y0 = np.concatenate((initial_state.real, initial_state.imag))
    solution = solve_ivp(
        derivative,
        (-problem.x_edge, problem.x_edge),
        y0,
        method="DOP853",
        rtol=problem.config.rtol,
        atol=problem.config.atol,
    )
    if not solution.success:
        return None
    final = solution.y[:, -1]
    return final[:2] + 1j * final[2:]


def _propagate_density_radau(
    problem: _EvolutionProblem,
    initial_state: np.ndarray,
    delta_gamma: float,
    store_trajectory: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    """Propagate the phase-free density variables with an implicit solver."""
    coherence = np.conjugate(initial_state[0]) * initial_state[1]
    y0 = np.asarray(
        [
            abs(initial_state[0]) ** 2,
            abs(initial_state[1]) ** 2,
            2.0 * coherence.real,
            2.0 * coherence.imag,
            0.0,
        ],
        dtype=float,
    )

    def derivative(x_value: float, values: np.ndarray) -> np.ndarray:
        p_initial, p_final, coherence_real, coherence_imag, _ = values
        omega = problem.omega_from_x(x_value)
        detuning = x_value * problem.sqrt_sweep
        eta = problem.config.coupling_scale * _tidal_eta_at_omega_M_unchecked(
            problem.config.alpha,
            problem.config.chi,
            problem.config.q,
            omega,
        )
        dt_dx = problem.dt_dx(x_value)
        return dt_dx * np.asarray(
            [
                eta * coherence_imag,
                -eta * coherence_imag - 2.0 * delta_gamma * p_final,
                -delta_gamma * coherence_real - detuning * coherence_imag,
                detuning * coherence_real
                - delta_gamma * coherence_imag
                + 2.0 * eta * (p_final - p_initial),
                2.0 * delta_gamma * p_final,
            ]
        )

    solution = solve_ivp(
        derivative,
        (-problem.x_edge, problem.x_edge),
        y0,
        method="Radau",
        rtol=problem.config.rtol,
        atol=problem.config.atol,
    )
    indices = np.arange(solution.t.size)
    if not store_trajectory:
        indices = np.asarray([0, solution.t.size - 1])
    elif solution.t.size > 5_000:
        indices = np.unique(np.linspace(0, solution.t.size - 1, 5_000).astype(int))
    populations = solution.y[:2, indices].T
    norms = populations.sum(axis=1)
    return (
        solution.t[indices],
        populations,
        norms,
        float(solution.y[4, -1]),
        max(0, solution.t.size - 1),
        bool(solution.success),
    )


def evolve_transition(config: EvolutionConfig) -> EvolutionResult:
    """Evolve the primary transition and compare four dynamical models."""
    validate_alpha_chi(config.alpha, config.chi)
    validate_q(config.q)
    if config.edge_factor <= 0.0:
        raise ValueError("edge_factor must be positive")
    if config.rtol <= 0.0 or config.atol <= 0.0:
        raise ValueError("integration tolerances must be positive")
    if config.coupling_scale < 0.0:
        raise ValueError("coupling_scale must be non-negative")
    if config.chirp_order not in {"newtonian", "1pn"}:
        raise ValueError("unsupported chirp_order")

    omega_initial, omega_final = _spectrum_values(config)
    delta_gamma = omega_initial.imag - omega_final.imag
    if delta_gamma <= 0.0:
        raise ValueError("the primary model requires Gamma_211 > Gamma_21-1")
    problem = _EvolutionProblem(config, omega_initial, omega_final, delta_gamma)
    initial_state = _initial_state(config)
    z_value = problem.eta_res**2 / problem.sweep_res

    # At very adiabatic physical EPs, amplitude propagation spends almost all
    # work resolving an irrelevant global phase.  The density variables remove
    # that phase; Radau then performs the full non-Hermitian evolution.  The
    # Hermitian comparator uses its asymptotic LZ value in this branch because
    # the physical scan has z>200 and its survival is far below double range.
    if (
        problem.x_edge > config.direct_scaled_edge_limit
        or z_value > config.direct_z_limit
    ):
        survival_lz = float(math.exp(-2.0 * math.pi * z_value))
        density = _propagate_density_radau(
            problem, initial_state, delta_gamma, config.store_trajectory
        )
        x_values, populations, norms, integrated_loss, steps, converged = density
        omega_values = np.asarray(
            [problem.omega_from_x(float(value)) for value in x_values]
        )
        p_initial_raw = float(populations[-1, 0])
        p_final_raw = float(populations[-1, 1])
        p_initial = min(1.0, max(0.0, p_initial_raw))
        p_final = min(1.0, max(0.0, p_final_raw))
        norm_exit = p_initial + p_final
        balance = abs(1.0 - (p_initial_raw + p_final_raw) - integrated_loss)
        effect = abs(p_initial - survival_lz)
        status = "formal_effect" if effect >= 0.10 else "effect_below_threshold"
        if not converged:
            status = "integration_not_converged"
        return EvolutionResult(
            alpha=config.alpha,
            chi=config.chi,
            q=config.q,
            status=status,
            spectrum_model=config.spectrum_model,
            chirp_order=config.chirp_order,
            omega_res_M=problem.omega_res,
            detuning_edge_M=problem.detuning_edge,
            scaled_edge=problem.x_edge,
            eta_res_M=problem.eta_res,
            delta_gamma_M=delta_gamma,
            sweep_rate_M2=problem.sweep_res,
            landau_zener_z=z_value,
            survival_nh=p_initial,
            survival_hermitian=survival_lz,
            survival_factorized=survival_lz,
            survival_lz=survival_lz,
            initial_population_exit=p_initial,
            final_population_exit=p_final,
            norm_exit=norm_exit,
            absorbed_asymptotic=1.0 - p_initial,
            effect_abs=effect,
            probability_balance_residual=balance,
            solver_error=None,
            accepted_steps=steps,
            rejected_steps=0,
            x=x_values,
            omega_M=omega_values,
            initial_population=np.clip(populations[:, 0], 0.0, 1.0),
            final_population=np.clip(populations[:, 1], 0.0, 1.0),
            norm=np.clip(norms, 0.0, 1.0),
        )

    nh = _propagate_exponential(
        problem, initial_state, delta_gamma, config.store_trajectory
    )
    hermitian = _propagate_exponential(
        problem, initial_state, 0.0, False
    )
    nh_state, integrated_loss, accepted, rejected, trajectory, converged = nh
    hermitian_state = hermitian[0]

    p_initial = float(abs(nh_state[0]) ** 2)
    p_final = float(abs(nh_state[1]) ** 2)
    norm_exit = p_initial + p_final
    survival_hermitian = float(abs(hermitian_state[0]) ** 2)
    survival_factorized = survival_hermitian
    survival_lz = float(math.exp(-2.0 * math.pi * z_value))
    balance_residual = abs((1.0 - norm_exit) - integrated_loss)

    solver_error: float | None = None
    if config.crosscheck:
        dop853_state = _propagate_dop853(problem, initial_state, delta_gamma)
        if dop853_state is not None:
            solver_error = float(
                max(
                    abs(abs(dop853_state[0]) ** 2 - p_initial),
                    abs(abs(dop853_state[1]) ** 2 - p_final),
                )
            )

    x_values = trajectory["x"]
    state_values = trajectory["state"]
    omega_values = np.asarray(
        [problem.omega_from_x(float(value)) for value in x_values], dtype=float
    )
    initial_populations = np.abs(state_values[:, 0]) ** 2
    final_populations = np.abs(state_values[:, 1]) ** 2
    norms = initial_populations + final_populations
    effect = abs(p_initial - survival_factorized)
    status = "formal_effect" if effect >= 0.10 else "effect_below_threshold"
    if not converged:
        status = "integration_not_converged"

    return EvolutionResult(
        alpha=config.alpha,
        chi=config.chi,
        q=config.q,
        status=status,
        spectrum_model=config.spectrum_model,
        chirp_order=config.chirp_order,
        omega_res_M=problem.omega_res,
        detuning_edge_M=problem.detuning_edge,
        scaled_edge=problem.x_edge,
        eta_res_M=problem.eta_res,
        delta_gamma_M=delta_gamma,
        sweep_rate_M2=problem.sweep_res,
        landau_zener_z=z_value,
        survival_nh=p_initial,
        survival_hermitian=survival_hermitian,
        survival_factorized=survival_factorized,
        survival_lz=survival_lz,
        initial_population_exit=p_initial,
        final_population_exit=p_final,
        norm_exit=norm_exit,
        absorbed_asymptotic=max(0.0, min(1.0, 1.0 - p_initial)),
        effect_abs=effect,
        probability_balance_residual=balance_residual,
        solver_error=solver_error,
        accepted_steps=accepted,
        rejected_steps=rejected,
        x=x_values,
        omega_M=omega_values,
        initial_population=initial_populations,
        final_population=final_populations,
        norm=norms,
    )


def window_convergence_error(config: EvolutionConfig) -> float:
    """Return the maximum survival change across edge factors 50/100/200."""
    results = [
        evolve_transition(
            replace(
                config,
                edge_factor=factor,
                store_trajectory=False,
                crosscheck=False,
            )
        )
        for factor in (50.0, 100.0, 200.0)
    ]
    return max(
        abs(results[1].survival_nh - results[0].survival_nh),
        abs(results[2].survival_nh - results[1].survival_nh),
    )
