"""Tests for the simulation engine and SimulationResult."""

import numpy as np
import pytest

from nnsimviz.configs import (
    NetworkConfig,
    SimulationConfig,
    ProjectConfig,
    VALID_INTEGRATION_METHODS,
)
from nnsimviz.models import build_weight_matrix
from nnsimviz.simulation import (
    Simulator,
    SimulationResult,
    _rhs,
    _euler_step,
    _heun_step,
    _rk4_step,
)


def _make_config(n=10, **sim_kwargs):
    return ProjectConfig(
        network=NetworkConfig(n_neurons=n, random_seed=42),
        simulation=SimulationConfig(**sim_kwargs),
    )


# =========================================================================== #
# Original 8 tests (kept intact)
# =========================================================================== #

def test_result_shapes():
    cfg = _make_config(n=10, duration=5.0, dt=0.1)
    W = build_weight_matrix(cfg.network)
    result = Simulator().run(cfg, W)
    assert isinstance(result, SimulationResult)
    assert result.activity.shape == (10, cfg.simulation.n_steps)
    assert result.final_state.shape == (10,)
    assert result.time.shape == (cfg.simulation.n_steps,)


def test_final_state_matches_last_column():
    cfg = _make_config(n=8)
    W = build_weight_matrix(cfg.network)
    result = Simulator().run(cfg, W)
    assert np.array_equal(result.final_state, result.activity[:, -1])


def test_simulation_is_finite():
    cfg = _make_config(n=20, duration=20.0, dt=0.1, noise_level=0.1)
    W = build_weight_matrix(cfg.network)
    result = Simulator().run(cfg, W)
    assert np.isfinite(result.activity).all()


def test_clipping_prevents_explosion():
    # Large weights would blow up without clipping.
    cfg = _make_config(n=30, duration=30.0, dt=0.5)
    cfg.network.weight_scale = 3.0
    W = build_weight_matrix(cfg.network)
    result = Simulator().run(cfg, W)
    assert np.abs(result.activity).max() <= Simulator.CLIP


@pytest.mark.parametrize("input_type", SimulationConfig.VALID_INPUT_TYPES)
def test_all_input_types_run(input_type):
    cfg = _make_config(n=10, input_type=input_type)
    W = build_weight_matrix(cfg.network)
    result = Simulator().run(cfg, W)
    assert np.isfinite(result.activity).all()


def test_metadata_populated():
    cfg = _make_config(n=14)
    W = build_weight_matrix(cfg.network)
    result = Simulator().run(cfg, W)
    assert result.metadata["n_neurons"] == 14
    assert result.metadata["n_steps"] == cfg.simulation.n_steps


def test_shape_mismatch_raises():
    cfg = _make_config(n=10)
    wrong_W = np.zeros((5, 5))
    with pytest.raises(ValueError):
        Simulator().run(cfg, wrong_W)


def test_reproducible_runs():
    cfg = _make_config(n=12)
    W = build_weight_matrix(cfg.network)
    r1 = Simulator().run(cfg, W)
    r2 = Simulator().run(cfg, W)
    assert np.array_equal(r1.activity, r2.activity)


# =========================================================================== #
# New tests: Euler step function correctness
# =========================================================================== #

def test_euler_single_step_correctness():
    """_euler_step matches the expected analytic x + dt*f(x) + noise."""
    rng = np.random.default_rng(0)
    n, dt = 5, 0.1
    W = np.eye(n) * 0.5
    x = rng.standard_normal(n)
    u = rng.standard_normal(n)
    noise = rng.standard_normal(n) * 0.01
    expected = x + dt * _rhs(x, u, W) + noise
    result = _euler_step(x, u, W, dt, noise)
    np.testing.assert_allclose(result, expected)


def test_euler_zero_noise_no_drift_in_fixed_point():
    """At a deterministic fixed point with zero noise, Euler stays put."""
    # Fixed point: x = W @ tanh(x) + u  i.e. _rhs == 0
    # Simplest: W=0, u=0 => x=0 is a fixed point
    n, dt = 4, 0.05
    x = np.zeros(n)
    u = np.zeros(n)
    W = np.zeros((n, n))
    noise = np.zeros(n)
    result = _euler_step(x, u, W, dt, noise)
    np.testing.assert_array_equal(result, x)


def test_euler_metadata_records_method():
    cfg = _make_config(n=6, integration_method="euler")
    W = build_weight_matrix(cfg.network)
    result = Simulator().run(cfg, W)
    assert result.metadata["integration_method"] == "euler"


# =========================================================================== #
# New tests: noise scaling (Euler-Maruyama sqrt(dt))
# =========================================================================== #

def test_noise_scales_with_sqrt_dt():
    """Coarser dt produces higher per-step variance; correct SDE behavior."""
    def trajectory_variance(dt):
        cfg = _make_config(n=5, duration=5.0, dt=dt, noise_level=0.5,
                           input_type="none", integration_method="euler")
        W = np.zeros((5, 5))  # no recurrent coupling -> pure noise walk
        results = [Simulator().run(cfg, W).activity for _ in range(30)]
        return np.var(np.stack(results, axis=0), axis=0).mean()

    var_coarse = trajectory_variance(0.5)
    var_fine   = trajectory_variance(0.05)
    # coarser dt => fewer steps, higher per-step noise amplitude => larger total variance
    assert var_coarse > var_fine


# =========================================================================== #
# New tests: all methods run and produce correct shapes
# =========================================================================== #

@pytest.mark.parametrize("method", VALID_INTEGRATION_METHODS)
def test_all_methods_run_and_finite(method):
    cfg = _make_config(n=10, duration=5.0, dt=0.1, integration_method=method)
    W = build_weight_matrix(cfg.network)
    result = Simulator().run(cfg, W)
    assert np.isfinite(result.activity).all()


@pytest.mark.parametrize("method", VALID_INTEGRATION_METHODS)
def test_all_methods_result_shapes(method):
    cfg = _make_config(n=8, duration=4.0, dt=0.1, integration_method=method)
    W = build_weight_matrix(cfg.network)
    result = Simulator().run(cfg, W)
    assert result.activity.shape == (8, cfg.simulation.n_steps)
    assert result.final_state.shape == (8,)


# =========================================================================== #
# New tests: Heun and RK4 accuracy
# =========================================================================== #

def test_heun_more_accurate_than_euler_on_linear_system():
    """On dx/dt = -x (analytic: exp(-t)), RK4 < Heun < Euler global error."""
    # scalar linear system via 1-neuron zero-weight network, constant input=0
    n, dt = 1, 0.2
    W = np.zeros((n, n))
    x0 = np.array([1.0])
    duration = 2.0
    n_steps = int(duration / dt)

    def run_method(method):
        x = x0.copy()
        noise = np.zeros(n)
        u = np.zeros(n)
        step_fn = {"euler": _euler_step, "heun": _heun_step, "rk4": _rk4_step}[method]
        for _ in range(n_steps):
            x = step_fn(x, u, W, dt, noise)
        return x[0]

    analytic = np.exp(-duration)
    err_euler = abs(run_method("euler") - analytic)
    err_heun  = abs(run_method("heun")  - analytic)
    err_rk4   = abs(run_method("rk4")   - analytic)

    assert err_rk4 < err_heun < err_euler


def test_heun_single_step_correctness():
    """_heun_step matches the explicit trapezoidal formula."""
    rng = np.random.default_rng(1)
    n, dt = 4, 0.1
    W = rng.standard_normal((n, n)) * 0.3
    x = rng.standard_normal(n)
    u = rng.standard_normal(n)
    noise = rng.standard_normal(n) * 0.01

    k1 = _rhs(x, u, W)
    x_pred = x + dt * k1
    k2 = _rhs(x_pred, u, W)
    expected = x + dt * 0.5 * (k1 + k2) + noise

    np.testing.assert_allclose(_heun_step(x, u, W, dt, noise), expected)


def test_rk4_single_step_correctness():
    """_rk4_step matches the explicit 4-stage formula."""
    rng = np.random.default_rng(2)
    n, dt = 4, 0.1
    W = rng.standard_normal((n, n)) * 0.3
    x = rng.standard_normal(n)
    u = rng.standard_normal(n)
    noise = rng.standard_normal(n) * 0.01

    k1 = _rhs(x,               u, W)
    k2 = _rhs(x + 0.5*dt*k1,  u, W)
    k3 = _rhs(x + 0.5*dt*k2,  u, W)
    k4 = _rhs(x +      dt*k3, u, W)
    expected = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4) + noise

    np.testing.assert_allclose(_rk4_step(x, u, W, dt, noise), expected)


def test_unknown_method_raises():
    cfg = _make_config(n=5)
    cfg.simulation.integration_method = "bogus"  # bypass validate
    W = build_weight_matrix(cfg.network)
    with pytest.raises(ValueError, match="Unknown integration_method"):
        Simulator().run(cfg, W)


# =========================================================================== #
# New tests: convergence_eps
# =========================================================================== #

def test_convergence_eps_terminates_early():
    """A pure-decay network (W=0, no input, no noise) converges before duration."""
    cfg = _make_config(
        n=5, duration=50.0, dt=0.1,
        input_type="none", noise_level=0.0,
        integration_method="euler", convergence_eps=1e-6,
    )
    W = np.zeros((5, 5))
    result = Simulator().run(cfg, W)
    assert result.metadata["converged_at"] is not None
    assert result.metadata["converged_at"] < cfg.simulation.n_steps - 1


def test_no_convergence_without_eps():
    cfg = _make_config(n=5, duration=5.0, dt=0.1)
    W = build_weight_matrix(cfg.network)
    result = Simulator().run(cfg, W)
    assert result.metadata["converged_at"] is None


def test_convergence_filled_columns_are_constant():
    """All columns after convergence must be identical."""
    cfg = _make_config(
        n=4, duration=30.0, dt=0.1,
        input_type="none", noise_level=0.0,
        integration_method="euler", convergence_eps=1e-6,
    )
    W = np.zeros((4, 4))
    result = Simulator().run(cfg, W)
    c = result.metadata["converged_at"]
    if c is not None:
        tail = result.activity[:, c:]
        assert np.all(tail == tail[:, :1])


# =========================================================================== #
# New tests: config validation
# =========================================================================== #

def test_convergence_eps_invalid_raises():
    cfg = SimulationConfig(convergence_eps=-1.0)
    with pytest.raises(ValueError, match="convergence_eps"):
        cfg.validate()


def test_invalid_integration_method_in_config_raises():
    cfg = SimulationConfig(integration_method="bogus")
    with pytest.raises(ValueError, match="integration_method"):
        cfg.validate()


@pytest.mark.parametrize("method", VALID_INTEGRATION_METHODS)
def test_valid_integration_methods_pass_validation(method):
    cfg = SimulationConfig(integration_method=method)
    cfg.validate()  # should not raise


if __name__ == "__main__":
    test_names = [name for name in globals() if name.startswith("test_")]
    pytest.main(["-v", "-k", " or ".join(test_names)])
