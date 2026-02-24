import logging
import numpy

from bessie.strategies import Strategy

from ._models import BacktestInputData, BacktestResults, BatterySpec

# FCAS market configuration, aligned to action/realised indices 1-6:
#   [RAISE6SEC, RAISE60SEC, RAISE5MIN, LOWER6SEC, LOWER60SEC, LOWER5MIN]

# Maximum response durations (hours) — BESS assumed to operate for full duration if called
FCAS_DURATIONS = numpy.array(
    [
        6 / 3600,  # RAISE6SEC
        60 / 3600,  # RAISE60SEC
        5 / 60,  # RAISE5MIN
        6 / 3600,  # LOWER6SEC
        60 / 3600,  # LOWER60SEC
        5 / 60,  # LOWER5MIN
    ]
)

# Probability of each FCAS market being called in a given dispatch interval
FCAS_PROBS = numpy.array(
    [
        0.05,  # RAISE6SEC
        0.05,  # RAISE60SEC
        0.05,  # RAISE5MIN
        0.05,  # LOWER6SEC
        0.05,  # LOWER60SEC
        0.05,  # LOWER5MIN
    ]
)


def bess_backtest(
    data: BacktestInputData,
    battery: BatterySpec,
    strategy: Strategy,
) -> BacktestResults:
    """
    Backtesting framework for a particular Strategy with the provided input
    data. If the strategy implements NJITStrategy, the njit-compiled backtest
    loop is used automatically.

    Args,
        data: Input data for the backtest, usually consisting of forecasts,
            realised prices, and battery configuration.
        battery: The specification of the battery to be used in the backtest.
        strategy: The strategy object defining based on input data what action
            to take on each timestep.

    Returns,
        A BacktestResults object containing  data for each timestep in the
            backtest period.

    TODO: Implement a mechanism for implying when frequency response was 
            required. I am hoping this is something we can source from
            historical data, i.e. a timeseries of counts for the number
            of times each FCAS market was called in each dispatch interval.
    NOTE: For now, it will be assume FCAS services are called randomly
            according to the probabilities defined in FCAS_PROBS.
    """
    logging.info(f"Running BESS backtest for strategy {strategy.name}")

    c_soc = 0.0  # Current SOC (MWh)
    c_max = float(battery.e_max)  # Current max battery capacity (MWh)
    p_max = float(battery.p_max)  # Max charge/discharge power rating (MW)
    eta_chg = float(battery.eta_chg)  # Charging efficiency
    eta_dchg = float(battery.eta_dchg)  # Discharging efficiency
    deg = float(battery.deg)  # Battery degradation rate (0 for now)
    dt = float(data.dt)  # Time step duration (hours)

    (n, _) = data.realised.shape

    output_actions = numpy.empty((n, 8))
    output_c_soc = numpy.empty(n)
    output_c_max = numpy.empty(n)
    output_revenue = numpy.empty((n, 8))

    for i in range(n):
        action = strategy.action(
            forecast=data.forecast[i, :, :],
            c_soc=c_soc,
            c_max=c_max,
            p_max=p_max,
            eta_chg=eta_chg,
            eta_dchg=eta_dchg,
            last_price=data.realised[i - 1, :],
        )

        if (action < 0).any() or (action > 1).any():
            logging.warning(
                f"Strategy {strategy.name} produced action {action} at index "
                f"{i} with values outside [0, 1]. Clipping to range."
            )
            action = numpy.clip(action, 0.0, 1.0)

        if action.sum() > 1.0:
            logging.warning(
                f"Strategy {strategy.name} produced action {action} at index "
                f"{i} with sum {action.sum()} > 1. Clipping to sum of 1."
            )
            action = action / action.sum()

        logging.debug(i, c_soc, c_max, p_max, data.realised[i - 1], action)

        # --- Energy market (action[0]=charge, action[1]=discharge) — bespoke handling ---
        # Both values must be non-negative fractions of p_max.
        a_charge = float(numpy.clip(action[0], 0.0, 1.0))
        a_discharge = float(numpy.clip(action[1], 0.0, 1.0))

        if a_charge > 0 or a_discharge > 0:
            c_max *= 1 - deg

        p_charge = min(a_charge * p_max * dt, c_max - c_soc)
        p_discharge = min(a_discharge * p_max * dt, c_soc)
        p_actual = p_charge * eta_chg - p_discharge * eta_dchg

        c_soc += p_actual

        # --- FCAS markets (action[2:8]) — vectorised ---
        # Actions are fractions [0, 1] of p_max offered as availability into each market.
        # Raise (indices 2-4): BESS discharges when called — needs SoC headroom downward.
        # Lower (indices 5-7): BESS charges when called — needs SoC headroom upward.
        fcas_mw = (
            numpy.clip(action[2:8], 0.0, 1.0) * p_max
        )  # shape (6,), MW offered

        # MWh that would be consumed/absorbed if the market is called at full power
        energy_required = fcas_mw * FCAS_DURATIONS  # shape (6,)

        # Zero out raise markets where SoC is insufficient to respond
        raise_feasible = energy_required[:3] <= c_soc
        fcas_mw[:3] *= raise_feasible

        # Zero out lower markets where headroom is insufficient to respond
        lower_feasible = energy_required[3:] <= (c_max - c_soc)
        fcas_mw[3:] *= lower_feasible

        # FCAS availability revenue: paid per MW enabled per hour, regardless of dispatch
        fcas_revenue = fcas_mw * data.realised[i, 1:7] * dt  # shape (6,)

        # Expected SoC change from probabilistic FCAS dispatch
        # Raise dispatches discharge (SoC down), lower dispatches charge (SoC up)
        expected_soc_delta = numpy.empty(6)
        expected_soc_delta[:3] = (
            -FCAS_PROBS[:3] * fcas_mw[:3] * FCAS_DURATIONS[:3]
        )
        expected_soc_delta[3:] = (
            FCAS_PROBS[3:] * fcas_mw[3:] * FCAS_DURATIONS[3:]
        )
        c_soc = numpy.clip(c_soc + expected_soc_delta.sum(), 0.0, c_max)

        output_actions[i, 0] = p_charge
        output_actions[i, 1] = p_discharge
        output_actions[i, 2:8] = fcas_mw
        output_c_soc[i] = c_soc
        output_c_max[i] = c_max
        output_revenue[i, 0] = -p_charge * data.realised[i, 0]
        output_revenue[i, 1] = p_discharge * data.realised[i, 0]
        output_revenue[i, 2:8] = fcas_revenue

    return BacktestResults(
        strategy=strategy,
        actions=output_actions,
        c_soc=output_c_soc,
        c_max=output_c_max,
        revenue=output_revenue,
    )
