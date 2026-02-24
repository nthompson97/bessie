import logging
import numpy

from bessie.strategies import Strategy

from ._models import BacktestInputData, BacktestResults, BatterySpec

# FCAS market configuration, aligned to action/realised indices 1-6:
#   [RAISE6SEC, RAISE60SEC, RAISE5MIN, LOWER6SEC, LOWER60SEC, LOWER5MIN]

# Maximum response durations (hours) — BESS assumed to operate for full duration if called
DURATIONS = numpy.array(
    [
        5 / 60,  # Charge
        5 / 60,  # Discharge
        6 / 3600,  # RAISE6SEC
        60 / 3600,  # RAISE60SEC
        5 / 60,  # RAISE5MIN
        6 / 3600,  # LOWER6SEC
        60 / 3600,  # LOWER60SEC
        5 / 60,  # LOWER5MIN
    ]
)

# Probability of each FCAS market being called in a given dispatch interval
EVENT_PROBS = numpy.array(
    [
        1.0,  # Charge
        1.0,  # Discharge
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
            according to the probabilities defined in EVENT_PROBS.
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

        if c_soc / c_max < 0.1 or c_soc / c_max > 0.9:
            # TODO: Ideally we wouldn't do this. For now it simplifies the
            # backtest implementation.
            logging.warning(
                f"Index {i}: SOC at {c_soc:.2f} MWh ({c_soc / c_max:.1%} of "
                f"capacity). Disable FCAS markets to avoid infeasible dispatches."
            )
            action[2:8] = 0.0

        # The indicator array indicates whether energy is flowing into (+1) or
        # out of (-1) the battery for each action type.
        indicator = numpy.array(
            [+1.0, -1.0, -1.0, -1.0, -1.0, +1.0, +1.0, +1.0]
        )

        # Efficiencies indicate the round-trip efficiency for each action type.
        efficiencies = numpy.array(
            [
                eta_chg,
                eta_dchg,
                eta_dchg,
                eta_dchg,
                eta_dchg,
                eta_chg,
                eta_chg,
                eta_chg,
            ]
        )

        # Randomly determine which FCAS markets are called
        # TODO: In theory FCAS could be called more than once, fix this
        events = numpy.random.rand(8) <= EVENT_PROBS

        # shape (8,), MW actually absorbed/delivered
        p_actual = (
            action * p_max * indicator * efficiencies * events * DURATIONS
        )

        if c_soc + p_actual.sum() > c_max or c_soc + p_actual.sum() < 0:
            logging.warning(
                f"Strategy {strategy.name} produced action {action} at index "
                f"{i} that would result in infeasible SOC. Skipping."
            )

            output_actions[i, :] = 0.0
            output_c_soc[i] = c_soc
            output_c_max[i] = c_max
            output_revenue[i, :] = 0.0

        else:
            c_soc += p_actual.sum()
            c_max *= 1 - deg

            output_actions[i, :] = action
            output_c_soc[i] = c_soc
            output_c_max[i] = c_max

            output_revenue[i, 0] = -p_actual[0] * data.realised[i, 0]  # Charge revenue
            output_revenue[i, 1] = -p_actual[1] * data.realised[i, 0]  # Discharge revenue
            output_revenue[i, 2:] = action[2:] * p_max * dt * data.realised[i, 1:]  # FCAS revenue

    return BacktestResults(
        strategy=strategy,
        actions=output_actions,
        c_soc=output_c_soc,
        c_max=output_c_max,
        revenue=output_revenue,
    )
