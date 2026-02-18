from bessie.strategies import NJITStrategy, Strategy

from ._backtest import bess_backtest
from ._backtest_njit import bess_backtest_njit
from ._models import BacktestInputData, BacktestResults, BatterySpec


def run_backtest(
    data: BacktestInputData,
    battery: BatterySpec,
    strategy: Strategy,
    use_njit: bool = True,
) -> BacktestResults:
    if use_njit and isinstance(strategy, NJITStrategy):
        return bess_backtest_njit(
            data=data,
            battery=battery,
            strategy=strategy,
        )

    else:
        return bess_backtest(
            data=data,
            battery=battery,
            strategy=strategy,
        )
