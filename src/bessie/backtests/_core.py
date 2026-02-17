from bessie.strategies import NJITStrategy, Strategy

from ._backtest import bess_backtest
from ._backtest_njit import bess_backtest_njit
from ._models import BacktestInputData, BacktestResults


def run_backtest(
    data: BacktestInputData,
    strategy: Strategy,
    use_njit: bool = True,
) -> BacktestResults:
    if use_njit and isinstance(strategy, NJITStrategy):
        return bess_backtest_njit(data, strategy)

    return bess_backtest(data, strategy)
