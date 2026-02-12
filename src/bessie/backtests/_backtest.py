from ._core import BacktestResults, BacktestInputData

from bessie.strategies import Strategy


def bess_backtest(
    data: BacktestInputData,
    strategy: Strategy,
) -> BacktestResults: ...
