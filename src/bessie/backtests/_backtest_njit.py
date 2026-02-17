import logging
import numpy

from bessie.strategies import Strategy

from ._models import BacktestInputData, BacktestResults


def bess_backtest_njit(
    data: BacktestInputData,
    strategy: Strategy,
) -> BacktestResults:
    ...