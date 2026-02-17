from bessie.strategies import Strategy
from ._models import BacktestInputData, BacktestResults
from ._backtest import bess_backtest
from ._backtest_njit import bess_backtest_njit

def run_backtest(
    data: BacktestInputData,
    strategy: Strategy,
) -> BacktestResults: ...
