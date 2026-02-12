import abc
import enum
from typing import Callable


class Action(enum.Enum):
    Charge = 1
    Maintain = 0
    Discharge = -1


class Strategy(abc.ABC):
    def __init__(self) -> None: ...

    @abc.abstractmethod
    def action(self) -> Action:
        """
        Based on provided price and forecast data, produce an action for the
        strategy.
        """
        ...

    @abc.abstractmethod
    def action_njit(self) -> Callable[..., Action]:
        """
        Based on provided price and forecast data, produce an action for the
        strategy.

        This method must return an njit wrapped function that can be used in
        an njit context.
        """
        ...
