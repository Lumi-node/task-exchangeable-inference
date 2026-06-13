"""exchangelib — Task-exchangeability inference with synthetic data."""

__version__ = "0.1.0"

from exchangelib.exchangeability import ExchangeabilityModel
from exchangelib.inference import ExchangeableInference
from exchangelib.kernel import ExchangeabilityKernel
from exchangelib.repository import TaskRepository

__all__ = [
    "ExchangeabilityModel",
    "ExchangeableInference",
    "ExchangeabilityKernel",
    "TaskRepository",
]
