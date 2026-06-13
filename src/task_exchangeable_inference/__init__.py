"""task_exchangeable_inference — Task-exchangeability inference with synthetic data."""

__version__ = "0.1.0"

from task_exchangeable_inference.exchangeability import ExchangeabilityModel
from task_exchangeable_inference.inference import ExchangeableInference
from task_exchangeable_inference.kernel import ExchangeabilityKernel
from task_exchangeable_inference.repository import TaskRepository

__all__ = [
    "ExchangeabilityModel",
    "ExchangeableInference",
    "ExchangeabilityKernel",
    "TaskRepository",
]
