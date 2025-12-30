# src/fractal/base.py
from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseFractal(ABC):
    """Базовый класс для всех фрактальных алгоритмов."""
    
    @abstractmethod
    def generate_execution(self) -> List[Tuple[int, float]]:
        """Генерация паттерна исполнения."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> dict:
        """Возвращает метаданные алгоритма."""
        pass