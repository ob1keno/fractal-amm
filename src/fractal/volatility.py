# src/fractal/volatility.py
import numpy as np
from typing import List

class VolatilityAwareFractal:
    """Фрактал, адаптирующийся к волатильности."""
    
    def __init__(self, volatility: float = 0.03):
        self.volatility = volatility
        
    def get_optimal_depth(self) -> int:
        """Определяет оптимальную глубину фрактала по волатильности."""
        if self.volatility > 0.05:  # Высокая волатильность
            return 2  # Мелкая рекурсия, крупные куски
        elif self.volatility > 0.02:  # Средняя
            return 4
        else:  # Низкая
            return 6  # Глубокая рекурсия, мелкие куски
    
    def get_fragment_size_distribution(self) -> List[float]:
        """Возвращает распределение размеров фрагментов."""
        depth = self.get_optimal_depth()
        sizes = []
        
        for i in range(depth):
            # Фрагменты уменьшаются по мере увеличения глубины
            size = 1.0 / (2 ** i)
            sizes.append(size)
        
        # Нормализуем
        total = sum(sizes)
        return [s/total * 100 for s in sizes]  # В процентах
    
    def simulate_slippage(self, order_size: float) -> float:
        """Симуляция проскальзывания."""
        base_slippage = order_size * 0.5
        
        # Адаптивное уменьшение при высокой волатильности
        if self.volatility > 0.05:
            return base_slippage * 0.5
        elif self.volatility > 0.02:
            return base_slippage * 0.7
        else:
            return base_slippage * 0.9
    
    def calculate_efficiency(self, risk_tolerance: float) -> float:
        """Расчет эффективности."""
        depth = self.get_optimal_depth()
        
        # Эффективность зависит от глубины и волатильности
        efficiency = (depth / 6) * (1 - self.volatility * 10)
        
        # Корректировка по risk tolerance
        efficiency *= (1 - risk_tolerance * 0.3)
        
        return max(0.1, min(1.0, efficiency))