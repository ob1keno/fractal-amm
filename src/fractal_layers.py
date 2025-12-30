import numpy as np
from typing import List, Tuple
from .core import FractalLayer


def generate_geometric_fractal(
    base_x: float,
    base_y: float,
    base_fee: float,
    num_layers: int,
    scale_factor: float = 2.0,
    fee_growth: float = 1.5,
    name_prefix: str = "Layer"
) -> List[FractalLayer]:
    """
    Генерация фрактальных слоев по геометрической прогрессии.
    
    Args:
        base_x: Базовое количество X для первого слоя
        base_y: Базовое количество Y для первого слоя
        base_fee: Базовая комиссия для первого слоя
        num_layers: Количество слоев
        scale_factor: Коэффициент масштабирования ликвидности
        fee_growth: Коэффициент роста комиссии
        name_prefix: Префикс для имен слоев
        
    Returns:
        List[FractalLayer]: Список сгенерированных слоев
    """
    layers = []
    
    for i in range(num_layers):
        layer_x = base_x * (scale_factor ** i)
        layer_y = base_y * (scale_factor ** i)
        layer_fee = base_fee * (fee_growth ** i)
        
        # Ограничиваем максимальную комиссию 5%
        layer_fee = min(layer_fee, 0.05)
        
        layers.append(
            FractalLayer(
                name=f"{name_prefix}_{i}",
                x_reserves=layer_x,
                y_reserves=layer_y,
                fee=layer_fee,
                priority=i
            )
        )
    
    return layers


def generate_power_law_fractal(
    total_x: float,
    total_y: float,
    num_layers: int,
    alpha: float = 1.5,
    min_fee: float = 0.0005,
    max_fee: float = 0.003,
    name_prefix: str = "PowerLayer"
) -> List[FractalLayer]:
    """
    Генерация фрактальных слоев по степенному закону.
    
    Args:
        total_x: Общее количество X для распределения
        total_y: Общее количество Y для распределения
        num_layers: Количество слоев
        alpha: Параметр степенного распределения
        min_fee: Минимальная комиссия
        max_fee: Максимальная комиссия
        name_prefix: Префикс для имен слоев
        
    Returns:
        List[FractalLayer]: Список сгенерированных слоев
    """
    layers = []
    
    # Создаем распределение по степенному закону
    indices = np.arange(1, num_layers + 1)
    weights = indices ** (-alpha)
    weights = weights / weights.sum()
    
    x_distribution = total_x * weights
    y_distribution = total_y * weights
    
    # Комиссия убывает для более глубоких слоев
    fee_range = np.linspace(max_fee, min_fee, num_layers)
    
    for i in range(num_layers):
        layers.append(
            FractalLayer(
                name=f"{name_prefix}_{i}",
                x_reserves=x_distribution[i],
                y_reserves=y_distribution[i],
                fee=fee_range[i],
                priority=i
            )
        )
    
    return layers


def generate_uniform_fractal(
    total_x: float,
    total_y: float,
    num_layers: int,
    base_fee: float = 0.001,
    name_prefix: str = "UniformLayer"
) -> List[FractalLayer]:
    """
    Генерация фрактальных слоев с равномерным распределением.
    
    Args:
        total_x: Общее количество X для распределения
        total_y: Общее количество Y для распределения
        num_layers: Количество слоев
        base_fee: Базовая комиссия
        name_prefix: Префикс для имен слоев
        
    Returns:
        List[FractalLayer]: Список сгенерированных слоев
    """
    layers = []
    
    x_per_layer = total_x / num_layers
    y_per_layer = total_y / num_layers
    
    for i in range(num_layers):
        layers.append(
            FractalLayer(
                name=f"{name_prefix}_{i}",
                x_reserves=x_per_layer,
                y_reserves=y_per_layer,
                fee=base_fee,
                priority=i
            )
        )
    
    return layers