from typing import List, Tuple
from .core import FractalLayer  # Измените эту строку


def generate_geometric_fractal(
    base_x: float,
    base_y: float,
    base_fee: float,
    num_layers: int,
    scale_factor: float = 2.0,
    fee_growth: float = 1.5,
    price_increase: float = 1.0,
    name_prefix: str = "Layer"
) -> List[FractalLayer]:
    """
    Генерация фрактальных слоев с разными ценами.
    
    Args:
        price_increase: Во сколько раз увеличивается цена в каждом следующем слое
    """
    layers = []
    
    for i in range(num_layers):
        # Ликвидность растет
        layer_x = base_x * (scale_factor ** i)
        
        # Цена меняется: в более глубоких слоях цена может быть лучше/хуже
        if price_increase > 1.0:
            # В глубоких слоях Y дороже (меньше Y за тот же X)
            layer_y = base_y * (scale_factor ** i) / (price_increase ** i)
        else:
            layer_y = base_y * (scale_factor ** i)
        
        layer_fee = base_fee * (fee_growth ** i)
        layer_fee = min(layer_fee, 0.05)  # Ограничение 5%
        
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