import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class FractalLayer:
    """
    Один слой фрактального AMM.
    
    Attributes:
        name: Идентификатор слоя
        x_reserves: Резервы токена X (базовый, например USDC)
        y_reserves: Резервы токена Y (котируемый, например ETH)
        fee: Комиссия слоя (от 0 до 1)
        priority: Приоритет исполнения (меньше = выше приоритет)
    """
    
    name: str
    x_reserves: float
    y_reserves: float
    fee: float
    priority: int = 0
    
    def __post_init__(self):
        """Валидация параметров после инициализации."""
        if self.fee < 0 or self.fee > 1:
            raise ValueError("Fee must be between 0 and 1")
        if self.x_reserves < 0 or self.y_reserves < 0:
            raise ValueError("Reserves cannot be negative")
    
    @property
    def invariant(self) -> float:
        """Константа произведения (x * y)."""
        return self.x_reserves * self.y_reserves
    
    @property
    def spot_price(self) -> float:
        """Мгновенная цена Y в терминах X."""
        if self.y_reserves == 0:
            return float('inf')
        return self.x_reserves / self.y_reserves
    
    def get_output_for_input(self, input_x: float = 0, input_y: float = 0) -> Tuple[float, float]:
        """
        Рассчитывает выходное количество при заданном входе.
        
        Args:
            input_x: Количество токена X для обмена
            input_y: Количество токена Y для обмена
            
        Returns:
            Tuple[output, input_used]: Выходное количество и использованный вход
        """
        if input_x > 0 and input_y > 0:
            raise ValueError("Only one-sided trades are supported")
        
        if input_x > 0:  # Покупаем Y за X
            input_after_fee = input_x * (1 - self.fee)
            new_x = self.x_reserves + input_after_fee
            new_y = self.invariant / new_x
            output_y = self.y_reserves - new_y
            
            # Не даем вывести больше, чем есть
            output_y = min(output_y, self.y_reserves * 0.9999)
            
            return max(output_y, 0), input_x
        
        elif input_y > 0:  # Покупаем X за Y
            input_after_fee = input_y * (1 - self.fee)
            new_y = self.y_reserves + input_after_fee
            new_x = self.invariant / new_y
            output_x = self.x_reserves - new_x
            
            output_x = min(output_x, self.x_reserves * 0.9999)
            
            return max(output_x, 0), input_y
        
        return 0.0, 0.0
    
    def execute_trade(self, input_x: float = 0, input_y: float = 0) -> Tuple[float, float]:
        """
        Исполняет сделку и обновляет резервы.
        
        Returns:
            Tuple[output, input_used]: То же, что и get_output_for_input
        """
        output, input_used = self.get_output_for_input(input_x, input_y)
        
        if output > 0:
            if input_x > 0:  # Покупаем Y за X
                input_after_fee = input_x * (1 - self.fee)
                self.x_reserves += input_after_fee
                self.y_reserves -= output
            elif input_y > 0:  # Покупаем X за Y
                input_after_fee = input_y * (1 - self.fee)
                self.y_reserves += input_after_fee
                self.x_reserves -= output
        
        return output, input_used


class FractalAMM:
    """
    Фрактальный AMM с многослойной структурой.
    
    Attributes:
        layers: Список слоев, отсортированных по приоритету
    """
    
    def __init__(self, layers: List[FractalLayer]):
        """
        Инициализация фрактального AMM.
        
        Args:
            layers: Список слоев AMM
        """
        self.layers = sorted(layers, key=lambda l: l.priority)
        self._initial_state = [(l.x_reserves, l.y_reserves) for l in self.layers]
    
    def reset(self) -> None:
        """Сброс всех слоев к исходному состоянию."""
        for layer, (init_x, init_y) in zip(self.layers, self._initial_state):
            layer.x_reserves, layer.y_reserves = init_x, init_y
    
    @property
    def total_reserves(self) -> Tuple[float, float]:
        """Суммарные резервы по всем слоям."""
        total_x = sum(l.x_reserves for l in self.layers)
        total_y = sum(l.y_reserves for l in self.layers)
        return total_x, total_y
    
    def trade_x_for_y(self, input_x: float) -> Dict:
        """
        Покупка токена Y за токен X.
        
        Args:
            input_x: Количество токена X для обмена
            
        Returns:
            Dict с результатами торговли
        """
        if input_x <= 0:
            raise ValueError("Input amount must be positive")
        
        remaining_x = input_x
        total_output_y = 0.0
        execution_details = []
        
        for layer in self.layers:
            if remaining_x <= 1e-12:  # Практически ноль
                break
            
            output_y, x_used = layer.execute_trade(input_x=remaining_x)
            
            if output_y > 0:
                total_output_y += output_y
                remaining_x -= x_used
                
                execution_details.append({
                    'layer': layer.name,
                    'output_y': output_y,
                    'x_used': x_used,
                    'fee': layer.fee,
                    'spot_price': layer.spot_price,
                    'remaining_reserves': (layer.x_reserves, layer.y_reserves)
                })
        
        effective_price = total_output_y / input_x if input_x > 0 else 0.0
        
        return {
            'input_x': input_x,
            'output_y': total_output_y,
            'effective_price': effective_price,
            'remaining_x': remaining_x,
            'execution_details': execution_details,
            'success': total_output_y > 0
        }
    
    def trade_y_for_x(self, input_y: float) -> Dict:
        """
        Покупка токена X за токен Y.
        
        Args:
            input_y: Количество токена Y для обмена
            
        Returns:
            Dict с результатами торговли
        """
        if input_y <= 0:
            raise ValueError("Input amount must be positive")
        
        remaining_y = input_y
        total_output_x = 0.0
        execution_details = []
        
        for layer in self.layers:
            if remaining_y <= 1e-12:
                break
            
            output_x, y_used = layer.execute_trade(input_y=remaining_y)
            
            if output_x > 0:
                total_output_x += output_x
                remaining_y -= y_used
                
                execution_details.append({
                    'layer': layer.name,
                    'output_x': output_x,
                    'y_used': y_used,
                    'fee': layer.fee,
                    'spot_price': layer.spot_price,
                    'remaining_reserves': (layer.x_reserves, layer.y_reserves)
                })
        
        effective_price = total_output_x / input_y if input_y > 0 else 0.0
        
        return {
            'input_y': input_y,
            'output_x': total_output_x,
            'effective_price': effective_price,
            'remaining_y': remaining_y,
            'execution_details': execution_details,
            'success': total_output_x > 0
        }
    
    def analyze_trade_range(self, 
                           min_amount: float = 10, 
                           max_amount: float = 10000, 
                           steps: int = 100,
                           trade_direction: str = 'x_to_y') -> Dict:
        """
        Анализ исполнения для диапазона сумм.
        
        Args:
            min_amount: Минимальная сумма торговли
            max_amount: Максимальная сумма торговли
            steps: Количество шагов
            trade_direction: Направление торговли ('x_to_y' или 'y_to_x')
            
        Returns:
            Dict с результатами анализа
        """
        amounts = np.linspace(min_amount, max_amount, steps)
        prices = []
        slippages = []
        layer_utilization = {layer.name: [] for layer in self.layers}
        
        for amount in amounts:
            self.reset()
            
            if trade_direction == 'x_to_y':
                result = self.trade_x_for_y(amount)
                output = result['output_y']
                spot_price_start = self.layers[0].spot_price if self.layers else 0
            else:
                result = self.trade_y_for_x(amount)
                output = result['output_x']
                spot_price_start = 1 / self.layers[0].spot_price if self.layers and self.layers[0].spot_price > 0 else 0
            
            effective_price = result['effective_price']
            
            if effective_price > 0:
                prices.append(effective_price)
                
                # Расчет проскальзывания
                if spot_price_start > 0:
                    if trade_direction == 'x_to_y':
                        slippage = (effective_price - 1/spot_price_start) / (1/spot_price_start) * 100
                    else:
                        slippage = (effective_price - spot_price_start) / spot_price_start * 100
                    slippages.append(slippage)
                else:
                    slippages.append(0)
                
                # Сбор использования слоев
                total_output = output
                layer_outputs = {layer.name: 0 for layer in self.layers}
                
                for detail in result['execution_details']:
                    layer_name = detail['layer']
                    if trade_direction == 'x_to_y':
                        layer_outputs[layer_name] = detail['output_y']
                    else:
                        layer_outputs[layer_name] = detail['output_x']
                
                for layer in self.layers:
                    share = (layer_outputs[layer.name] / total_output * 100) if total_output > 0 else 0
                    layer_utilization[layer.name].append(share)
            else:
                prices.append(0)
                slippages.append(0)
                for layer in self.layers:
                    layer_utilization[layer.name].append(0)
        
        return {
            'amounts': amounts,
            'prices': prices,
            'slippages': slippages,
            'layer_utilization': layer_utilization,
            'trade_direction': trade_direction
        }