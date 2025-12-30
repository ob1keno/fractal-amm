#!/usr/bin/env python3
"""
Базовая реализация Order-Specific Chaos.
"""

import hashlib
import random
from typing import List, Tuple

class ChaoticOrder:
    """Хаотический ордер с seed-based неопределенностью."""
    
    def __init__(self, total_amount: float, duration_blocks: int, 
                 seed: str, sender: str = "0x" + "a"*40,
                 previous_blockhash: str = "0x" + "b"*64,
                 chaos_factor: float = 0.5):
        self.total_amount = total_amount
        self.duration_blocks = duration_blocks
        self.seed = seed
        self.sender = sender
        self.previous_blockhash = previous_blockhash
        self.chaos_factor = chaos_factor
        
        # Генерируем детерминированный seed
        self._seed_int = self._generate_seed_int()
        self.random = random.Random(self._seed_int)
        
    def _generate_seed_int(self) -> int:
        """Генерация целочисленного seed из входных параметров."""
        seed_string = f"{self.seed}{self.sender}{self.previous_blockhash}"
        seed_hash = hashlib.sha256(seed_string.encode()).hexdigest()
        return int(seed_hash[:16], 16)
    
    def get_execution_pattern(self) -> List[Tuple[int, float]]:
        """Генерация паттерна исполнения."""
        pattern = []
        remaining_blocks = self.duration_blocks
        remaining_amount = self.total_amount
        
        while remaining_blocks > 0 and remaining_amount > 0:
            # Размер блока зависит от chaos_factor
            max_block_size = max(1, int(self.duration_blocks * 0.1 * self.chaos_factor))
            block_size = self.random.randint(1, max_block_size)
            block_size = min(block_size, remaining_blocks)
            
            # Количество зависит от оставшегося
            max_amount = remaining_amount * 0.3
            min_amount = remaining_amount * 0.05
            amount = self.random.uniform(min_amount, max_amount)
            amount = min(amount, remaining_amount)
            
            pattern.append((block_size, amount))
            
            remaining_blocks -= block_size
            remaining_amount -= amount
        
        return pattern
    
    def get_timeline_array(self, duration: int = None) -> List[float]:
        """Преобразует паттерн в массив временной шкалы."""
        if duration is None:
            duration = self.duration_blocks
        
        pattern = self.get_execution_pattern()
        timeline = [0.0] * duration
        
        current_block = 0
        for block_size, amount in pattern:
            if current_block >= duration:
                break
                
            amount_per_block = amount / block_size
            for i in range(min(block_size, duration - current_block)):
                timeline[current_block + i] = amount_per_block
            
            current_block += block_size
        
        return timeline
    
    def calculate_predictability(self) -> float:
        """Расчет предсказуемости паттерна."""
        timeline = self.get_timeline_array()
        
        if len(timeline) == 0:
            return 1.0
        
        # Рассчитываем энтропию
        from scipy import stats
        
        # Нормализуем и добавляем маленькое значение для избежания log(0)
        normalized = np.array(timeline) + 1e-10
        normalized = normalized / normalized.sum()
        
        entropy = stats.entropy(normalized)
        max_entropy = np.log(len(timeline))
        
        # Предсказуемость обратно пропорциональна энтропии
        predictability = 1 - (entropy / max_entropy)
        
        return max(0.0, min(1.0, predictability))
    def get_execution_pattern(self) -> List[Tuple[int, float]]:
        """Генерация паттерна исполнения."""
        pattern = []
        remaining_blocks = self.duration_blocks
        remaining_amount = self.total_amount
        
        # Устанавливаем seed для воспроизводимости
        random_state = np.random.RandomState(self._seed_int % 10000)
        
        while remaining_blocks > 0 and remaining_amount > 0:
            max_block_size = max(1, int(self.duration_blocks * 0.1 * self.chaos_factor))
            block_size = random_state.randint(1, min(max_block_size, remaining_blocks))
            
            max_amount = remaining_amount * 0.3
            min_amount = remaining_amount * 0.05
            amount = random_state.uniform(min_amount, max_amount)
            amount = min(amount, remaining_amount)
            
            pattern.append((block_size, amount))
            remaining_blocks -= block_size
            remaining_amount -= amount
        
        return pattern