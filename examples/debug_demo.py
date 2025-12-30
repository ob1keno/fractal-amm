#!/usr/bin/env python3
"""
Демонстрация с отладкой распределения ордеров по слоям.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core import FractalAMM
from src.fractal_layers import generate_geometric_fractal
import matplotlib.pyplot as plt


def debug_trade(amm, input_x):
    """Подробная отладка одного ордера."""
    print(f"\n{'='*60}")
    print(f"ОТЛАДКА: Ордер {input_x:.0f} X")
    print('='*60)
    
    # Показываем состояние до торговли
    print("Состояние слоев ДО торговли:")
    for i, layer in enumerate(amm.layers):
        print(f"  {layer.name}: {layer.x_reserves:.0f} X, {layer.y_reserves:.0f} Y, "
              f"цена: {layer.spot_price:.2f} X/Y")
    
    # Исполняем ордер
    result = amm.trade_x_for_y(input_x)
    
    print(f"\nРезультат:")
    print(f"  Получено: {result['output_y']:.2f} Y")
    print(f"  Средняя цена: {result['effective_price']:.6f} Y/X")
    print(f"  Цена за 1 Y: {1/result['effective_price']:.2f} X" if result['effective_price'] > 0 else "")
    
    print("\nИсполнение по слоям:")
    total_x_used = 0
    for detail in result['execution_details']:
        total_x_used += detail['x_used']
        print(f"  {detail['layer']}: {detail['output_y']:.2f} Y "
              f"({detail['x_used']:.0f} X, комиссия: {detail['fee']*100:.2f}%)")
    
    print(f"\nИспользовано X: {total_x_used:.0f} из {input_x:.0f}")
    print(f"Осталось X: {result['remaining_x']:.2f}")
    
    # Показываем состояние после торговли
    print("\nСостояние слоев ПОСЛЕ торговли:")
    for i, layer in enumerate(amm.layers):
        print(f"  {layer.name}: {layer.x_reserves:.0f} X, {layer.y_reserves:.0f} Y, "
              f"цена: {layer.spot_price:.2f} X/Y")
    
    return result


def run_debug_demo():
    """Запуск демонстрации с отладкой."""
    print("ДЕМОНСТРАЦИЯ РАСПРЕДЕЛЕНИЯ ОРДЕРОВ ПО СЛОЯМ")
    print("="*60)
    
    # Генерируем фрактальные слои с БОЛЬШЕЙ разницей в ликвидности
    print("\n1. Генерация фрактальных слоев с разной ёмкостью...")
    layers = []
for i in range(20):
    # Каждый следующий слой имеет ЛУЧШУЮ цену (больше Y за тот же X)
    # Это стимулирует использовать более глубокие слои
    base_multiplier = 3.0 ** i
    price_improvement = 0.9 ** i  # Каждый слой дает на 10% лучшую цену
    
    layer_x = 1000 * base_multiplier
    layer_y = 100 * base_multiplier / price_improvement  # Больше Y в глубоких слоях
    
    layers.append(
        FractalLayer(
            name=f"Layer_{i}",
            x_reserves=layer_x,
            y_reserves=layer_y,
            fee=0.001 * (1.3 ** i),
            priority=i
        )
    )
    
    print(f"   Создано {len(layers)} слоев:")
    for layer in layers:
        print(f"   {layer.name}: {layer.x_reserves:.0f} X, {layer.y_reserves:.0f} Y, "
              f"комиссия: {layer.fee*100:.2f}%, цена: {layer.spot_price:.2f} X/Y")
    
    # Создаем AMM
    amm = FractalAMM(layers)
    
    # Тестируем ордера разного размера
    test_orders = [100, 500, 2000, 8000, 20000]
    
    for order_size in test_orders:
        amm.reset()  # Сбрасываем состояние перед каждым тестом
        debug_trade(amm, order_size)
    
    # Анализ кривой ликвидности
    print(f"\n{'='*60}")
    print("АНАЛИЗ КРИВОЙ ЛИКВИДНОСТИ")
    print('='*60)
    
    amm.reset()
    analysis = amm.analyze_trade_range(
        min_amount=10,
        max_amount=30000,
        steps=100
    )
    
    # Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # График 1: Цена
    amounts = analysis['amounts']
    prices = analysis['prices']
    ax1.plot(amounts, [1/p if p > 0 else 0 for p in prices], 'b-', linewidth=2)
    ax1.set_xlabel('Размер ордера (X)', fontsize=11)
    ax1.set_ylabel('Цена за 1 Y (X)', fontsize=11)
    ax1.set_title('Кривая ликвидности: цена Y в X', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Добавляем вертикальные линии для ёмкости каждого слоя
    capacities = []
    for layer in layers:
        # Примерная ёмкость слоя (сколько X он может принять)
        capacity = layer.x_reserves * 0.8  # Берем 80% для примера
        capacities.append(capacity)
        ax1.axvline(x=capacity, color='r', linestyle='--', alpha=0.5)
        ax1.text(capacity, ax1.get_ylim()[1]*0.9, f' {layer.name}', 
                rotation=90, verticalalignment='top', fontsize=8)
    
    # График 2: Использование слоев
    if 'layer_utilization' in analysis:
        for layer_name, utilization in analysis['layer_utilization'].items():
            ax2.plot(amounts, utilization, label=layer_name, linewidth=1.5)
    
    ax2.set_xlabel('Размер ордера (X)', fontsize=11)
    ax2.set_ylabel('Использование слоя (%)', fontsize=11)
    ax2.set_title('Распределение ордеров по слоям', fontsize=12)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    plt.suptitle('ФРАКТАЛЬНЫЙ AMM: Распределение ликвидности', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    print("\n📊 Ключевые наблюдения:")
    print("1. Мелкие ордеры (< 1000 X) исполняются в Layer_0")
    print("2. Средние ордеры (1000-5000 X) задействуют Layer_0 и Layer_1")
    print("3. Крупные ордеры (> 10000 X) используют все слои")
    print("4. Каждый слой имеет свою 'ёмкость' (вертикальные линии)")
    
    print("\n✅ Демонстрация завершена! Открываю графики...")
    plt.show()


if __name__ == "__main__":
    run_debug_demo()