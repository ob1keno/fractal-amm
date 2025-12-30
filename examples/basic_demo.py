#!/usr/bin/env python3
"""
Базовый пример использования фрактального AMM.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core import FractalAMM
from src.fractal_layers import generate_geometric_fractal
from src.visualization import plot_liquidity_curve, plot_layer_distribution
import matplotlib.pyplot as plt


def run_basic_demo():
    """Запуск базовой демонстрации."""
    print("=" * 60)
    print("БАЗОВАЯ ДЕМОНСТРАЦИЯ ФРАКТАЛЬНОГО AMM")
    print("=" * 60)
    
    # Генерируем фрактальные слои
    print("\n1. Генерация фрактальных слоев...")
    layers = generate_geometric_fractal(
        base_x=1000,
        base_y=100,
        base_fee=0.001,
        num_layers=5,
        scale_factor=2.0,
        fee_growth=1.5
    )
    
    print(f"   Создано слоев: {len(layers)}")
    for i, layer in enumerate(layers):
        print(f"   {layer.name}: {layer.x_reserves:.0f} X, {layer.y_reserves:.0f} Y, "
              f"комиссия: {layer.fee*100:.2f}%")
    
    # Создаем AMM
    amm = FractalAMM(layers)
    total_x, total_y = amm.total_reserves
    print(f"\n2. Создан фрактальный AMM:")
    print(f"   Общие резервы: {total_x:.0f} X, {total_y:.0f} Y")
    print(f"   Начальная цена: {layers[0].spot_price:.2f} X за 1 Y")
    
    # Тестовые ордера
    print("\n3. Тестирование ордеров разного размера:")
    test_orders = [100, 1000, 5000, 10000]
    
    for order_size in test_orders:
        result = amm.trade_x_for_y(order_size)
        amm.reset()  # Сбрасываем для следующего теста
        
        if result['success']:
            eth_price = 1 / result['effective_price'] if result['effective_price'] > 0 else 0
            print(f"\n   Ордер {order_size:6.0f} X:")
            print(f"     Получено: {result['output_y']:8.2f} Y")
            print(f"     Средняя цена: {result['effective_price']:.6f} Y за X")
            print(f"     Цена за 1 Y: {eth_price:8.2f} X")
            
            # Детали исполнения
            print("     Исполнение по слоям:")
            for detail in result['execution_details']:
                print(f"       {detail['layer']}: {detail['output_y']:6.2f} Y "
                      f"(комиссия: {detail['fee']*100:.2f}%)")
    
    # Анализ диапазона
    print("\n4. Анализ кривой ликвидности...")
    analysis = amm.analyze_trade_range(
        min_amount=10,
        max_amount=15000,
        steps=100,
        trade_direction='x_to_y'
    )
    
    # Визуализация
    print("\n5. Визуализация результатов...")
    
    # Распределение по слоям
    fig1 = plot_layer_distribution(layers, "Распределение ликвидности (5 слоев)")
    
    # Кривая ликвидности
    fig2 = plot_liquidity_curve(analysis, "Кривая ликвидности фрактального AMM")
    
    print("\n✅ Демонстрация завершена!")
    print("   Откройте графики для анализа результатов.")
    
    plt.show()


if __name__ == "__main__":
    run_basic_demo()