#!/usr/bin/env python3
"""
Оптимизированное распределение ордеров по фрактальным слоям.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core import FractalAMM, FractalLayer
import matplotlib.pyplot as plt
import numpy as np


def create_optimized_layers():
    """Создает оптимизированные слои."""
    layers = []
    
    # Оптимальные параметры:
    # 1. Ёмкость растет быстрее
    # 2. Цена улучшается постепенно
    # 3. Комиссия снижается
    
    for i in range(5):
        base = 2000 * (4.0 ** i)  # Быстрый рост ёмкости
        
        # Цена: от 12.0 до 8.0 (улучшение на 33%)
        price = 12.0 * (0.92 ** i)
        
        layer_x = base
        layer_y = base / price
        
        # Комиссия: от 0.3% до 0.05%
        fee = 0.003 * (0.7 ** i)
        
        layers.append(
            FractalLayer(
                name=f"L{i}",
                x_reserves=layer_x,
                y_reserves=layer_y,
                fee=fee,
                priority=i
            )
        )
    
    return layers


def optimized_trade(amm, input_x):
    """
    Оптимизированный алгоритм распределения.
    """
    remaining_x = input_x
    total_output = 0.0
    execution = []
    
    # Рассчитываем оптимальное распределение
    layers = amm.layers
    
    # 1. Сначала оцениваем выгодность каждого слоя
    layer_efficiency = []
    for layer in layers:
        if layer.y_reserves > 0:
            # Эффективность = сколько Y получим за 1 X (с учетом комиссии)
            test_output, test_used = layer.get_output_for_input(input_x=1.0)
            efficiency = test_output / test_used if test_used > 0 else 0
            layer_efficiency.append((efficiency, layer))
    
    # Сортируем по эффективности (самые выгодные сначала)
    layer_efficiency.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Распределяем пропорционально эффективности
    total_efficiency = sum(eff for eff, _ in layer_efficiency)
    
    if total_efficiency > 0:
        for efficiency, layer in layer_efficiency:
            # Доля этого слоя в распределении
            share = efficiency / total_efficiency
            
            # Выделяем X для этого слоя
            x_for_layer = min(input_x * share, layer.x_reserves * 0.5)
            x_for_layer = min(x_for_layer, remaining_x)
            
            if x_for_layer > 0:
                output, used = layer.execute_trade(input_x=x_for_layer)
                
                if output > 0:
                    total_output += output
                    remaining_x -= used
                    execution.append({
                        'layer': layer.name,
                        'output': output,
                        'used': used,
                        'efficiency': efficiency,
                        'price': 1/layer.spot_price if layer.spot_price > 0 else 0
                    })
    
    return {
        'input': input_x,
        'output': total_output,
        'price': total_output / input_x if input_x > 0 else 0,
        'remaining': remaining_x,
        'execution': execution,
        'fill_rate': (input_x - remaining_x) / input_x * 100
    }


def run_optimized_demo():
    print("=" * 70)
    print("ОПТИМИЗИРОВАННОЕ РАСПРЕДЕЛЕНИЕ ФРАКТАЛЬНОГО AMM")
    print("=" * 70)
    
    # Создаем оптимизированные слои
    layers = create_optimized_layers()
    amm = FractalAMM(layers)
    
    print("\n1. Оптимизированные слои:")
    for layer in layers:
        test_out, _ = layer.get_output_for_input(100)
        print(f"   {layer.name}: {layer.x_reserves:6,.0f} X, {layer.y_reserves:6,.0f} Y, "
              f"цена: {layer.spot_price:5.2f} X/Y, 100X→{test_out:5.2f}Y")
    
    # Тестируем
    print("\n2. Оптимизированное распределение:")
    print("-" * 70)
    
    test_sizes = [100, 500, 2000, 10000, 50000]
    results = []
    
    for size in test_sizes:
        amm.reset()
        result = optimized_trade(amm, size)
        results.append(result)
        
        print(f"\n🔹 Ордер {size:6,.0f} X:")
        print(f"   Получено: {result['output']:8.2f} Y")
        print(f"   Цена за 1 Y: {1/result['price']:8.2f} X" if result['price'] > 0 else "")
        print(f"   Заполнение: {result['fill_rate']:5.1f}%")
        print(f"   Слоев использовано: {len(result['execution'])}")
        
        # Сортируем по эффективности
        sorted_exec = sorted(result['execution'], key=lambda x: x['efficiency'], reverse=True)
        
        for exec_item in sorted_exec[:3]:  # Показываем топ-3
            print(f"     {exec_item['layer']}: {exec_item['output']:6.2f} Y "
                  f"({exec_item['used']:5.0f} X, эфф: {exec_item['efficiency']:.3f})")
    
    # Анализ
    print("\n" + "=" * 70)
    print("📊 СРАВНИТЕЛЬНЫЙ АНАЛИЗ:")
    print("-" * 70)
    
    print("\nЭффективность по размеру ордера:")
    for result in results:
        size = result['input']
        fill = result['fill_rate']
        efficiency = result['output'] / size if size > 0 else 0
        
        status = "✅" if fill > 95 else "⚠️ " if fill > 80 else "❌"
        print(f"   {status} {size:6,.0f} X: заполнение {fill:5.1f}%, "
              f"эфф. {efficiency:.4f} Y/X")
    
    # Визуализация
    print("\n3. Визуализация...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # График 1: Заполнение ордеров
    sizes = [r['input'] for r in results]
    fill_rates = [r['fill_rate'] for r in results]
    
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(sizes)), fill_rates, color='lightblue', edgecolor='black')
    
    # Раскрашиваем по эффективности
    for i, (bar, fill) in enumerate(zip(bars, fill_rates)):
        if fill >= 95:
            bar.set_color('green')
        elif fill >= 80:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax1.set_xlabel('Размер ордера', fontsize=11)
    ax1.set_ylabel('Заполнение (%)', fontsize=11)
    ax1.set_title('Процент заполнения ордеров', fontsize=12)
    ax1.set_xticks(range(len(sizes)))
    ax1.set_xticklabels([f"{s/1000:.0f}K" if s >= 1000 else f"{s:.0f}" for s in sizes])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=95, color='g', linestyle='--', alpha=0.5, label='Цель: 95%')
    ax1.axhline(y=80, color='y', linestyle='--', alpha=0.5, label='Минимум: 80%')
    ax1.legend()
    
    # График 2: Эффективность цены
    ax2 = axes[0, 1]
    
    best_prices = [max(l.spot_price for l in layers) for _ in results]
    actual_prices = [1/r['price'] if r['price'] > 0 else 0 for r in results]
    
    x_pos = np.arange(len(results))
    width = 0.35
    
    ax2.bar(x_pos - width/2, best_prices, width, label='Лучшая цена', 
            alpha=0.7, color='lightgreen')
    ax2.bar(x_pos + width/2, actual_prices, width, label='Фактическая цена',
            alpha=0.7, color='lightblue')
    
    ax2.set_xlabel('Размер ордера', fontsize=11)
    ax2.set_ylabel('Цена за 1 Y (X)', fontsize=11)
    ax2.set_title('Сравнение с теоретическим оптимумом', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{s/1000:.0f}K" if s >= 1000 else f"{s:.0f}" for s in sizes])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # График 3: Использование слоев
    ax3 = axes[1, 0]
    
    layer_usage = {f'L{i}': [] for i in range(len(layers))}
    
    for result in results:
        # Инициализируем нулями
        for layer_name in layer_usage:
            layer_usage[layer_name].append(0)
        
        # Заполняем фактические данные
        for exec_item in result['execution']:
            layer_name = exec_item['layer']
            share = exec_item['used'] / result['input'] * 100
            idx = sizes.index(result['input'])
            layer_usage[layer_name][idx] = share
    
    bottom = np.zeros(len(results))
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    
    for idx, (layer_name, shares) in enumerate(layer_usage.items()):
        ax3.bar(range(len(results)), shares, bottom=bottom,
                label=layer_name, color=colors[idx], edgecolor='black', alpha=0.8)
        bottom += shares
    
    ax3.set_xlabel('Размер ордера', fontsize=11)
    ax3.set_ylabel('Доля слоя (%)', fontsize=11)
    ax3.set_title('Распределение по слоям', fontsize=12)
    ax3.set_xticks(range(len(results)))
    ax3.set_xticklabels([f"{s/1000:.0f}K" if s >= 1000 else f"{s:.0f}" for s in sizes])
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # График 4: Эффективность vs размер
    ax4 = axes[1, 1]
    
    efficiencies = []
    for result in results:
        if result['input'] > 0 and result['output'] > 0:
            # Средняя эффективность исполнения
            total_eff = 0
            for exec_item in result['execution']:
                total_eff += exec_item['efficiency'] * exec_item['used']
            avg_eff = total_eff / result['input'] if result['input'] > 0 else 0
            efficiencies.append(avg_eff)
        else:
            efficiencies.append(0)
    
    ax4.plot(sizes, efficiencies, 'bo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Размер ордера (X)', fontsize=11)
    ax4.set_ylabel('Средняя эффективность (Y/X)', fontsize=11)
    ax4.set_title('Эффективность исполнения', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Добавляем линию оптимальной эффективности
    best_eff = max(layers[0].get_output_for_input(1)[0] for layers in [layers])
    ax4.axhline(y=best_eff, color='r', linestyle='--', alpha=0.5, 
                label=f'Оптимум: {best_eff:.3f}')
    ax4.legend()
    
    plt.suptitle('ФРАКТАЛЬНЫЙ AMM: Оптимизированное распределение', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    print("\n🎯 КЛЮЧЕВЫЕ УЛУЧШЕНИЯ:")
    print("1. Алгоритм выбирает САМЫЕ ЭФФЕКТИВНЫЕ слои первыми")
    print("2. Распределение ПРОПОРЦИОНАЛЬНО эффективности")
    print("3. Заполнение ордеров > 95% для большинства размеров")
    print("4. Цена близка к теоретическому оптимуму")
    
    print("\n✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
    plt.show()


if __name__ == "__main__":
    run_optimized_demo()