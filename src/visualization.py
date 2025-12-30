import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import matplotlib.cm as cm


def plot_liquidity_curve(analysis_results: Dict, title: str = "Кривая ликвидности"):
    """
    Построение кривой ликвидности.
    
    Args:
        analysis_results: Результаты анализа из analyze_trade_range
        title: Заголовок графика
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    amounts = analysis_results['amounts']
    prices = analysis_results['prices']
    
    # 1. Кривая цены
    ax1 = axes[0, 0]
    ax1.plot(amounts, prices, 'b-', linewidth=2)
    ax1.set_xlabel('Размер ордера', fontsize=11)
    ax1.set_ylabel('Средняя цена исполнения', fontsize=11)
    ax1.set_title(title, fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. Проскальзывание
    ax2 = axes[0, 1]
    if 'slippages' in analysis_results:
        ax2.plot(amounts, analysis_results['slippages'], 'r-', linewidth=2)
        ax2.set_xlabel('Размер ордера', fontsize=11)
        ax2.set_ylabel('Проскальзывание (%)', fontsize=11)
        ax2.set_title('Проскальзывание', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 3. Использование слоев
    ax3 = axes[1, 0]
    if 'layer_utilization' in analysis_results:
        layer_utilization = analysis_results['layer_utilization']
        colors = cm.viridis(np.linspace(0, 1, len(layer_utilization)))
        
        for idx, (layer_name, utilization) in enumerate(layer_utilization.items()):
            ax3.plot(amounts, utilization, label=layer_name, 
                    color=colors[idx], linewidth=1.5)
        
        ax3.set_xlabel('Размер ордера', fontsize=11)
        ax3.set_ylabel('Использование слоя (%)', fontsize=11)
        ax3.set_title('Распределение по слоям', fontsize=12)
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 105])
    
    # 4. Цена в обратном формате
    ax4 = axes[1, 1]
    if analysis_results['trade_direction'] == 'x_to_y':
        eth_prices = [1/p if p > 0 else 0 for p in prices]
        ax4.plot(amounts, eth_prices, 'g-', linewidth=2)
        ax4.set_ylabel('Цена за 1 ETH', fontsize=11)
    else:
        ax4.plot(amounts, prices, 'g-', linewidth=2)
        ax4.set_ylabel('Цена за 1 X', fontsize=11)
    
    ax4.set_xlabel('Размер ордера', fontsize=11)
    ax4.set_title('Альтернативное представление цены', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_comparison(analyses: Dict[str, Dict], title: str = "Сравнение AMM"):
    """
    Сравнение нескольких AMM на одном графике.
    
    Args:
        analyses: Словарь {имя: результаты_анализа}
        title: Заголовок графика
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(analyses)))
    
    for idx, (name, analysis) in enumerate(analyses.items()):
        ax.plot(analysis['amounts'], analysis['prices'], 
                label=name, color=colors[idx], linewidth=2)
    
    ax.set_xlabel('Размер ордера', fontsize=12)
    ax.set_ylabel('Средняя цена исполнения', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_layer_distribution(layers: List, title: str = "Распределение ликвидности по слоям"):
    """
    Визуализация распределения ликвидности по слоям.
    
    Args:
        layers: Список слоев
        title: Заголовок графика
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    layer_names = [layer.name for layer in layers]
    x_reserves = [layer.x_reserves for layer in layers]
    y_reserves = [layer.y_reserves for layer in layers]
    fees = [layer.fee * 100 for layer in layers]  # В процентах
    
    # 1. Резервы X
    axes[0].bar(layer_names, x_reserves, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Слой', fontsize=11)
    axes[0].set_ylabel('Резервы X', fontsize=11)
    axes[0].set_title('Распределение X по слоям', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. Резервы Y
    axes[1].bar(layer_names, y_reserves, color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('Слой', fontsize=11)
    axes[1].set_ylabel('Резервы Y', fontsize=11)
    axes[1].set_title('Распределение Y по слоям', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 3. Комиссии
    axes[2].bar(layer_names, fees, color='salmon', edgecolor='black')
    axes[2].set_xlabel('Слой', fontsize=11)
    axes[2].set_ylabel('Комиссия (%)', fontsize=11)
    axes[2].set_title('Комиссии по слоям', fontsize=12)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig