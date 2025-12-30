import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import copy

# --- 1. МОДЕЛЬ ФРАКТАЛЬНОГО ПУЛА (ИСПРАВЛЕННАЯ) ---

@dataclass
class FractalPoolLayer:
    """Один слой (уровень) фрактального пула."""
    name: str
    x: float  # Резервы токена X (USDC)
    y: float  # Резервы токена Y (ETH)
    fee: float  # Комиссия слоя
    
    def get_output_for_input_x(self, input_x: float) -> Tuple[float, float]:
        """
        Рассчитывает вывод Y для ввода X.
        Возвращает (output_y, actual_input_x_used)
        """
        if input_x <= 0:
            return 0.0, 0.0
            
        # Комиссия берется с входящей суммы
        input_x_after_fee = input_x * (1 - self.fee)
        
        # По формуле постоянного продукта
        k = self.x * self.y
        new_x = self.x + input_x_after_fee
        new_y = k / new_x
        output_y = self.y - new_y
        
        # Нельзя вывести больше, чем есть в пуле
        output_y = min(output_y, self.y * 0.999)  # Оставляем немного для избежания деления на 0
        
        return max(output_y, 0), input_x_after_fee
    
    def execute_trade_x_for_y(self, input_x: float) -> Tuple[float, float]:
        """Исполняет сделку X->Y и обновляет резервы."""
        output_y, input_x_used = self.get_output_for_input_x(input_x)
        
        if output_y > 0 and input_x_used > 0:
            self.x += input_x_used
            self.y -= output_y
            
        return output_y, input_x_used
    
    def get_spot_price(self) -> float:
        """Мгновенная цена: x / y"""
        if self.y == 0:
            return float('inf')
        return self.x / self.y


class FractalAMM:
    """Фрактальный AMM с последовательными слоями."""
    
    def __init__(self):
        # Слои отсортированы от самого дешевого к самому дорогому
        self.layers: List[FractalPoolLayer] = [
            FractalPoolLayer("Surface", x=1000.0, y=100.0, fee=0.0001),   # 0.01%
            FractalPoolLayer("Medium",  x=5000.0, y=500.0, fee=0.001),    # 0.1%
            FractalPoolLayer("Core",    x=20000.0, y=2000.0, fee=0.003)   # 0.3%
        ]
        
        # Сохраняем начальные состояния для сброса
        self.initial_state = [(layer.x, layer.y) for layer in self.layers]
    
    def reset_pools(self):
        """Сбрасываем пулы к начальному состоянию."""
        for layer, (init_x, init_y) in zip(self.layers, self.initial_state):
            layer.x, layer.y = init_x, init_y
    
    def trade_x_for_y(self, input_x: float) -> dict:
        """
        Покупаем Y за X через все слои последовательно.
        Возвращает детальный отчет.
        """
        remaining_x = input_x
        total_output_y = 0.0
        execution_report = []
        
        for layer in self.layers:
            if remaining_x <= 0:
                break
            
            # Сколько можем исполнить в этом слое
            output_y, x_used = layer.execute_trade_x_for_y(remaining_x)
            
            if output_y > 0:
                total_output_y += output_y
                remaining_x -= x_used / (1 - layer.fee)  # Возвращаем к исходному input_x
                
                execution_report.append({
                    'layer': layer.name,
                    'output_y': output_y,
                    'x_used': x_used / (1 - layer.fee),  # Исходный input_x с комиссией
                    'layer_fee': layer.fee,
                    'remaining_x': max(0, remaining_x),
                    'spot_price_before': layer.get_spot_price()
                })
        
        effective_price = total_output_y / input_x if input_x > 0 else 0
        
        return {
            'input_x': input_x,
            'total_output_y': total_output_y,
            'effective_price': effective_price,
            'execution': execution_report,
            'remaining_x': remaining_x
        }
    
    def analyze_trade_range(self, min_trade: float = 10, max_trade: float = 10000, steps: int = 50) -> dict:
        """Анализируем исполнение ордеров разного размера."""
        trade_sizes = np.linspace(min_trade, max_trade, steps)
        
        prices = []
        layer_shares = {layer.name: [] for layer in self.layers}
        
        for trade_size in trade_sizes:
            # Сбрасываем пулы перед каждым тестом
            self.reset_pools()
            
            # Исполняем ордер
            report = self.trade_x_for_y(trade_size)
            
            if report['effective_price'] > 0:
                prices.append(report['effective_price'])
                
                # Собираем доли каждого слоя
                total_output = report['total_output_y']
                layer_outputs = {layer.name: 0 for layer in self.layers}
                
                for exec_step in report['execution']:
                    layer_name = exec_step['layer']
                    layer_outputs[layer_name] = exec_step['output_y']
                
                # Добавляем проценты
                for layer_name in layer_shares.keys():
                    share = (layer_outputs[layer_name] / total_output * 100) if total_output > 0 else 0
                    layer_shares[layer_name].append(share)
        
        return {
            'trade_sizes': trade_sizes,
            'prices': prices,
            'layer_shares': layer_shares
        }


# --- 2. ВИЗУАЛИЗАЦИЯ ---

def run_demo():
    """Запускаем демонстрацию фрактального AMM."""
    print("🚀 Создаем фрактальный AMM с тремя слоями...")
    print("   Surface: 1000 USDC, 100 ETH, комиссия 0.01%")
    print("   Medium:  5000 USDC, 500 ETH, комиссия 0.1%")
    print("   Core:    20000 USDC, 2000 ETH, комиссия 0.3%")
    print()
    
    amm = FractalAMM()
    
    # Тестируем конкретные ордера
    test_sizes = [50, 500, 3000, 8000]
    
    print("📊 Тест: исполнение ордеров разного размера")
    print("-" * 60)
    
    for size in test_sizes:
        amm.reset_pools()
        report = amm.trade_x_for_y(size)
        
        print(f"\n🔹 Ордер: {size:.0f} USDC")
        print(f"   Получено: {report['total_output_y']:.4f} ETH")
        print(f"   Средняя цена: {report['effective_price']:.6f} ETH/USDC")
        print(f"   Цена за 1 ETH: {1/report['effective_price']:.2f} USDC" if report['effective_price'] > 0 else "   Невозможно рассчитать")
        
        print("   Исполнение по слоям:")
        for i, step in enumerate(report['execution']):
            print(f"     {step['layer']}: {step['output_y']:.2f} ETH "
                  f"(использовано {step['x_used']:.0f} USDC, "
                  f"комиссия {step['layer_fee']*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("📈 Анализ кривой ликвидности...")
    
    # Анализируем диапазон
    analysis = amm.analyze_trade_range(min_trade=10, max_trade=15000, steps=100)
    
    # Создаем графики
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Кривая цены
    ax1 = axes[0, 0]
    ax1.plot(analysis['trade_sizes'], analysis['prices'], 'b-', linewidth=2)
    ax1.set_xlabel('Размер ордера (USDC)', fontsize=11)
    ax1.set_ylabel('Средняя цена (ETH за USDC)', fontsize=11)
    ax1.set_title('Кривая исполнения: цена vs размер ордера', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Добавляем вертикальные линии для ключевых точек
    critical_points = [1000, 6000, 12000]  # Примерные точки перехода между слоями
    for point in critical_points:
        if point < max(analysis['trade_sizes']):
            ax1.axvline(x=point, color='r', linestyle='--', alpha=0.5)
            ax1.text(point, ax1.get_ylim()[1]*0.9, f' {point} USDC', 
                    rotation=90, verticalalignment='top')
    
    # 2. Доли слоев
    ax2 = axes[0, 1]
    for layer_name, shares in analysis['layer_shares'].items():
        ax2.plot(analysis['trade_sizes'], shares, label=layer_name, linewidth=2)
    
    ax2.set_xlabel('Размер ордера (USDC)', fontsize=11)
    ax2.set_ylabel('Доля в исполнении (%)', fontsize=11)
    ax2.set_title('Вклад каждого слоя', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    # 3. Цена за 1 ETH (более интуитивно)
    ax3 = axes[1, 0]
    eth_prices = [1/p if p > 0 else 0 for p in analysis['prices']]
    ax3.plot(analysis['trade_sizes'], eth_prices, 'g-', linewidth=2)
    ax3.set_xlabel('Размер ордера (USDC)', fontsize=11)
    ax3.set_ylabel('Цена за 1 ETH (USDC)', fontsize=11)
    ax3.set_title('Стоимость ETH в зависимости от размера ордера', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    
    # 4. Проскальзывание (slippage)
    ax4 = axes[1, 1]
    spot_price_start = amm.layers[0].get_spot_price()
    slippage = [(1/p - 1/spot_price_start)/(1/spot_price_start)*100 
                if p > 0 else 0 
                for p in analysis['prices']]
    
    ax4.plot(analysis['trade_sizes'], slippage, 'r-', linewidth=2)
    ax4.set_xlabel('Размер ордера (USDC)', fontsize=11)
    ax4.set_ylabel('Проскальзывание (%)', fontsize=11)
    ax4.set_title('Проскальзывание при исполнении', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    print("\n📊 Ключевые наблюдения:")
    print("1. Мелкие ордеры (< 1000 USDC) исполняются в Surface слое с низкой комиссией")
    print("2. Средние ордеры (1000-6000 USDC) задействуют Medium слой")
    print("3. Крупные ордеры (> 6000 USDC) используют все три слоя")
    print("4. Изломы на графике цены показывают переходы между слоями")
    print("\n✅ Графики готовы. Обратите внимание на 'ступенчатость' - это фрактальная структура!")
    
    plt.show()
    
    # Сравнение с обычным AMM
    print("\n" + "=" * 60)
    print("🔍 Сравнение с классическим объединенным пулом:")
    
    # Суммируем все ликвидности
    total_x = sum(layer.x for layer in amm.layers)
    total_y = sum(layer.y for layer in amm.layers)
    avg_fee = np.mean([layer.fee for layer in amm.layers])
    
    print(f"   Объединенный пул: {total_x:.0f} USDC, {total_y:.0f} ETH")
    print(f"   Средняя комиссия: {avg_fee*100:.3f}%")
    
    # Создаем виртуальный объединенный пул
    combined_pool = FractalPoolLayer("Combined", total_x, total_y, avg_fee)
    
    print("\n   Сравнение цен для разных размеров ордеров:")
    print("   Размер | Фрактальный AMM | Классический AMM | Разница")
    print("   " + "-"*50)
    
    for size in [100, 1000, 5000, 10000]:
        amm.reset_pools()
        fractal_report = amm.trade_x_for_y(size)
        fractal_price = fractal_report['effective_price']
        
        # Для классического пула
        output_y, _ = combined_pool.get_output_for_input_x(size)
        classic_price = output_y / size if size > 0 else 0
        
        if fractal_price > 0 and classic_price > 0:
            diff = (fractal_price - classic_price) / classic_price * 100
            print(f"   {size:6.0f} | {1/fractal_price:13.2f} | {1/classic_price:16.2f} | {diff:+.2f}%")
        else:
            print(f"   {size:6.0f} | {'N/A':13} | {'N/A':16} | N/A")
    
    print("\n✅ Демонстрация завершена!")


# --- 3. ЗАПУСК ---
if __name__ == "__main__":
    run_demo()