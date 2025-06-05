import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.btc_futures_env import BTCFuturesEnv
import yaml
import os
import sys
from pyfolio import timeseries  # For advanced metrics

# Добавить корень проекта в PYTHONPATH на случай запуска из других мест
sys.path.append('.')

def main(model_path, data_path, config_path):
    # 1. Загрузка данных (локальный файл или URL)
    is_url = data_path.startswith('http://') or data_path.startswith('https://')
    try:
        if is_url:
            test_data = pd.read_csv(data_path)
            print(f"✅ Данные загружены с URL. Размер: {test_data.shape}")
        else:
            if not os.path.exists(data_path):
                print(f"❌ Файл данных не найден: {data_path}")
                return
            test_data = pd.read_csv(data_path)
            print(f"✅ Данные загружены из файла. Размер: {test_data.shape}")
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return

    # Приведение альтернативных названий столбцов к стандартным
    column_renames = {}
    if "Date" not in test_data.columns:
        if "Open time" in test_data.columns:
            column_renames["Open time"] = "Date"
    if column_renames:
        test_data = test_data.rename(columns=column_renames)

    # Привести все названия столбцов к нижнему регистру для совместимости с env
    test_data.columns = [c.lower() for c in test_data.columns]

    # Проверка качества данных
    required_columns = ["date", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_columns if col not in test_data.columns]
    if missing_cols:
        print(f"❌ В данных отсутствуют необходимые столбцы: {missing_cols}")
        return
    if test_data[required_columns].isnull().any().any():
        print("❌ В данных есть пропущенные значения!")
        return
    test_data = test_data.sort_values("date").reset_index(drop=True)
    print("✅ Данные прошли валидацию и отсортированы по дате.")

    # 2. Загрузка конфига
    if not os.path.exists(config_path):
        print(f"❌ Файл конфига не найден: {config_path}")
        return
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 3. Инициализация среды
    env = BTCFuturesEnv(config, test_data)

    # 4. Загрузка модели
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return
    model = PPO.load(model_path)
    print(f"✅ Модель загружена: {model_path}")

    # 5. Запуск оценки
    portfolio_history = []
    state, _ = env.reset()
    done = False
    steps = 0
    while not done:
        action, _ = model.predict(state)
        next_state, reward, terminated, truncated, info = env.step(int(action))
        state = next_state
        portfolio_history.append(env.equity)
        done = terminated or truncated
        steps += 1

    # 6. Визуализация результатов
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_history)
    plt.title("Динамика портфеля")
    plt.xlabel("Шаг")
    plt.ylabel("Стоимость портфеля (USD)")
    plt.grid(True)
    plt.savefig("portfolio_trend.png")
    print("📈 График сохранен: portfolio_trend.png")

    # 7. Расчет метрик
initial_balance = env.initial_balance
final_balance = env.equity
returns = (final_balance - initial_balance) / initial_balance * 100
# Расчет максимальной просадки
peak = initial_balance
max_drawdown = 0
for value in portfolio_history:
    if value > peak:
        peak = value
    drawdown = (peak - value) / peak * 100
    if drawdown > max_drawdown:
        max_drawdown = drawdown

# --- Advanced statistics ---
returns_series = pd.Series(np.diff(portfolio_history) / np.array(portfolio_history[:-1]))
try:
    sharpe_ratio = timeseries.sharpe_ratio(returns_series)
    max_dd = timeseries.max_drawdown(returns_series)
    calmar_ratio = returns_series.mean() / max_dd if max_dd != 0 else np.nan
except Exception as e:
    sharpe_ratio = np.nan
    max_dd = np.nan
    calmar_ratio = np.nan

print("\n📊 Результаты оценки:")
print(f"Начальный баланс: ")
print(f"Конечный баланс: ")
print(f"Доходность: {returns:.2f}%")
print(f"Максимальная просадка: {max_drawdown:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Calmar Ratio: {calmar_ratio:.2f}")

# 8. Сохранение логов
    log_df = pd.DataFrame({
        'step': np.arange(len(portfolio_history)),
        'portfolio_value': portfolio_history
    })
    log_df.to_csv('evaluation_log.csv', index=False)
    print("📝 Логи сохранены: evaluation_log.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Путь к файлу модели')
    parser.add_argument('--data', type=str, required=True, help='Путь к тестовым данным')
    parser.add_argument('--config', type=str, default='configs/btc_futures.yaml', help='Путь к конфигу среды')
    args = parser.parse_args()
    main(args.model, args.data, args.config)
