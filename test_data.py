from utils.data_loader import load_historical_data

data = load_historical_data('BTC/USDT', '1m', limit=1000)
print("✔ Данные загружены")
print("Колонки:", data.columns.tolist())
print("NaN значений:", data.isnull().sum().sum())
