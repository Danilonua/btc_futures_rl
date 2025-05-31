import ccxt
import time
import pandas as pd

def load_historical_data(symbol, timeframe, limit=1000):
    # Для исторических данных используем публичный клиент без ключей
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

    data = []
    since = None
    while len(data) < limit:
        try:
            new_data = exchange.fetch_ohlcv(
                symbol,
                timeframe,
                since=since,
                limit=min(1000, limit - len(data))
            )
            if not new_data:
                break

            if since is None:
                data = new_data
            else:
                data.extend(new_data)

            since = new_data[-1][0] + 1
            time.sleep(0.1)  # Защита от rate limit

        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(5)

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df.sort_index().iloc[-limit:]
