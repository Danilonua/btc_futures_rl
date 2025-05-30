import ccxt


def get_binance_client(api_key='YOUR_API_KEY', api_secret='YOUR_API_SECRET', testnet=True):
    config = {
        'apiKey': api_key,
        'secret': api_secret,
        'options': {'defaultType': 'future'},
        'enableRateLimit': True
    }

    if testnet:
        config['urls'] = {'api': 'https://testnet.binance.vision'}

    return ccxt.binance(config)