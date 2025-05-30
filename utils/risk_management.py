import numpy as np

def calculate_var(portfolio_value, position, current_price, confidence=0.95):
    """Упрощенный расчет Value at Risk"""
    return 0.02 * portfolio_value  # Фиксированный 2% риск

def apply_stop_loss(position, current_price, stop_loss, take_profit):
    """Применение стоп-лосс и тейк-профит"""
    if position > 0:  # Long позиция
        if current_price <= (1 - stop_loss):
            return 0  # Закрыть позицию
        elif current_price >= (1 + take_profit):
            return 0
    elif position < 0:  # Short позиция
        if current_price >= (1 + stop_loss):
            return 0
        elif current_price <= (1 - take_profit):
            return 0
    return position