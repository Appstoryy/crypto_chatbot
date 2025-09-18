import os
import io
import base64
import time
import math
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Библиотеки
# -------------------------
import gradio as gr
import ccxt
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Для LSTM
try:
    import torch.nn as nn
    LSTM_AVAILABLE = True
except:
    LSTM_AVAILABLE = False

# -------------------------
# Константы и начальные настройки
# -------------------------
MODEL_NAME = "cointegrated/rut5-base-multitask"
DEVICE = 0 if torch.cuda.is_available() else -1

DEFAULT_EXCHANGE = "binance"
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_LIMIT = 200

SYSTEM_PROMPT = (
    "Ты — дружелюбный русскоязычный ассистент по крипте и программированию. "
    "Отвечай кратко, по делу и предлагай конкретные шаги. "
    "Если пользователя интересуют цены или графики, подскажи команды: "
    "`/price SYMBOL TIMEFRAME EXCHANGE LIMIT`, `/chart SYMBOL TIMEFRAME EXCHANGE LIMIT`, "
    "`/indicators SYMBOL TIMEFRAME EXCHANGE LIMIT`, `/predict SYMBOL TIMEFRAME EXCHANGE LIMIT`, "
    "`/patterns SYMBOL TIMEFRAME EXCHANGE LIMIT`. "
    "Если запрос неясен — дай безопасный, полезный ответ."
)

# -------------------------
# Загрузка модели и пайплайна
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=DEVICE
)

# -------------------------
# Простейший датасет Q&A
# -------------------------
toy_data = {
    "instruction": [
        "Объясни, что делает команда /price в нашем боте.",
        "Как получить график по ETH/USDT?",
        "Дай совет по рискам при торговле фьючерсами.",
        "Как начать пользоваться ботом?",
        "Зачем нужна библиотека ccxt?",
        "Что такое RSI индикатор?",
        "Как работает LSTM прогнозирование?",
        "Какие паттерны может найти бот?"
    ],
    "answer": [
        "Команда /price SYMBOL TIMEFRAME EXCHANGE LIMIT возвращает краткую сводку цены и стату: последние котировки, минимум/максимум, среднюю, изменение. Пример: /price BTC/USDT 1h binance 100",
        "Используй команду /chart SYMBOL TIMEFRAME EXCHANGE LIMIT. Например: /chart ETH/USDT 15m binance 200 — бот построит свечной график и покажет его в интерфейсе.",
        "Торговля фьючерсами связана с высоким риском и возможностью ликвидации. Используйте стоп-лоссы, управляйте размером позиции, не превышайте риск на сделку, и избегайте чрезмерного левериджа.",
        "Напиши любое сообщение или воспользуйся командами. Примеры: /price BTC/USDT 1h binance 100 или /chart BTC/USDT 15m binance 200. Для справки — спроси у бота простыми словами.",
        "ccxt позволяет подключаться к множеству криптобирж из Python (и не только), получать котировки, отправлять ордера (при наличии ключей), а также унифицирует их API.",
        "RSI (Relative Strength Index) — индикатор силы, показывает перекупленность (>70) или перепроданность (<30) актива. Помогает найти точки разворота.",
        "LSTM (Long Short-Term Memory) — нейросеть для временных рядов. Анализирует исторические данные и предсказывает будущие цены на основе найденных паттернов.",
        "Бот ищет классические паттерны: двойная вершина/дно, голова и плечи, треугольники, флаги. Это помогает предсказать движение цены."
    ]
}
toy_dataset = Dataset.from_dict(toy_data)

# -------------------------
# Технические индикаторы на pandas
# -------------------------
def calculate_rsi(df, period=14):
    """Расчёт RSI"""
    try:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        print(f"RSI error: {e}")
        return pd.Series([50] * len(df))

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Расчёт MACD"""
    try:
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    except Exception as e:
        print(f"MACD error: {e}")
        return pd.Series([0] * len(df)), pd.Series([0] * len(df)), pd.Series([0] * len(df))

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Расчёт Bollinger Bands"""
    try:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    except Exception as e:
        print(f"BB error: {e}")
        return df['close'], df['close'], df['close']

def calculate_ema(df, period=20):
    """Расчёт EMA"""
    try:
        return df['close'].ewm(span=period, adjust=False).mean()
    except Exception as e:
        print(f"EMA error: {e}")
        return df['close']

# -------------------------
# Улучшенные индикаторы с надёжной обработкой ошибок
# -------------------------

def calculate_rsi_robust(df, period=14):
    """Улучшенный расчёт RSI с валидацией"""
    try:
        if len(df) < period + 1:
            return None, f"Недостаточно данных для RSI (нужно минимум {period + 1})"
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Проверка на деление на ноль
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        # Заполнение NaN значений
        rsi = rsi.fillna(50)  # Нейтральное значение
        
        return rsi, "OK"
    except Exception as e:
        return None, f"RSI ошибка: {e}"

def calculate_macd_robust(df, fast=12, slow=26, signal=9):
    """Улучшенный расчёт MACD"""
    try:
        if len(df) < slow + signal:
            return None, None, None, f"Недостаточно данных для MACD (нужно минимум {slow + signal})"
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram, "OK"
    except Exception as e:
        return None, None, None, f"MACD ошибка: {e}"

def calculate_bollinger_bands_robust(df, period=20, std_dev=2):
    """Улучшенный расчёт Bollinger Bands"""
    try:
        if len(df) < period:
            return None, None, None, f"Недостаточно данных для BB (нужно минимум {period})"
        
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        # Проверка на нулевое стандартное отклонение
        std = std.replace(0, np.nan)
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band, "OK"
    except Exception as e:
        return None, None, None, f"BB ошибка: {e}"

def validate_data_sufficiency(df, analysis_type="basic"):
    """Проверка достаточности данных для анализа"""
    min_requirements = {
        "basic": 10,
        "rsi": 15,
        "macd": 35,
        "bollinger": 20,
        "patterns": 30,
        "lstm": 60
    }
    
    required = min_requirements.get(analysis_type, 10)
    
    if len(df) < required:
        return False, f"Недостаточно данных: {len(df)}/{required} для {analysis_type}"
    
    return True, "OK"

def safe_technical_analysis(df):
    """Безопасный технический анализ с проверками"""
    results = {}
    
    # Проверка базовых данных
    is_valid, msg = validate_data_sufficiency(df, "basic")
    if not is_valid:
        return {"error": msg}
    
    # RSI
    is_valid, msg = validate_data_sufficiency(df, "rsi")
    if is_valid:
        rsi, rsi_msg = calculate_rsi_robust(df)
        results["rsi"] = {"data": rsi, "status": rsi_msg}
    else:
        results["rsi"] = {"data": None, "status": msg}
    
    # MACD
    is_valid, msg = validate_data_sufficiency(df, "macd")
    if is_valid:
        macd, signal, hist, macd_msg = calculate_macd_robust(df)
        results["macd"] = {
            "macd_line": macd,
            "signal_line": signal,
            "histogram": hist,
            "status": macd_msg
        }
    else:
        results["macd"] = {"data": None, "status": msg}
    
    # Bollinger Bands
    is_valid, msg = validate_data_sufficiency(df, "bollinger")
    if is_valid:
        upper, middle, lower, bb_msg = calculate_bollinger_bands_robust(df)
        results["bollinger"] = {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "status": bb_msg
        }
    else:
        results["bollinger"] = {"data": None, "status": msg}
    
    return results

# -------------------------
# LSTM модель для прогнозирования
# -------------------------
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def predict_lstm_improved(df, future_steps=10, epochs=50):
    """LSTM с минимальным обучением"""
    if not LSTM_AVAILABLE or len(df) < 60:
        # Fallback на простой линейный тренд
        return predict_linear_trend(df, future_steps), "Линейный тренд (недостаточно данных для LSTM)"
    
    try:
        # Подготовка данных
        prices = df['close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        prices_scaled = scaler.fit_transform(prices)
        
        # Создание последовательностей
        seq_length = 20
        X, y = [], []
        for i in range(len(prices_scaled) - seq_length):
            X.append(prices_scaled[i:i+seq_length, 0])
            y.append(prices_scaled[i+seq_length, 0])
        
        if len(X) < 20:  # Недостаточно данных для обучения
            return predict_linear_trend(df, future_steps), "Линейный тренд (мало данных)"
        
        X = torch.FloatTensor(np.array(X)).unsqueeze(-1)
        y = torch.FloatTensor(np.array(y))
        
        # Разделение на train/test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Модель
        model = SimpleLSTM()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Быстрое обучение
        model.train()
        for epoch in range(min(epochs, 50)):  # Максимум 50 эпох
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs.squeeze(), y_train)
            loss.backward()
            optimizer.step()
        
        # Тестирование качества модели
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred.squeeze(), y_test).item()
            
            # Если модель плохая, используем линейный тренд
            if test_loss > 0.1:  # Порог качества
                return predict_linear_trend(df, future_steps), f"Линейный тренд (LSTM качество низкое: {test_loss:.4f})"
        
        # Прогноз
        with torch.no_grad():
            last_seq = torch.FloatTensor(prices_scaled[-seq_length:, 0]).unsqueeze(0).unsqueeze(-1)
            predictions = []
            for _ in range(future_steps):
                pred = model(last_seq)
                predictions.append(pred.item())
                new_val = pred.unsqueeze(-1)
                last_seq = torch.cat([last_seq[:, 1:, :], new_val], dim=1)
        
        # Обратное масштабирование
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten(), f"LSTM прогноз (loss: {test_loss:.4f}, epochs: {epochs})"
        
    except Exception as e:
        return predict_linear_trend(df, future_steps), f"Линейный тренд (LSTM ошибка: {e})"

def predict_linear_trend(df, future_steps=10):
    """Простой линейный тренд как fallback"""
    try:
        prices = df['close'].values
        x = np.arange(len(prices))
        
        # Линейная регрессия
        z = np.polyfit(x, prices, 1)
        p = np.poly1d(z)
        
        # Прогноз
        future_x = np.arange(len(prices), len(prices) + future_steps)
        predictions = p(future_x)
        
        return predictions
    except:
        # Последний fallback - константа
        last_price = df['close'].iloc[-1]
        return np.full(future_steps, last_price)

def predict_moving_average(df, window=20, future_steps=10):
    """Прогноз на основе скользящего среднего"""
    try:
        prices = df['close'].values
        ma = np.mean(prices[-window:])
        trend = (prices[-1] - prices[-window]) / window  # Простой тренд
        
        predictions = []
        for i in range(future_steps):
            pred = ma + trend * i
            predictions.append(pred)
        
        return np.array(predictions)
    except:
        last_price = df['close'].iloc[-1]
        return np.full(future_steps, last_price)

# Оставляем старую функцию для совместимости
def predict_lstm(df, future_steps=10):
    """Обёртка для совместимости - использует улучшенную версию"""
    return predict_lstm_improved(df, future_steps, epochs=30)

# -------------------------
# Ансамбль методов прогнозирования
# -------------------------

def ensemble_prediction(df, future_steps=10):
    """Ансамбль различных методов прогнозирования"""
    predictions = {}
    weights = {}
    
    try:
        # 1. Линейный тренд (быстро и надежно)
        try:
            linear_pred = predict_linear_trend(df, future_steps)
            predictions['linear'] = linear_pred
            weights['linear'] = 0.3
        except:
            pass
        
        # 2. Экспоненциальное сглаживание
        try:
            exp_pred = predict_exponential_smoothing(df, future_steps)
            predictions['exponential'] = exp_pred
            weights['exponential'] = 0.25
        except:
            pass
        
        # 3. ARIMA (простая версия)
        try:
            arima_pred = predict_simple_arima(df, future_steps)
            predictions['arima'] = arima_pred
            weights['arima'] = 0.2
        except:
            pass
        
        # 4. Скользящее среднее с трендом
        try:
            ma_pred = predict_moving_average_trend(df, future_steps)
            predictions['ma_trend'] = ma_pred
            weights['ma_trend'] = 0.15
        except:
            pass
        
        # 5. LSTM (только если достаточно данных)
        if len(df) > 100:
            try:
                lstm_pred, lstm_status = predict_lstm_improved(df, future_steps, epochs=30)
                if "низкое качество" not in lstm_status:
                    predictions['lstm'] = lstm_pred
                    weights['lstm'] = 0.1
            except:
                pass
        
        # Нормализация весов
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Ансамблевый прогноз
        if predictions:
            ensemble_pred = np.zeros(future_steps)
            confidence_scores = []
            
            for method, pred in predictions.items():
                weight = weights.get(method, 0)
                ensemble_pred += pred * weight
                confidence_scores.append(weight)
            
            # Оценка доверия
            confidence = np.mean(confidence_scores) * len(predictions) / 5  # Нормализация на количество методов
            confidence = min(confidence, 1.0)
            
            # Доверительные интервалы
            pred_std = np.std([pred for pred in predictions.values()], axis=0)
            lower_bound = ensemble_pred - 1.96 * pred_std
            upper_bound = ensemble_pred + 1.96 * pred_std
            
            return {
                'prediction': ensemble_pred,
                'methods_used': list(predictions.keys()),
                'confidence': confidence,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'status': f"Ансамбль из {len(predictions)} методов"
            }
        else:
            # Fallback на простое среднее
            last_price = df['close'].iloc[-1]
            return {
                'prediction': np.full(future_steps, last_price),
                'methods_used': ['constant'],
                'confidence': 0.1,
                'lower_bound': np.full(future_steps, last_price * 0.95),
                'upper_bound': np.full(future_steps, last_price * 1.05),
                'status': "Fallback: константный прогноз"
            }
            
    except Exception as e:
        last_price = df['close'].iloc[-1]
        return {
            'prediction': np.full(future_steps, last_price),
            'methods_used': ['error_fallback'],
            'confidence': 0.05,
            'lower_bound': np.full(future_steps, last_price * 0.9),
            'upper_bound': np.full(future_steps, last_price * 1.1),
            'status': f"Ошибка: {e}"
        }

def predict_exponential_smoothing(df, future_steps=10, alpha=0.3):
    """Экспоненциальное сглаживание"""
    prices = df['close'].values
    
    # Простое экспоненциальное сглаживание
    smoothed = [prices[0]]
    for i in range(1, len(prices)):
        smoothed.append(alpha * prices[i] + (1 - alpha) * smoothed[-1])
    
    # Прогноз (с учетом тренда)
    if len(smoothed) > 10:
        recent_trend = (smoothed[-1] - smoothed[-10]) / 10
    else:
        recent_trend = 0
    
    predictions = []
    last_value = smoothed[-1]
    
    for i in range(future_steps):
        pred = last_value + recent_trend * (i + 1)
        predictions.append(pred)
    
    return np.array(predictions)

def predict_simple_arima(df, future_steps=10):
    """Упрощенная ARIMA (AR модель)"""
    prices = df['close'].values
    
    # Простая авторегрессионная модель AR(3)
    if len(prices) < 10:
        return predict_linear_trend(df, future_steps)
    
    # Разности для стационарности
    diffs = np.diff(prices)
    
    # Простая AR(3) модель через корреляцию
    if len(diffs) > 6:
        x = np.column_stack([diffs[2:-1], diffs[1:-2], diffs[:-3]])
        y = diffs[3:]
        
        # Псевдо-коэффициенты через корреляцию
        coef = np.corrcoef(x.T, y)[:-1, -1]
        coef = coef / np.sum(np.abs(coef))  # Нормализация
        
        # Прогноз
        last_diffs = diffs[-3:]
        predictions = []
        current_price = prices[-1]
        
        for _ in range(future_steps):
            next_diff = np.dot(coef, last_diffs)
            next_price = current_price + next_diff
            predictions.append(next_price)
            
            # Обновляем окно разностей
            last_diffs = np.roll(last_diffs, -1)
            last_diffs[-1] = next_diff
            current_price = next_price
        
        return np.array(predictions)
    else:
        return predict_linear_trend(df, future_steps)

def predict_moving_average_trend(df, future_steps=10, short_window=5, long_window=20):
    """Прогноз на основе пересечения скользящих средних"""
    prices = df['close'].values
    
    if len(prices) < long_window:
        return predict_linear_trend(df, future_steps)
    
    # Скользящие средние
    short_ma = np.convolve(prices, np.ones(short_window)/short_window, mode='valid')
    long_ma = np.convolve(prices, np.ones(long_window)/long_window, mode='valid')
    
    # Выравниваем длины
    min_len = min(len(short_ma), len(long_ma))
    short_ma = short_ma[-min_len:]
    long_ma = long_ma[-min_len:]
    
    # Определяем тренд
    if short_ma[-1] > long_ma[-1]:
        # Восходящий тренд
        trend = (short_ma[-1] - short_ma[-min(5, len(short_ma))]) / min(5, len(short_ma))
        base_price = short_ma[-1]
    else:
        # Нисходящий тренд  
        trend = (long_ma[-1] - long_ma[-min(5, len(long_ma))]) / min(5, len(long_ma))
        base_price = long_ma[-1]
    
    # Прогноз с затуханием тренда
    predictions = []
    for i in range(future_steps):
        decay_factor = 0.95 ** i  # Затухание тренда
        pred = base_price + trend * (i + 1) * decay_factor
        predictions.append(pred)
    
    return np.array(predictions)

def calculate_prediction_accuracy(df, method_func, lookback_periods=[5, 10, 20]):
    """Оценка точности метода прогнозирования на исторических данных"""
    accuracies = []
    
    for period in lookback_periods:
        if len(df) < period + 10:
            continue
            
        try:
            # Берем исторические данные
            train_df = df.iloc[:-period]
            actual_prices = df['close'].iloc[-period:].values
            
            # Делаем прогноз
            if method_func == predict_lstm_improved:
                predicted_prices, _ = method_func(train_df, period)
            else:
                predicted_prices = method_func(train_df, period)
            
            if predicted_prices is not None and len(predicted_prices) == len(actual_prices):
                # Считаем MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
                accuracies.append(100 - min(mape, 100))  # Точность в процентах
        except:
            continue
    
    return np.mean(accuracies) if accuracies else 0

def adaptive_ensemble_prediction(df, future_steps=10):
    """Адаптивный ансамбль с весами на основе исторической точности"""
    
    # Список методов и их функций
    methods = {
        'linear': predict_linear_trend,
        'exponential': predict_exponential_smoothing, 
        'arima': predict_simple_arima,
        'ma_trend': predict_moving_average_trend
    }
    
    # Оценка точности каждого метода
    method_accuracies = {}
    for name, func in methods.items():
        try:
            accuracy = calculate_prediction_accuracy(df, func)
            method_accuracies[name] = max(accuracy, 5)  # Минимум 5% точности
        except:
            method_accuracies[name] = 5
    
    # Нормализация весов на основе точности
    total_accuracy = sum(method_accuracies.values())
    adaptive_weights = {k: v/total_accuracy for k, v in method_accuracies.items()}
    
    # Получение прогнозов
    predictions = {}
    for name, func in methods.items():
        try:
            pred = func(df, future_steps)
            if pred is not None:
                predictions[name] = pred
        except:
            continue
    
    # LSTM только если данных достаточно и другие методы показывают приемлемую точность
    if len(df) > 100 and np.mean(list(method_accuracies.values())) > 20:
        try:
            lstm_pred, lstm_status = predict_lstm_improved(df, future_steps, epochs=50)
            if "низкое качество" not in lstm_status:
                predictions['lstm'] = lstm_pred
                adaptive_weights['lstm'] = 0.15  # Небольшой вес для LSTM
                
                # Перенормализация
                total_weight = sum(adaptive_weights.values())
                adaptive_weights = {k: v/total_weight for k, v in adaptive_weights.items()}
        except:
            pass
    
    # Создание ансамбля
    if predictions:
        ensemble_pred = np.zeros(future_steps)
        used_methods = []
        
        for method, pred in predictions.items():
            weight = adaptive_weights.get(method, 0)
            if weight > 0.01:  # Используем только методы с весом > 1%
                ensemble_pred += pred * weight
                used_methods.append(f"{method}({weight:.2f})")
        
        # Расчет доверительных интервалов
        pred_values = list(predictions.values())
        pred_std = np.std(pred_values, axis=0) if len(pred_values) > 1 else np.zeros(future_steps)
        
        # Адаптивная ширина интервала на основе разброса прогнозов
        interval_width = 1.96 + 0.5 * (np.mean(pred_std) / np.mean(ensemble_pred)) if np.mean(ensemble_pred) > 0 else 1.96
        
        lower_bound = ensemble_pred - interval_width * pred_std
        upper_bound = ensemble_pred + interval_width * pred_std
        
        # Общая уверенность
        confidence = min(np.mean(list(method_accuracies.values())) / 100, 0.95)
        
        return {
            'prediction': ensemble_pred,
            'methods_used': used_methods,
            'confidence': confidence,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'method_accuracies': method_accuracies,
            'status': f"Адаптивный ансамбль (уверенность: {confidence:.1%})"
        }
    else:
        # Fallback
        last_price = df['close'].iloc[-1]
        return {
            'prediction': np.full(future_steps, last_price),
            'methods_used': ['fallback'],
            'confidence': 0.05,
            'lower_bound': np.full(future_steps, last_price * 0.9),
            'upper_bound': np.full(future_steps, last_price * 1.1),
            'status': "Все методы недоступны, используется константа"
        }

# -------------------------
# Детекция паттернов
# -------------------------
def detect_patterns(df):
    """Простая детекция паттернов"""
    patterns = []
    
    try:
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        if len(closes) < 20:
            return ["Недостаточно данных для анализа паттернов"]
        
        # Двойная вершина
        if len(highs) >= 10:
            recent_highs = highs[-10:]
            peaks = []
            for i in range(1, len(recent_highs)-1):
                if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
                    peaks.append((i, recent_highs[i]))
            if len(peaks) >= 2:
                if abs(peaks[-1][1] - peaks[-2][1]) / peaks[-1][1] < 0.02:
                    patterns.append("🔴 Двойная вершина - возможен разворот вниз")
        
        # Двойное дно
        if len(lows) >= 10:
            recent_lows = lows[-10:]
            bottoms = []
            for i in range(1, len(recent_lows)-1):
                if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
                    bottoms.append((i, recent_lows[i]))
            if len(bottoms) >= 2:
                if abs(bottoms[-1][1] - bottoms[-2][1]) / bottoms[-1][1] < 0.02:
                    patterns.append("🟢 Двойное дно - возможен разворот вверх")
        
        # Треугольник (сужение диапазона)
        if len(highs) >= 20:
            recent_range = highs[-20:] - lows[-20:]
            first_half_avg = np.mean(recent_range[:10])
            second_half_avg = np.mean(recent_range[10:])
            if second_half_avg < first_half_avg * 0.7:
                patterns.append("📐 Треугольник - ожидается прорыв")
        
        # Тренд
        if len(closes) >= 20:
            sma_20 = np.mean(closes[-20:])
            sma_5 = np.mean(closes[-5:])
            if sma_5 > sma_20 * 1.02:
                patterns.append("📈 Восходящий тренд")
            elif sma_5 < sma_20 * 0.98:
                patterns.append("📉 Нисходящий тренд")
            else:
                patterns.append("➡️ Боковое движение")
        
        if not patterns:
            patterns.append("Явных паттернов не обнаружено")
            
    except Exception as e:
        patterns.append(f"Ошибка анализа: {e}")
    
    return patterns

# -------------------------
# Улучшенная детекция паттернов
# -------------------------

def detect_patterns_improved(df):
    """Улучшенная детекция паттернов с более строгими условиями"""
    patterns = []
    
    try:
        if len(df) < 30:
            return ["Недостаточно данных для анализа паттернов (минимум 30 свечей)"]
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values if 'volume' in df.columns else None
        
        # 1. Двойная вершина (улучшенная версия)
        double_top = detect_double_top(highs, closes, min_distance=10, tolerance=0.015)
        if double_top:
            patterns.append(f"🔴 Двойная вершина: {double_top}")
        
        # 2. Двойное дно (улучшенная версия)
        double_bottom = detect_double_bottom(lows, closes, min_distance=10, tolerance=0.015)
        if double_bottom:
            patterns.append(f"🟢 Двойное дно: {double_bottom}")
        
        # 3. Голова и плечи
        head_shoulders = detect_head_and_shoulders(highs, closes)
        if head_shoulders:
            patterns.append(f"🔴 Голова и плечи: {head_shoulders}")
        
        # 4. Восходящий/нисходящий треугольник
        triangle = detect_triangle_pattern(highs, lows, closes)
        if triangle:
            patterns.append(f"📐 {triangle}")
        
        # 5. Прорыв с объёмом (если есть данные по объёму)
        if volumes is not None:
            breakout = detect_volume_breakout(closes, volumes)
            if breakout:
                patterns.append(f"💥 {breakout}")
        
        # 6. Дивергенция RSI
        rsi_div = detect_rsi_divergence(df)
        if rsi_div:
            patterns.append(f"⚡ {rsi_div}")
        
        # 7. Анализ тренда (более точный)
        trend_analysis = analyze_trend_strength(closes)
        patterns.append(f"📊 {trend_analysis}")
        
        if len(patterns) == 1 and "тренд" in patterns[0].lower():
            patterns.append("Других явных паттернов не обнаружено")
            
    except Exception as e:
        patterns.append(f"Ошибка анализа паттернов: {e}")
    
    return patterns

def detect_double_top(highs, closes, min_distance=10, tolerance=0.015):
    """Детекция двойной вершины с улучшенными условиями"""
    try:
        if len(highs) < min_distance * 2 + 5:
            return None
            
        peaks = []
        # Поиск локальных максимумов
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i+1] and 
                highs[i] > highs[i-2] and highs[i] > highs[i+2]):
                peaks.append((i, highs[i]))
        
        if len(peaks) < 2:
            return None
        
        # Проверяем последние пики
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                peak1_idx, peak1_price = peaks[i]
                peak2_idx, peak2_price = peaks[j]
                
                # Условия для двойной вершины:
                # 1. Расстояние между пиками
                if peak2_idx - peak1_idx < min_distance:
                    continue
                    
                # 2. Высота пиков близка (в пределах tolerance)
                price_diff = abs(peak1_price - peak2_price) / max(peak1_price, peak2_price)
                if price_diff > tolerance:
                    continue
                
                # 3. Между пиками есть значимый спад (ищем минимум в lows, не в highs!)
                valley_start_idx = peak1_idx
                valley_end_idx = peak2_idx + 1
                valley_lows = lows[valley_start_idx:valley_end_idx] if valley_start_idx < len(lows) and valley_end_idx <= len(lows) else lows[valley_start_idx:peak2_idx]
                if len(valley_lows) == 0:
                    continue
                valley_idx = valley_start_idx + np.argmin(valley_lows)
                valley_price = lows[valley_idx] if valley_idx < len(lows) else min(peak1_price, peak2_price)
                valley_depth = min(peak1_price, peak2_price) - valley_price
                min_valley_depth = max(peak1_price, peak2_price) * 0.03  # Минимум 3% спад
                
                if valley_depth > min_valley_depth:
                    confidence = max(0, 100 - price_diff * 100)
                    return f"Уровни {peak1_price:.4f} и {peak2_price:.4f} (уверенность: {confidence:.1f}%)"
        
        return None
    except:
        return None

def detect_double_bottom(lows, closes, min_distance=10, tolerance=0.015):
    """Детекция двойного дна"""
    try:
        if len(lows) < min_distance * 2 + 5:
            return None
            
        troughs = []
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i+1] and 
                lows[i] < lows[i-2] and lows[i] < lows[i+2]):
                troughs.append((i, lows[i]))
        
        if len(troughs) < 2:
            return None
        
        for i in range(len(troughs) - 1):
            for j in range(i + 1, len(troughs)):
                trough1_idx, trough1_price = troughs[i]
                trough2_idx, trough2_price = troughs[j]
                
                if trough2_idx - trough1_idx < min_distance:
                    continue
                
                price_diff = abs(trough1_price - trough2_price) / min(trough1_price, trough2_price)
                if price_diff > tolerance:
                    continue
                
                # Проверка на наличие пика между впадинами (ищем максимум в highs, не в lows!)
                peak_start_idx = trough1_idx
                peak_end_idx = trough2_idx + 1
                peak_highs = highs[peak_start_idx:peak_end_idx] if peak_start_idx < len(highs) and peak_end_idx <= len(highs) else highs[peak_start_idx:trough2_idx]
                if len(peak_highs) == 0:
                    continue
                peak_idx = peak_start_idx + np.argmax(peak_highs)
                peak_price = highs[peak_idx] if peak_idx < len(highs) else max(trough1_price, trough2_price)
                peak_height = peak_price - max(trough1_price, trough2_price)
                min_peak_height = min(trough1_price, trough2_price) * 0.03
                
                if peak_height > min_peak_height:
                    confidence = max(0, 100 - price_diff * 100)
                    return f"Уровни {trough1_price:.4f} и {trough2_price:.4f} (уверенность: {confidence:.1f}%)"
        
        return None
    except:
        return None

def detect_head_and_shoulders(highs, closes):
    """Детекция паттерна 'голова и плечи'"""
    try:
        if len(highs) < 20:
            return None
        
        # Поиск трёх основных пиков
        peaks = []
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i+1] and 
                highs[i] > highs[i-2] and highs[i] > highs[i+2]):
                peaks.append((i, highs[i]))
        
        if len(peaks) < 3:
            return None
        
        # Проверяем последние три пика
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]
            
            # Условия для головы и плеч:
            # 1. Голова выше плеч
            if (head[1] > left_shoulder[1] * 1.02 and 
                head[1] > right_shoulder[1] * 1.02):
                # 2. Плечи примерно на одном уровне
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / max(left_shoulder[1], right_shoulder[1])
                if shoulder_diff < 0.05:  # 5% разница
                    return f"Голова {head[1]:.4f}, плечи {left_shoulder[1]:.4f}/{right_shoulder[1]:.4f}"
        
        return None
    except:
        return None

def detect_triangle_pattern(highs, lows, closes, min_points=6):
    """Детекция треугольных паттернов"""
    try:
        if len(closes) < min_points * 2:
            return None
        
        recent_highs = highs[-min_points*2:]
        recent_lows = lows[-min_points*2:]
        
        # Линейная регрессия для максимумов и минимумов
        x = np.arange(len(recent_highs))
        
        # Тренд максимумов
        high_slope = np.polyfit(x, recent_highs, 1)[0]
        # Тренд минимумов  
        low_slope = np.polyfit(x, recent_lows, 1)[0]
        
        # Сходящийся треугольник
        if high_slope < -0.001 and low_slope > 0.001:
            return "Симметричный треугольник - ожидается прорыв"
        # Восходящий треугольник
        elif abs(high_slope) < 0.0005 and low_slope > 0.001:
            return "Восходящий треугольник - бычий сигнал"
        # Нисходящий треугольник  
        elif high_slope < -0.001 and abs(low_slope) < 0.0005:
            return "Нисходящий треугольник - медвежий сигнал"
        
        return None
    except:
        return None

def detect_volume_breakout(closes, volumes):
    """Детекция прорыва с подтверждением объёма"""
    try:
        if len(closes) < 20 or len(volumes) < 20:
            return None
        
        recent_closes = closes[-10:]
        recent_volumes = volumes[-10:]
        
        # Средний объём за предыдущий период
        avg_volume = np.mean(volumes[-20:-10])
        current_volume = volumes[-1]
        
        # Прорыв цены
        resistance = np.max(closes[-20:-5])  # Уровень сопротивления
        current_price = closes[-1]
        
        # Условия прорыва:
        # 1. Цена пробила уровень сопротивления
        # 2. Объём выше среднего
        if (current_price > resistance * 1.01 and  # 1% прорыв
            current_volume > avg_volume * 1.5):     # Объём в 1.5 раза больше
            return f"Прорыв уровня {resistance:.4f} с увеличенным объёмом"
        
        return None
    except:
        return None

def detect_rsi_divergence(df):
    """Детекция дивергенции RSI"""
    try:
        if len(df) < 30:
            return None
            
        rsi, status = calculate_rsi_robust(df, 14)
        if rsi is None:
            return None
        
        prices = df['close'].values[-20:]  # Последние 20 свечей
        rsi_values = rsi.values[-20:]
        
        # Поиск локальных максимумов в цене и RSI
        price_peaks = []
        rsi_peaks = []
        
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                price_peaks.append((i, prices[i]))
            if rsi_values[i] > rsi_values[i-1] and rsi_values[i] > rsi_values[i+1]:
                rsi_peaks.append((i, rsi_values[i]))
        
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            # Медвежья дивергенция: цена растет, RSI падает
            if (price_peaks[-1][1] > price_peaks[-2][1] and 
                rsi_peaks[-1][1] < rsi_peaks[-2][1]):
                return "Медвежья дивергенция RSI - возможен разворот вниз"
            # Бычья дивергенция: цена падает, RSI растет
            elif (price_peaks[-1][1] < price_peaks[-2][1] and 
                  rsi_peaks[-1][1] > rsi_peaks[-2][1]):
                return "Бычья дивергенция RSI - возможен разворот вверх"
        
        return None
    except:
        return None

def analyze_trend_strength(closes, periods=[5, 10, 20]):
    """Анализ силы тренда по нескольким периодам"""
    try:
        if len(closes) < max(periods):
            return "Недостаточно данных для анализа тренда"
        
        trends = []
        for period in periods:
            start_price = np.mean(closes[-period-5:-period])
            end_price = np.mean(closes[-5:])
            change = (end_price - start_price) / start_price * 100
            trends.append(change)
        
        # Определение силы тренда
        avg_trend = np.mean(trends)
        trend_consistency = len([t for t in trends if t * avg_trend > 0]) / len(trends)
        
        if abs(avg_trend) < 1:
            strength = "слабый"
        elif abs(avg_trend) < 3:
            strength = "умеренный"  
        else:
            strength = "сильный"
        
        direction = "восходящий" if avg_trend > 0 else "нисходящий"
        
        return f"{direction.title()} {strength} тренд ({avg_trend:+.2f}%, стабильность: {trend_consistency:.1%})"
        
    except:
        return "Ошибка анализа тренда"

# -------------------------
# Вспомогательные функции
# -------------------------
def few_shot_prefix(user_text: str, k: int = 2) -> str:
    text = user_text.lower()
    scored = []
    for inst, ans in zip(toy_dataset["instruction"], toy_dataset["answer"]):
        score = sum(1 for token in set(text.split()) if token in inst.lower())
        scored.append((score, inst, ans))
    scored.sort(key=lambda x: x[0], reverse=True)
    shots = scored[:k]
    prefix_lines = ["Примеры (для помощи модели):"]
    for _, inst, ans in shots:
        prefix_lines.append(f"Вопрос: {inst}\nОтвет: {ans}")
    return "\n\n".join(prefix_lines)

def fetch_ohlcv(symbol: str, timeframe: str, exchange_name: str, limit: int):
    exchange_name = exchange_name.lower()
    if not hasattr(ccxt, exchange_name):
        raise ValueError(f"Неизвестная биржа: {exchange_name}")
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({"enableRateLimit": True})
    markets = exchange.load_markets()
    if symbol not in markets:
        raise ValueError(f"Символ {symbol} не найден на {exchange_name}")
    ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
    return ohlcv

def ohlcv_to_dataframe(ohlcv):
    """Конвертация OHLCV в DataFrame с правильным временным индексом"""
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # Устанавливаем timestamp как индекс для корректной работы с временными осями
    df.set_index('timestamp', inplace=True)
    return df

def ohlcv_summary(ohlcv):
    if not ohlcv:
        return "Нет данных."
    closes = [c[4] for c in ohlcv]
    highs = [c[2] for c in ohlcv]
    lows = [c[3] for c in ohlcv]
    last = closes[-1]
    mean = sum(closes) / len(closes)
    min_p = min(lows)
    max_p = max(highs)
    chg = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] != 0 else 0.0
    return (
        f"Последняя цена: {last:.4f}\n"
        f"Средняя за период: {mean:.4f}\n"
        f"Мин/Макс: {min_p:.4f} / {max_p:.4f}\n"
        f"Изменение: {chg:+.2f}%"
    )

def plot_candles(ohlcv, title="Chart"):
    if not ohlcv:
        return None
    
    # Конвертируем в DataFrame для удобной работы с временем
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # Используем временные индексы для правильного позиционирования
    timestamps = df['timestamp']
    opens = df['open'].values
    highs = df['high'].values  
    lows = df['low'].values
    closes = df['close'].values
    
    # Вычисляем ширину свечей на основе временного интервала
    if len(timestamps) > 1:
        time_diff = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds()
        width_seconds = time_diff * 0.6
        width = pd.Timedelta(seconds=width_seconds)
    else:
        width = pd.Timedelta(minutes=30)  # fallback
    
    for i in range(len(timestamps)):
        color = "green" if closes[i] >= opens[i] else "red"
        
        # Рисуем тень (high-low линия)
        ax.plot([timestamps.iloc[i], timestamps.iloc[i]], [lows[i], highs[i]], 
                color="black", linewidth=1)
        
        # Рисуем тело свечи
        body_height = abs(closes[i] - opens[i])
        body_bottom = min(opens[i], closes[i])
        
        rect = plt.Rectangle((timestamps.iloc[i] - width/2, body_bottom),
                           width, body_height,
                           fill=True, alpha=0.7, edgecolor="black", 
                           linewidth=0.5, facecolor=color)
        ax.add_patch(rect)
    
    # Форматируем ось времени
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(timestamps)//10)))
    
    # Поворачиваем подписи для лучшей читаемости
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def plot_indicators(df, title="Technical Indicators"):
    """Построение графика с индикаторами"""
    try:
        # Используем безопасный анализ
        analysis = safe_technical_analysis(df)
        
        if "error" in analysis:
            # Создаём простой график только с ценой
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.index, df['close'], label='Close', color='black', linewidth=1.5)
            ax.set_title(f"{title} - {analysis['error']}")
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            fig = plt.figure(figsize=(14, 10))
            gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
            
            # График цены с Bollinger Bands и EMA
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(df.index, df['close'], label='Close', color='black', linewidth=1.5)
            
            # Bollinger Bands
            bb = analysis.get("bollinger", {})
            if bb.get("status") == "OK":
                upper, middle, lower = bb["upper"], bb["middle"], bb["lower"]
                ax1.plot(df.index, upper, 'b--', alpha=0.5, label='BB Upper')
                ax1.plot(df.index, middle, 'g-', alpha=0.5, label='BB Middle')
                ax1.plot(df.index, lower, 'b--', alpha=0.5, label='BB Lower')
            
            # EMA (используем старые функции для EMA - они простые и надёжные)
            if len(df) >= 20:
                ema20 = calculate_ema(df, 20)
                ax1.plot(df.index, ema20, 'orange', alpha=0.7, label='EMA 20')
            if len(df) >= 50:
                ema50 = calculate_ema(df, 50)
                ax1.plot(df.index, ema50, 'purple', alpha=0.7, label='EMA 50')
            
            ax1.set_title(title)
            ax1.set_ylabel('Price')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # RSI
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            rsi_data = analysis.get("rsi", {})
            if rsi_data.get("status") == "OK" and rsi_data["data"] is not None:
                rsi = rsi_data["data"]
                ax2.plot(df.index, rsi, 'purple', label='RSI')
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                ax2.fill_between(df.index, 70, 100, alpha=0.2, color='red')
                ax2.fill_between(df.index, 0, 30, alpha=0.2, color='green')
                ax2.set_ylabel('RSI')
                ax2.set_ylim(0, 100)
            else:
                ax2.text(0.5, 0.5, f"RSI: {rsi_data.get('status', 'Недоступен')}", 
                        transform=ax2.transAxes, ha='center', va='center')
                ax2.set_ylabel('RSI (недоступен)')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # MACD
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            macd_data = analysis.get("macd", {})
            if macd_data.get("status") == "OK":
                macd = macd_data["macd_line"]
                signal = macd_data["signal_line"]
                histogram = macd_data["histogram"]
                ax3.plot(df.index, macd, 'blue', label='MACD')
                ax3.plot(df.index, signal, 'red', label='Signal')
                ax3.bar(df.index, histogram, alpha=0.3, label='Histogram')
                ax3.set_ylabel('MACD')
            else:
                ax3.text(0.5, 0.5, f"MACD: {macd_data.get('status', 'Недоступен')}", 
                        transform=ax3.transAxes, ha='center', va='center')
                ax3.set_ylabel('MACD (недоступен)')
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)
            
            # Volume
            ax4 = fig.add_subplot(gs[3], sharex=ax1)
            if 'volume' in df.columns:
                colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red' 
                          for i in range(len(df))]
                ax4.bar(df.index, df['volume'], color=colors, alpha=0.5)
                ax4.set_ylabel('Volume')
            else:
                ax4.text(0.5, 0.5, "Volume: Данные недоступны", 
                        transform=ax4.transAxes, ha='center', va='center')
            ax4.set_xlabel('Time')
            ax4.grid(True, alpha=0.3)
            
            # Форматируем ось времени для всех подграфиков
            import matplotlib.dates as mdates
            if hasattr(df.index, 'to_pydatetime'):  # Проверяем, что индекс - временной
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
                ax1.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df)//8)))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
                # Скрываем подписи на промежуточных осях для чистоты
                plt.setp(ax2.xaxis.get_majorticklabels(), visible=False)
                plt.setp(ax3.xaxis.get_majorticklabels(), visible=False)
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        print(f"Plot indicators error: {e}")
        return None

def plot_prediction(df, predictions, title="Price Prediction"):
    """График с прогнозом с правильным временным форматированием"""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Исторические данные
        ax.plot(df.index, df['close'].values, 'b-', label='Historical', linewidth=1.5)
        
        # Прогноз
        if predictions is not None:
            # Создаём будущие временные метки
            if hasattr(df.index, 'freq') and df.index.freq:
                # Используем частоту индекса если доступна
                future_times = pd.date_range(start=df.index[-1], periods=len(predictions)+1, freq=df.index.freq)[1:]
            else:
                # Вычисляем среднюю разность между временными метками
                if len(df.index) > 1:
                    time_diff = df.index[-1] - df.index[-2]
                    future_times = [df.index[-1] + time_diff * (i+1) for i in range(len(predictions))]
                else:
                    # Fallback: используем часовые интервалы
                    future_times = pd.date_range(start=df.index[-1], periods=len(predictions)+1, freq='H')[1:]
            
            ax.plot(future_times, predictions, 'r--', label='Prediction', linewidth=2)
            
            # Доверительный интервал (упрощённый)
            std = np.std(df['close'].values[-20:])
            upper = predictions + std
            lower = predictions - std
            ax.fill_between(future_times, lower, upper, alpha=0.2, color='red')
        
        # Форматируем ось времени
        import matplotlib.dates as mdates
        if hasattr(df.index, 'to_pydatetime'):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df)//8)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        print(f"Plot prediction error: {e}")
        return None

def plot_ensemble_prediction(df, ensemble_result, title="Ensemble Prediction"):
    """График с ансамбль-прогнозом и доверительными интервалами"""
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Исторические данные
        ax.plot(df.index, df['close'].values, 'b-', label='Historical Price', linewidth=2)
        
        # Прогноз
        prediction = ensemble_result.get('prediction')
        if prediction is not None:
            # Создаём будущие временные метки
            if hasattr(df.index, 'freq') and df.index.freq:
                future_times = pd.date_range(start=df.index[-1], periods=len(prediction)+1, freq=df.index.freq)[1:]
            else:
                if len(df.index) > 1:
                    time_diff = df.index[-1] - df.index[-2]
                    future_times = [df.index[-1] + time_diff * (i+1) for i in range(len(prediction))]
                else:
                    future_times = pd.date_range(start=df.index[-1], periods=len(prediction)+1, freq='H')[1:]
            
            ax.plot(future_times, prediction, 'r-', label='Ensemble Prediction', linewidth=3)
            
            # Доверительные интервалы
            lower_bound = ensemble_result.get('lower_bound')
            upper_bound = ensemble_result.get('upper_bound')
            
            if lower_bound is not None and upper_bound is not None:
                ax.fill_between(future_times, lower_bound, upper_bound, 
                               alpha=0.2, color='red', label='Confidence Interval')
                ax.plot(future_times, lower_bound, 'r--', alpha=0.5, linewidth=1)
                ax.plot(future_times, upper_bound, 'r--', alpha=0.5, linewidth=1)
            
            # Вертикальная линия разделения
            ax.axvline(x=df.index[-1], color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
            
            # Аннотации
            current_price = df['close'].iloc[-1]
            final_pred = prediction[-1]
            change_pct = (final_pred - current_price) / current_price * 100
            
            # Стрелка изменения
            if len(future_times) > 0:
                arrow_color = 'green' if change_pct > 0 else 'red'
                mid_time = future_times[len(future_times)//2] if len(future_times) > 1 else future_times[0]
                ax.annotate(f'{change_pct:+.2f}%', 
                           xy=(mid_time, (current_price + final_pred)/2),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=arrow_color, alpha=0.3),
                           arrowprops=dict(arrowstyle='->', color=arrow_color))
        
        # Настройка графика
        ax.set_title(f"{title}\nMethods: {', '.join(ensemble_result.get('methods_used', []))}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Форматируем ось времени
        import matplotlib.dates as mdates
        if hasattr(df.index, 'to_pydatetime'):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df)//6)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Информация об уверенности
        confidence = ensemble_result.get('confidence', 0)
        ax.text(0.02, 0.98, f'Confidence: {confidence:.1%}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        print(f"Plot ensemble prediction error: {e}")
        # Fallback на обычный график прогноза
        return plot_prediction(df, ensemble_result.get('prediction'), title)

# -------------------------
# Безопасный анализатор данных
# -------------------------

class SafeAnalyzer:
    """Безопасный анализатор с валидацией и улучшенными методами"""
    
    def __init__(self):
        self.min_data_requirements = {
            'basic': 10,
            'indicators': 30,
            'patterns': 50,
            'prediction': 60
        }
    
    def get_safe_data(self, symbol, timeframe, exchange, limit):
        """Безопасное получение данных с валидацией"""
        try:
            ohlcv = fetch_ohlcv(symbol, timeframe, exchange, limit)
            df = ohlcv_to_dataframe(ohlcv)
            
            if len(df) < self.min_data_requirements['basic']:
                return None, f"Недостаточно данных: {len(df)} свечей (минимум {self.min_data_requirements['basic']})"
            
            return df, "OK"
        except Exception as e:
            return None, f"Ошибка получения данных: {e}"
    
    def safe_technical_analysis(self, df):
        """Безопасный технический анализ"""
        if len(df) < self.min_data_requirements['indicators']:
            return {"error": f"Недостаточно данных для индикаторов: {len(df)}/{self.min_data_requirements['indicators']}"}
        
        return safe_technical_analysis(df)
    
    def safe_prediction(self, df, future_steps=10):
        """Безопасное прогнозирование с ансамблем"""
        if len(df) < self.min_data_requirements['prediction']:
            return {"error": f"Недостаточно данных для прогноза: {len(df)}/{self.min_data_requirements['prediction']}"}
        
        try:
            # Используем адаптивный ансамбль
            result = adaptive_ensemble_prediction(df, future_steps)
            return result
        except Exception as e:
            return {"error": f"Ошибка прогнозирования: {e}"}

# Создаем глобальный экземпляр
safe_analyzer = SafeAnalyzer()

def plot_indicators_safe(df, title="Technical Indicators"):
    """Безопасная версия построения графика индикаторов"""
    try:
        return plot_indicators(df, title)
    except Exception as e:
        print(f"Error in plot_indicators_safe: {e}")
        # Fallback на простой график цены
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['close'], label='Close Price', linewidth=2)
        ax.set_title(f"{title} - Simplified View")
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

def plot_prediction_advanced(df, predictions, lower_bound, upper_bound, title="Advanced Prediction"):
    """Продвинутый график прогноза с доверительными интервалами"""
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Исторические данные
        ax.plot(range(len(df)), df['close'].values, 'b-', label='Historical Price', linewidth=2)
        
        # Прогноз
        if predictions is not None:
            future_x = range(len(df), len(df) + len(predictions))
            ax.plot(future_x, predictions, 'r-', label='Prediction', linewidth=3)
            
            # Доверительные интервалы
            if lower_bound is not None and upper_bound is not None:
                ax.fill_between(future_x, lower_bound, upper_bound, 
                               alpha=0.2, color='red', label='Confidence Interval')
                ax.plot(future_x, lower_bound, 'r--', alpha=0.5, linewidth=1)
                ax.plot(future_x, upper_bound, 'r--', alpha=0.5, linewidth=1)
            
            # Разделительная линия
            ax.axvline(x=len(df)-1, color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
        
        ax.set_title(title)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        print(f"Error in plot_prediction_advanced: {e}")
        return plot_prediction(df, predictions, title)

# -------------------------
# Улучшенная функция чата
# -------------------------

def improved_chat_fn(message, history, symbol_inp, timeframe_inp, exchange_inp, limit_inp, run_toy_adapt):
    """Улучшенная функция чата с валидацией и лучшими прогнозами"""
    img = None
    msg = (message or "").strip()
    advice = bool(run_toy_adapt)

    cmd, args = parse_command(msg)
    
    # Команда /price
    if cmd == "/price":
        symbol, timeframe, exchange, limit = args
        df, status = safe_analyzer.get_safe_data(symbol, timeframe, exchange, limit)
        
        if df is None:
            return f"❌ {status}", None, history
        
        try:
            # Конвертируем DataFrame в OHLCV формат для summary
            ohlcv_list = []
            for _, row in df.iterrows():
                timestamp_ms = int(row['timestamp'].timestamp() * 1000)
                ohlcv_list.append([timestamp_ms, row['open'], row['high'], row['low'], row['close'], row['volume']])
            
            summary = ohlcv_summary(ohlcv_list)
            reply = (
                f"📊 **{symbol} @ {exchange}**\n"
                f"Таймфрейм: {timeframe} | Свечей: {len(df)}\n\n"
                f"{summary}\n\n"
                f"💡 Попробуйте также:\n"
                f"`/chart` - свечной график\n"
                f"`/indicators` - технические индикаторы\n" 
                f"`/ensemble` - улучшенный ансамбль-прогноз\n"
                f"`/patterns` - поиск паттернов"
            )
            return reply, None, history
        except Exception as e:
            return f"❌ Ошибка при обработке данных: {e}", None, history

    # Команда /chart
    if cmd == "/chart":
        symbol, timeframe, exchange, limit = args
        df, status = safe_analyzer.get_safe_data(symbol, timeframe, exchange, limit)
        
        if df is None:
            return f"❌ {status}", None, history
            
        try:
            # Конвертируем DataFrame в OHLCV формат для графика
            ohlcv_list = []
            for _, row in df.iterrows():
                timestamp_ms = int(row['timestamp'].timestamp() * 1000)
                ohlcv_list.append([timestamp_ms, row['open'], row['high'], row['low'], row['close'], row['volume']])
            
            img = plot_candles(ohlcv_list, title=f"{symbol} {timeframe} @ {exchange}")
            summary = ohlcv_summary(ohlcv_list)
            reply = f"📈 График построен.\n\n{summary}"
            return reply, img, history
        except Exception as e:
            return f"❌ Ошибка при построении графика: {e}", None, history

    # Команда /indicators (улучшенная)
    if cmd == "/indicators":
        symbol, timeframe, exchange, limit = args
        df, status = safe_analyzer.get_safe_data(symbol, timeframe, exchange, limit)
        
        if df is None:
            return f"❌ {status}", None, history
            
        try:
            # Технический анализ
            tech_results = safe_analyzer.safe_technical_analysis(df)
            
            if "error" in tech_results:
                return f"❌ {tech_results['error']}", None, history
            
            # Построение графика с индикаторами
            img = plot_indicators_safe(df, title=f"{symbol} - Technical Analysis")
            
            # Создание отчета
            reply = f"📊 **Технический анализ {symbol}:**\n\n"
            reply += f"📈 **Данных:** {len(df)} свечей\n\n"
            
            # RSI анализ
            if "rsi" in tech_results and tech_results["rsi"]["data"] is not None:
                rsi_current = tech_results["rsi"]["data"].iloc[-1]
                if rsi_current > 70:
                    reply += f"🔴 **RSI:** {rsi_current:.2f} - Перекупленность\n"
                elif rsi_current < 30:
                    reply += f"🟢 **RSI:** {rsi_current:.2f} - Перепроданность\n"
                else:
                    reply += f"⚪ **RSI:** {rsi_current:.2f} - Нейтральная зона\n"
            else:
                reply += f"⚠️ **RSI:** {tech_results['rsi']['status']}\n"
            
            # MACD анализ
            if "macd" in tech_results and tech_results["macd"].get("macd_line") is not None:
                macd_line = tech_results["macd"]["macd_line"].iloc[-1]
                signal_line = tech_results["macd"]["signal_line"].iloc[-1]
                
                if macd_line > signal_line:
                    reply += f"🟢 **MACD:** выше сигнальной - бычий сигнал\n"
                else:
                    reply += f"🔴 **MACD:** ниже сигнальной - медвежий сигнал\n"
            else:
                reply += f"⚠️ **MACD:** {tech_results['macd']['status']}\n"
            
            # Bollinger Bands
            if "bollinger" in tech_results and tech_results["bollinger"].get("upper") is not None:
                current_price = df['close'].iloc[-1]
                upper = tech_results["bollinger"]["upper"].iloc[-1]
                lower = tech_results["bollinger"]["lower"].iloc[-1]
                
                bb_position = (current_price - lower) / (upper - lower) * 100
                
                if bb_position > 80:
                    reply += f"🔴 **BB:** Цена у верхней границы ({bb_position:.1f}%)\n"
                elif bb_position < 20:
                    reply += f"🟢 **BB:** Цена у нижней границы ({bb_position:.1f}%)\n"
                else:
                    reply += f"⚪ **BB:** Цена в средней зоне ({bb_position:.1f}%)\n"
            else:
                reply += f"⚠️ **BB:** {tech_results['bollinger']['status']}\n"
            
            reply += f"\n📊 График включает: RSI, MACD, Bollinger Bands, EMA"
            
            return reply, img, history
            
        except Exception as e:
            return f"❌ Ошибка при анализе индикаторов: {e}", None, history

    # Команда /predict (значительно улучшенная)
    if cmd == "/predict":
        symbol, timeframe, exchange, limit = args
        df, status = safe_analyzer.get_safe_data(symbol, timeframe, exchange, max(limit, 100))
        
        if df is None:
            return f"❌ {status}", None, history
            
        try:
            # Получаем улучшенный прогноз
            prediction_result = safe_analyzer.safe_prediction(df, future_steps=10)
            
            if "error" in prediction_result:
                return f"❌ Прогноз недоступен: {prediction_result['error']}", None, history
            
            predictions = prediction_result['prediction']
            confidence = prediction_result['confidence']
            methods_used = prediction_result['methods_used']
            lower_bound = prediction_result.get('lower_bound')
            upper_bound = prediction_result.get('upper_bound')
            
            # Построение графика с доверительными интервалами
            img = plot_prediction_advanced(df, predictions, lower_bound, upper_bound, 
                                         title=f"{symbol} - Advanced Prediction")
            
            # Анализ прогноза
            current_price = df['close'].iloc[-1]
            predicted_price = predictions[-1]
            change = (predicted_price - current_price) / current_price * 100
            
            reply = f"🔮 **Улучшенный прогноз для {symbol}:**\n\n"
            reply += f"**Методы:** {', '.join(methods_used)}\n"
            reply += f"**Уверенность:** {confidence:.1%}\n\n"
            
            reply += f"**Текущая цена:** {current_price:.4f}\n"
            reply += f"**Прогноз (10 шагов):** {predicted_price:.4f}\n"
            reply += f"**Ожидаемое изменение:** {change:+.2f}%\n"
            
            if lower_bound is not None and upper_bound is not None:
                reply += f"**Диапазон:** {lower_bound[-1]:.4f} - {upper_bound[-1]:.4f}\n\n"
            
            # Интерпретация
            if confidence > 0.7:
                conf_emoji = "🟢"
                conf_text = "Высокая"
            elif confidence > 0.4:
                conf_emoji = "🟡"
                conf_text = "Средняя"
            else:
                conf_emoji = "🔴"
                conf_text = "Низкая"
            
            reply += f"{conf_emoji} **Надежность:** {conf_text}\n\n"
            
            if abs(change) > 5:
                if change > 0:
                    reply += "📈 Модель предсказывает значительный рост"
                else:
                    reply += "📉 Модель предсказывает значительное снижение"
            else:
                reply += "➡️ Модель предсказывает боковое движение"
            
            # Предупреждение
            if confidence < 0.3:
                reply += "\n\n⚠️ **Внимание:** Низкая надежность прогноза из-за высокой волатильности"
            
            reply += "\n\n💡 **Совет:** Используйте ансамбль-прогноз (`/ensemble`) для максимальной точности!"
            
            return reply, img, history
            
        except Exception as e:
            return f"❌ Ошибка при прогнозировании: {e}", None, history

    # Остальные команды используют старую функцию
    return chat_fn(message, history, symbol_inp, timeframe_inp, exchange_inp, limit_inp, run_toy_adapt)

# -------------------------
# Генерация ответа модели
# -------------------------
def generate_reply(user_text: str, advice_prefix=True):
    prefix = few_shot_prefix(user_text, k=2) if advice_prefix else ""
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n{prefix}\n\n"
        f"Запрос пользователя: {user_text}\n\n"
        f"Дай полезный ответ по-русски."
    )
    out = generator(
        full_prompt,
        max_length=256,
        do_sample=True,
        top_p=0.92,
        temperature=0.7,
        num_return_sequences=1
    )[0]["generated_text"]
    return out

# -------------------------
# Разбор команд
# -------------------------
def parse_command(text: str):
    parts = text.strip().split()
    if not parts:
        return None, []
    cmd = parts[0].lower()
    args = parts[1:]
    
    if cmd in ["/price", "/chart", "/indicators", "/predict", "/patterns", "/ensemble"]:
        symbol = args[0] if len(args) > 0 else DEFAULT_SYMBOL
        timeframe = args[1] if len(args) > 1 else DEFAULT_TIMEFRAME
        exchange = args[2] if len(args) > 2 else DEFAULT_EXCHANGE
        try:
            limit = int(args[3]) if len(args) > 3 else DEFAULT_LIMIT
        except:
            limit = DEFAULT_LIMIT
        return cmd, [symbol, timeframe, exchange, limit]
    return None, []

# -------------------------
# Gradio handlers
# -------------------------
def chat_fn(message, history, symbol_inp, timeframe_inp, exchange_inp, limit_inp, run_toy_adapt):
    img = None
    msg = (message or "").strip()
    advice = bool(run_toy_adapt)

    cmd, args = parse_command(msg)
    
    # Команда /price
    if cmd == "/price":
        symbol, timeframe, exchange, limit = args
        try:
            ohlcv = fetch_ohlcv(symbol, timeframe, exchange, limit)
            summary = ohlcv_summary(ohlcv)
            reply = (
                f"📊 **{symbol} @ {exchange}**\n"
                f"Таймфрейм: {timeframe} | Свечей: {limit}\n\n"
                f"{summary}\n\n"
                f"💡 Попробуйте также:\n"
                f"`/chart` - свечной график\n"
                f"`/indicators` - технические индикаторы\n"
                f"`/predict` - прогноз LSTM\n"
                f"`/patterns` - поиск паттернов"
            )
            return reply, None, history
        except Exception as e:
            return f"❌ Ошибка при получении цены: {e}", None, history

    # Команда /chart
    if cmd == "/chart":
        symbol, timeframe, exchange, limit = args
        try:
            ohlcv = fetch_ohlcv(symbol, timeframe, exchange, limit)
            img = plot_candles(ohlcv, title=f"{symbol} {timeframe} @ {exchange}")
            summary = ohlcv_summary(ohlcv)
            reply = f"📈 График построен.\n\n{summary}"
            return reply, img, history
        except Exception as e:
            return f"❌ Ошибка при построении графика: {e}", None, history

    # Команда /indicators
    if cmd == "/indicators":
        symbol, timeframe, exchange, limit = args
        try:
            ohlcv = fetch_ohlcv(symbol, timeframe, exchange, limit)
            df = ohlcv_to_dataframe(ohlcv)
            
            # Используем безопасный анализ
            tech_analysis = safe_technical_analysis(df)
            img = plot_indicators(df, title=f"{symbol} - Technical Analysis")
            
            analysis = f"📊 **Анализ индикаторов {symbol}:**\n\n"
            analysis += f"📈 **Данных:** {len(df)} свечей\n\n"
            
            if "error" in tech_analysis:
                analysis += f"⚠️ {tech_analysis['error']}\n"
                analysis += "💡 Попробуйте увеличить количество свечей (limit) для полного анализа"
                return analysis, img, history
            
            # RSI анализ
            rsi_data = tech_analysis.get("rsi", {})
            if rsi_data.get("status") == "OK" and rsi_data["data"] is not None:
                rsi_current = rsi_data["data"].iloc[-1]
                if rsi_current > 70:
                    analysis += f"🔴 **RSI:** {rsi_current:.2f} - Перекупленность\n"
                elif rsi_current < 30:
                    analysis += f"🟢 **RSI:** {rsi_current:.2f} - Перепроданность\n"
                else:
                    analysis += f"⚪ **RSI:** {rsi_current:.2f} - Нейтральная зона\n"
            else:
                analysis += f"⚠️ **RSI:** {rsi_data.get('status', 'Недоступен')}\n"
            
            # MACD анализ
            macd_data = tech_analysis.get("macd", {})
            if macd_data.get("status") == "OK":
                macd_current = macd_data["macd_line"].iloc[-1]
                signal_current = macd_data["signal_line"].iloc[-1]
                if macd_current > signal_current:
                    analysis += f"🟢 **MACD:** {macd_current:.4f} выше сигнальной {signal_current:.4f} - бычий сигнал\n"
                else:
                    analysis += f"🔴 **MACD:** {macd_current:.4f} ниже сигнальной {signal_current:.4f} - медвежий сигнал\n"
            else:
                analysis += f"⚠️ **MACD:** {macd_data.get('status', 'Недоступен')}\n"
            
            # Bollinger Bands анализ
            bb_data = tech_analysis.get("bollinger", {})
            if bb_data.get("status") == "OK":
                current_price = df['close'].iloc[-1]
                upper = bb_data["upper"].iloc[-1]
                lower = bb_data["lower"].iloc[-1]
                middle = bb_data["middle"].iloc[-1]
                
                bb_position = (current_price - lower) / (upper - lower) * 100
                
                if bb_position > 80:
                    analysis += f"🔴 **BB:** Цена у верхней границы ({bb_position:.1f}%) - возможна коррекция\n"
                elif bb_position < 20:
                    analysis += f"🟢 **BB:** Цена у нижней границы ({bb_position:.1f}%) - возможен отскок\n"
                else:
                    analysis += f"⚪ **BB:** Цена в средней зоне ({bb_position:.1f}%)\n"
            else:
                analysis += f"⚠️ **Bollinger Bands:** {bb_data.get('status', 'Недоступен')}\n"
            
            # Общий вывод
            analysis += f"\n📋 **Статус индикаторов:**\n"
            working_indicators = sum(1 for ind in ["rsi", "macd", "bollinger"] 
                                   if tech_analysis.get(ind, {}).get("status") == "OK")
            analysis += f"✅ Работают: {working_indicators}/3 индикатора\n"
            
            if working_indicators == 3:
                analysis += "🎯 Полный технический анализ доступен"
            elif working_indicators >= 1:
                analysis += "⚡ Частичный анализ - увеличьте количество данных для лучшей точности"
            else:
                analysis += "📊 Базовый анализ - требуется больше исторических данных"
            
            return analysis, img, history
        except Exception as e:
            return f"❌ Ошибка при расчёте индикаторов: {e}", None, history

    # Команда /predict
    if cmd == "/predict":
        symbol, timeframe, exchange, limit = args
        try:
            ohlcv = fetch_ohlcv(symbol, timeframe, exchange, max(limit, 100))
            df = ohlcv_to_dataframe(ohlcv)
            
            # Используем улучшенную версию с обучением
            predictions, status = predict_lstm_improved(df, future_steps=10, epochs=40)
            
            img = plot_prediction(df, predictions, title=f"{symbol} - LSTM Prediction")
            
            reply = f"🔮 **Прогноз для {symbol}:**\n\n"
            reply += f"Метод: {status}\n"
            
            if predictions is not None and len(predictions) > 0:
                current_price = df['close'].iloc[-1]
                predicted_price = predictions[-1]
                change = (predicted_price - current_price) / current_price * 100
                
                reply += f"Текущая цена: {current_price:.4f}\n"
                reply += f"Прогноз (10 шагов): {predicted_price:.4f}\n"
                reply += f"Ожидаемое изменение: {change:+.2f}%\n\n"
                
                # Анализ тренда по всему прогнозу
                trend_change = (predictions[-1] - predictions[0]) / predictions[0] * 100 if len(predictions) > 1 else 0
                
                if "LSTM" in status:
                    if change > 5:
                        reply += "📈 LSTM предсказывает значительный рост"
                    elif change < -5:
                        reply += "📉 LSTM предсказывает значительное снижение"
                    else:
                        reply += "➡️ LSTM предсказывает боковое движение"
                else:
                    if change > 2:
                        reply += "📈 Линейный тренд указывает на рост"
                    elif change < -2:
                        reply += "📉 Линейный тренд указывает на снижение"
                    else:
                        reply += "➡️ Тренд нейтральный"
                
                if abs(trend_change) > 1:
                    reply += f"\n📊 Общий тренд прогноза: {trend_change:+.2f}%"
            else:
                reply += "⚠️ Прогноз недоступен (недостаточно данных)"
            
            if "LSTM" in status:
                reply += "\n\n🧠 **Модель обучена** на исторических данных"
            else:
                reply += "\n\n📈 **Fallback-метод** - LSTM недоступен или неточен"
            
            reply += "\n⚠️ **Внимание:** Не используйте для реальной торговли!"
            
            return reply, img, history
        except Exception as e:
            return f"❌ Ошибка при прогнозировании: {e}", None, history

    # Команда /patterns
    if cmd == "/patterns":
        symbol, timeframe, exchange, limit = args
        try:
            ohlcv = fetch_ohlcv(symbol, timeframe, exchange, limit)
            df = ohlcv_to_dataframe(ohlcv)
            
            # Используем улучшенную детекцию
            patterns = detect_patterns_improved(df)
            
            # Строим график с отметками паттернов
            img = plot_candles(ohlcv, title=f"{symbol} - Advanced Pattern Detection")
            
            reply = f"🔍 **Продвинутый анализ паттернов {symbol}:**\n\n"
            reply += f"📊 **Данных:** {len(df)} свечей\n\n"
            
            # Проверяем на недостаток данных
            if len(patterns) == 1 and "Недостаточно данных" in patterns[0]:
                reply += f"⚠️ {patterns[0]}\n"
                reply += "💡 Увеличьте количество свечей (limit) для полного анализа паттернов"
                return reply, img, history
            
            # Выводим найденные паттерны
            pattern_count = 0
            for pattern in patterns:
                if not ("тренд" in pattern.lower() or "других явных" in pattern.lower()):
                    pattern_count += 1
                reply += f"• {pattern}\n"
            
            # Добавляем интеллектуальные рекомендации
            reply += f"\n📋 **Анализ ({pattern_count} паттернов найдено):**\n"
            
            if any("двойная вершина" in p.lower() for p in patterns):
                reply += "🔴 **Медвежий сигнал** - двойная вершина указывает на возможный разворот вниз\n"
            elif any("двойное дно" in p.lower() for p in patterns):
                reply += "🟢 **Бычий сигнал** - двойное дно указывает на возможный разворот вверх\n"
            
            if any("голова и плечи" in p.lower() for p in patterns):
                reply += "🔴 **Сильный медвежий паттерн** - высокая вероятность снижения\n"
            
            if any("треугольник" in p.lower() for p in patterns):
                if "восходящий" in " ".join(patterns).lower():
                    reply += "📈 **Бычий треугольник** - ожидается прорыв вверх\n"
                elif "нисходящий" in " ".join(patterns).lower():
                    reply += "📉 **Медвежий треугольник** - ожидается прорыв вниз\n"
                else:
                    reply += "⚡ **Треугольник** - готовьтесь к сильному движению в любую сторону\n"
            
            if any("прорыв" in p.lower() for p in patterns):
                reply += "💥 **Прорыв с объёмом** - сильный сигнал для продолжения движения\n"
            
            if any("дивергенция" in p.lower() for p in patterns):
                if "медвежья" in " ".join(patterns).lower():
                    reply += "⚡ **Медвежья дивергенция** - ослабление бычьего тренда\n"
                elif "бычья" in " ".join(patterns).lower():
                    reply += "⚡ **Бычья дивергенция** - ослабление медвежьего тренда\n"
            
            # Общие рекомендации
            reply += f"\n💡 **Торговые рекомендации:**\n"
            
            if pattern_count == 0:
                reply += "📊 Явных паттернов не найдено - следите за общим трендом\n"
            elif pattern_count >= 3:
                reply += "⚠️ Множественные сигналы - дождитесь подтверждения\n"
            else:
                reply += "🎯 Умеренное количество сигналов - анализируйте в контексте общего тренда\n"
            
            reply += "📈 Всегда используйте стоп-лоссы и управление рисками"
            
            return reply, img, history
        except Exception as e:
            return f"❌ Ошибка при анализе паттернов: {e}", None, history

    # Команда /ensemble
    if cmd == "/ensemble":
        symbol, timeframe, exchange, limit = args
        try:
            ohlcv = fetch_ohlcv(symbol, timeframe, exchange, max(limit, 100))
            df = ohlcv_to_dataframe(ohlcv)
            
            # Используем адаптивный ансамбль
            ensemble_result = adaptive_ensemble_prediction(df, future_steps=10)
            
            # Строим график с ансамбль-прогнозом
            img = plot_ensemble_prediction(df, ensemble_result, title=f"{symbol} - Ensemble Forecast")
            
            reply = f"🎯 **Ансамбль-прогноз для {symbol}:**\n\n"
            reply += f"Метод: {ensemble_result['status']}\n"
            
            if ensemble_result['prediction'] is not None and len(ensemble_result['prediction']) > 0:
                current_price = df['close'].iloc[-1]
                predicted_price = ensemble_result['prediction'][-1]
                change = (predicted_price - current_price) / current_price * 100
                
                reply += f"Текущая цена: {current_price:.4f}\n"
                reply += f"Ансамбль-прогноз: {predicted_price:.4f}\n"
                reply += f"Ожидаемое изменение: {change:+.2f}%\n\n"
                
                # Доверительный интервал
                if 'lower_bound' in ensemble_result and 'upper_bound' in ensemble_result:
                    lower = ensemble_result['lower_bound'][-1]
                    upper = ensemble_result['upper_bound'][-1]
                    reply += f"📊 **Доверительный интервал:**\n"
                    reply += f"Нижняя граница: {lower:.4f} ({(lower-current_price)/current_price*100:+.2f}%)\n"
                    reply += f"Верхняя граница: {upper:.4f} ({(upper-current_price)/current_price*100:+.2f}%)\n\n"
                
                # Анализ по методам
                reply += f"🔬 **Методы в ансамбле:**\n"
                for method in ensemble_result['methods_used']:
                    reply += f"• {method}\n"
                
                # Точность методов (если доступна)
                if 'method_accuracies' in ensemble_result:
                    reply += f"\n📈 **Историческая точность методов:**\n"
                    for method, accuracy in ensemble_result['method_accuracies'].items():
                        reply += f"• {method}: {accuracy:.1f}%\n"
                
                # Общая уверенность
                confidence = ensemble_result.get('confidence', 0)
                reply += f"\n🎯 **Уверенность:** {confidence:.1%}\n"
                
                if confidence > 0.7:
                    reply += "🟢 Высокая уверенность - прогноз надёжный"
                elif confidence > 0.4:
                    reply += "🟡 Средняя уверенность - используйте с осторожностью"
                else:
                    reply += "🔴 Низкая уверенность - высокая неопределённость"
                
                # Рекомендации
                reply += f"\n\n💡 **Торговые рекомендации:**\n"
                if abs(change) > 5:
                    if change > 0:
                        reply += "📈 Сильный бычий сигнал - рассмотрите покупку\n"
                    else:
                        reply += "📉 Сильный медвежий сигнал - рассмотрите продажу\n"
                elif abs(change) > 2:
                    if change > 0:
                        reply += "📊 Умеренный рост - осторожная покупка\n"
                    else:
                        reply += "📊 Умеренное снижение - осторожная продажа\n"
                else:
                    reply += "➡️ Боковое движение - ожидайте подтверждения\n"
                
                reply += "⚠️ Используйте стоп-лоссы и управление рисками"
            else:
                reply += "⚠️ Прогноз недоступен"
            
            reply += f"\n\n🧠 **Ансамбль объединяет {len(ensemble_result.get('methods_used', []))} методов** для максимальной точности!"
            
            return reply, img, history
        except Exception as e:
            return f"❌ Ошибка при ансамбль-прогнозировании: {e}", None, history

    # Команда /help
    if msg.startswith("/help"):
        help_text = (
            "📚 **Доступные команды:**\n\n"
            "**Базовые:**\n"
            "`/price SYMBOL TIMEFRAME EXCHANGE LIMIT` - текущая цена и статистика\n"
            "`/chart SYMBOL TIMEFRAME EXCHANGE LIMIT` - свечной график\n\n"
            "**Аналитика:**\n"
            "`/indicators SYMBOL TIMEFRAME EXCHANGE LIMIT` - технические индикаторы (RSI, MACD, BB, EMA)\n"
            "`/patterns SYMBOL TIMEFRAME EXCHANGE LIMIT` - поиск паттернов\n"
            "`/predict SYMBOL TIMEFRAME EXCHANGE LIMIT` - LSTM прогноз цены\n"
            "`/ensemble SYMBOL TIMEFRAME EXCHANGE LIMIT` - ансамбль-прогноз (5 методов)\n\n"
            "**Примеры:**\n"
            "`/indicators BTC/USDT 1h binance 200`\n"
            "`/predict ETH/USDT 4h binance 300`\n"
            "`/patterns SOL/USDT 15m binance 100`\n\n"
            "**Параметры:**\n"
            "• SYMBOL: BTC/USDT, ETH/USDT, SOL/USDT и др.\n"
            "• TIMEFRAME: 1m, 5m, 15m, 30m, 1h, 4h, 1d\n"
            "• EXCHANGE: binance, bybit, okx, kucoin\n"
            "• LIMIT: количество свечей (50-500)"
        )
        return help_text, None, history

    # Команда /go - быстрый запуск с параметрами справа
    if msg.lower().startswith("/go"):
        try:
            symbol = symbol_inp or DEFAULT_SYMBOL
            timeframe = timeframe_inp or DEFAULT_TIMEFRAME
            exchange = exchange_inp or DEFAULT_EXCHANGE
            limit = int(limit_inp) if str(limit_inp).isdigit() else DEFAULT_LIMIT
            
            # Получаем все данные
            ohlcv = fetch_ohlcv(symbol, timeframe, exchange, limit)
            df = ohlcv_to_dataframe(ohlcv)
            
            # Создаём комплексный график
            img = plot_indicators(df, title=f"{symbol} - Full Analysis")
            
            # Анализ
            patterns = detect_patterns(df)
            rsi = calculate_rsi(df).iloc[-1]
            
            reply = f"🎯 **Комплексный анализ {symbol}:**\n\n"
            reply += ohlcv_summary(ohlcv) + "\n\n"
            reply += f"**Индикаторы:**\n"
            reply += f"• RSI: {rsi:.2f}\n\n"
            reply += f"**Паттерны:**\n"
            for p in patterns[:3]:  # Первые 3 паттерна
                reply += f"• {p}\n"
            
            return reply, img, history
        except Exception as e:
            return f"❌ Ошибка при выполнении /go: {e}", None, history

    # Обычный генеративный ответ
    reply = generate_reply(msg, advice_prefix=advice)
    return reply, None, history

# -------------------------
# Построение интерфейса Gradio
# -------------------------
with gr.Blocks(title="RU Crypto Bot Pro - AI Trading Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 RU Crypto Bot Pro
    ### AI-powered криптовалютный ассистент с техническим анализом
    
    **Возможности:**
    - 📊 Технические индикаторы (RSI, MACD, Bollinger Bands, EMA)
    - 🔮 LSTM прогнозирование цен
    - 🔍 Детекция паттернов (двойная вершина/дно, треугольники)
    - 📈 Множественные типы графиков
    - 💬 AI-чат на русском языке
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=500, 
                label="💬 Диалог с ассистентом",
                bubble_full_width=False
            )
            msg = gr.Textbox(
                placeholder="Введите команду или вопрос... Попробуйте /help для списка команд", 
                label="Ваше сообщение",
                lines=2
            )
            out_image = gr.Image(
                type="filepath", 
                label="📊 График", 
                height=400
            )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Очистить чат", scale=1)
                help_btn = gr.Button("❓ Помощь", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Быстрые настройки")
            
            with gr.Group():
                symbol_inp = gr.Textbox(
                    value=DEFAULT_SYMBOL, 
                    label="Торговая пара",
                    info="BTC/USDT, ETH/USDT, SOL/USDT"
                )
                timeframe_inp = gr.Dropdown(
                    choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
                    value=DEFAULT_TIMEFRAME,
                    label="Таймфрейм"
                )
                exchange_inp = gr.Dropdown(
                    choices=["binance", "bybit", "okx", "kucoin", "kraken"],
                    value=DEFAULT_EXCHANGE,
                    label="Биржа"
                )
                limit_inp = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=DEFAULT_LIMIT,
                    step=50,
                    label="Количество свечей"
                )
            
            run_toy_adapt = gr.Checkbox(
                value=True, 
                label="🧠 AI-подсказки (few-shot)",
                info="Улучшает качество ответов"
            )
            
            gr.Markdown("### 🎯 Быстрые команды")
            with gr.Row():
                chart_btn = gr.Button("📈 График", scale=1)
                indicators_btn = gr.Button("📊 Индикаторы", scale=1)
            with gr.Row():
                predict_btn = gr.Button("🔮 Прогноз", scale=1)
                patterns_btn = gr.Button("🔍 Паттерны", scale=1)
            with gr.Row():
                ensemble_btn = gr.Button("🎯 Ансамбль", scale=2)
            
            go_btn = gr.Button("🚀 Полный анализ", variant="primary", size="lg")
            
            gr.Markdown("""
            ### 📌 Подсказки:
            - Используйте `/help` для списка команд
            - Кнопка "Полный анализ" запускает комплексный анализ
            - LSTM работает в демо-режиме
            """)

    # Состояние чата
    state = gr.State([])

    # Обработчики
    def respond(user_message, chat_history, symbol_v, timeframe_v, exchange_v, limit_v, adapt_flag):
        if not user_message:
            return "", None, chat_history
        reply, img_b64, _ = chat_fn(user_message, chat_history, symbol_v, timeframe_v, exchange_v, limit_v, adapt_flag)
        chat_history = chat_history + [[user_message, reply]]
        
        if img_b64 is not None and img_b64.startswith("data:image/png;base64,"):
            img_data = base64.b64decode(img_b64.split(",")[1])
            path = f"chart_{int(time.time())}.png"
            with open(path, "wb") as f:
                f.write(img_data)
            return "", path, chat_history
        return "", None, chat_history

    def quick_command(cmd, chat_history, symbol_v, timeframe_v, exchange_v, limit_v, adapt_flag):
        command = f"{cmd} {symbol_v} {timeframe_v} {exchange_v} {limit_v}"
        return respond(command, chat_history, symbol_v, timeframe_v, exchange_v, limit_v, adapt_flag)

    # Привязка событий
    msg.submit(
        respond,
        inputs=[msg, chatbot, symbol_inp, timeframe_inp, exchange_inp, limit_inp, run_toy_adapt],
        outputs=[msg, out_image, chatbot]
    )
    
    go_btn.click(
        lambda h, s, t, e, l, a: respond("/go", h, s, t, e, l, a),
        inputs=[chatbot, symbol_inp, timeframe_inp, exchange_inp, limit_inp, run_toy_adapt],
        outputs=[msg, out_image, chatbot]
    )
    
    chart_btn.click(
        lambda h, s, t, e, l, a: quick_command("/chart", h, s, t, e, l, a),
        inputs=[chatbot, symbol_inp, timeframe_inp, exchange_inp, limit_inp, run_toy_adapt],
        outputs=[msg, out_image, chatbot]
    )
    
    indicators_btn.click(
        lambda h, s, t, e, l, a: quick_command("/indicators", h, s, t, e, l, a),
        inputs=[chatbot, symbol_inp, timeframe_inp, exchange_inp, limit_inp, run_toy_adapt],
        outputs=[msg, out_image, chatbot]
    )
    
    predict_btn.click(
        lambda h, s, t, e, l, a: quick_command("/predict", h, s, t, e, l, a),
        inputs=[chatbot, symbol_inp, timeframe_inp, exchange_inp, limit_inp, run_toy_adapt],
        outputs=[msg, out_image, chatbot]
    )
    
    patterns_btn.click(
        lambda h, s, t, e, l, a: quick_command("/patterns", h, s, t, e, l, a),
        inputs=[chatbot, symbol_inp, timeframe_inp, exchange_inp, limit_inp, run_toy_adapt],
        outputs=[msg, out_image, chatbot]
    )
    
    ensemble_btn.click(
        lambda h, s, t, e, l, a: quick_command("/ensemble", h, s, t, e, l, a),
        inputs=[chatbot, symbol_inp, timeframe_inp, exchange_inp, limit_inp, run_toy_adapt],
        outputs=[msg, out_image, chatbot]
    )
    
    help_btn.click(
        lambda h, a: respond("/help", h, "", "", "", "", a),
        inputs=[chatbot, run_toy_adapt],
        outputs=[msg, out_image, chatbot]
    )
    
    clear_btn.click(
        lambda: ([], "", None), 
        inputs=None, 
        outputs=[chatbot, msg, out_image]
    )

    gr.Markdown("""
    ---
    ⚠️ **Дисклеймер:** Это демонстрационный бот. Не используйте для реальной торговли!
    Всегда проводите собственный анализ перед принятием торговых решений.
    """)

if __name__ == "__main__":
    # Публичная ссылка через gradio-туннель + явные параметры сервера.
    # Подойдёт в любой среде, даже если localhost блокируется прокси/VPN.
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )

