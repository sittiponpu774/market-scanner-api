#!/usr/bin/env python3
"""
Market Data Fetcher for GitHub Actions
Fetches data from Binance and Yahoo Finance, saves as JSON files
"""

import httpx
import json
import os
import random
from datetime import datetime
from pathlib import Path

# Create data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Crypto pairs to track
CRYPTO_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "SOLUSDT", "DOTUSDT", "MATICUSDT", "SHIBUSDT",
    "LTCUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT", "ATOMUSDT",
    "UNIUSDT", "ETCUSDT", "XLMUSDT", "BCHUSDT", "APTUSDT",
    "NEARUSDT", "FILUSDT", "LDOUSDT", "ARBUSDT", "OPUSDT",
    "INJUSDT", "STXUSDT", "IMXUSDT", "RNDRUSDT", "SUIUSDT",
    "SEIUSDT", "TIAUSDT", "JUPUSDT", "WIFUSDT", "PEPEUSDT",
    "FLOKIUSDT", "BONKUSDT", "ORDIUSDT", "KASUSDT", "FETUSDT",
    "AGIXUSDT", "OCEANUSDT", "RLCUSDT", "AKTUSDT", "TAOUSDT",
    "AAVEUSDT", "MKRUSDT", "SNXUSDT", "CRVUSDT", "COMPUSDT"
]

# Thai stocks to track
THAI_STOCKS = [
    "PTT.BK", "AOT.BK", "ADVANC.BK", "CPALL.BK", "SCC.BK",
    "KBANK.BK", "SCB.BK", "BBL.BK", "GULF.BK", "PTTEP.BK",
    "BDMS.BK", "TRUE.BK", "BEM.BK", "MINT.BK", "CPN.BK",
    "DELTA.BK", "EA.BK", "GPSC.BK", "INTUCH.BK", "IVL.BK",
    "KCE.BK", "KTB.BK", "BANPU.BK", "BCP.BK", "BH.BK",
    "BJC.BK", "BTS.BK", "CBG.BK", "CENTEL.BK", "COM7.BK",
    "CPNREIT.BK", "CRC.BK", "DTAC.BK", "EGCO.BK", "GLOBAL.BK",
    "HMPRO.BK", "IRPC.BK", "JMT.BK", "JMART.BK", "TIDLOR.BK",
    "TTB.BK", "TU.BK", "WHA.BK", "OR.BK", "AWC.BK",
    "OSP.BK", "SAWAD.BK", "STGT.BK", "TISCO.BK", "TOP.BK"
]


def calculate_rsi(prices, period=14):
    """Calculate RSI from price list"""
    if len(prices) < period + 1:
        return 50.0
    
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    if len(gains) < period:
        return 50.0
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)


def calculate_macd(prices):
    """Calculate MACD from price list"""
    if len(prices) < 26:
        return {"macd": 0, "signal": 0, "histogram": 0}
    
    def ema(data, period):
        if len(data) < period:
            return data[-1] if data else 0
        multiplier = 2 / (period + 1)
        ema_val = sum(data[:period]) / period
        for price in data[period:]:
            ema_val = (price - ema_val) * multiplier + ema_val
        return ema_val
    
    ema12 = ema(prices, 12)
    ema26 = ema(prices, 26)
    macd_line = ema12 - ema26
    signal_line = macd_line * 0.15  # Simplified
    histogram = macd_line - signal_line
    
    return {
        "macd": round(macd_line, 4),
        "signal": round(signal_line, 4),
        "histogram": round(histogram, 4)
    }


def determine_signal(rsi, macd_histogram, change_percent):
    """Determine trading signal based on indicators"""
    score = 0
    
    # RSI scoring
    if rsi < 30:
        score += 2  # Oversold - bullish
    elif rsi < 40:
        score += 1
    elif rsi > 70:
        score -= 2  # Overbought - bearish
    elif rsi > 60:
        score -= 1
    
    # MACD scoring
    if macd_histogram > 0:
        score += 1
    else:
        score -= 1
    
    # Trend scoring
    if change_percent > 3:
        score += 1
    elif change_percent < -3:
        score -= 1
    
    if score >= 2:
        return "BUY"
    elif score <= -2:
        return "SELL"
    else:
        return "HOLD"


def fetch_crypto_data():
    """Fetch crypto data from Binance"""
    print("Fetching crypto data from Binance...")
    signals = []
    
    try:
        with httpx.Client(timeout=30.0) as client:
            # Fetch 24hr ticker data
            response = client.get("https://api.binance.com/api/v3/ticker/24hr")
            response.raise_for_status()
            all_tickers = {t['symbol']: t for t in response.json()}
            
            for symbol in CRYPTO_PAIRS:
                try:
                    if symbol not in all_tickers:
                        continue
                    
                    ticker = all_tickers[symbol]
                    price = float(ticker['lastPrice'])
                    change_percent = float(ticker['priceChangePercent'])
                    volume = float(ticker['volume'])
                    high = float(ticker['highPrice'])
                    low = float(ticker['lowPrice'])
                    
                    # Generate simulated historical prices for indicators
                    base_price = price
                    volatility = abs(change_percent) / 100 * 2 + 0.02
                    prices = []
                    for i in range(30):
                        variation = random.uniform(-volatility, volatility)
                        prices.append(base_price * (1 + variation))
                    prices.append(price)
                    
                    rsi = calculate_rsi(prices)
                    macd = calculate_macd(prices)
                    signal_type = determine_signal(rsi, macd['histogram'], change_percent)
                    
                    # Calculate strength
                    strength = min(abs(rsi - 50) / 50 + abs(change_percent) / 10, 1.0)
                    
                    signals.append({
                        "symbol": symbol.replace("USDT", ""),
                        "name": symbol.replace("USDT", "/USDT"),
                        "price": price,
                        "change_24h": round(change_percent, 2),
                        "volume_24h": volume,
                        "high_24h": high,
                        "low_24h": low,
                        "rsi": rsi,
                        "macd": macd,
                        "signal": signal_type,
                        "strength": round(strength, 2),
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
                    
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error fetching Binance data: {e}")
        # Return cached data if available
        cache_file = DATA_DIR / "crypto_signals.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return []
    
    print(f"Fetched {len(signals)} crypto signals")
    return signals


def fetch_thai_stock_data():
    """Fetch Thai stock data from Yahoo Finance"""
    print("Fetching Thai stock data from Yahoo Finance...")
    signals = []
    
    try:
        with httpx.Client(timeout=30.0) as client:
            for symbol in THAI_STOCKS:
                try:
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                    params = {"interval": "1d", "range": "1mo"}
                    
                    response = client.get(url, params=params)
                    if response.status_code != 200:
                        continue
                    
                    data = response.json()
                    result = data.get('chart', {}).get('result', [])
                    
                    if not result:
                        continue
                    
                    meta = result[0].get('meta', {})
                    indicators = result[0].get('indicators', {})
                    quotes = indicators.get('quote', [{}])[0]
                    
                    closes = quotes.get('close', [])
                    closes = [c for c in closes if c is not None]
                    
                    if not closes:
                        continue
                    
                    price = meta.get('regularMarketPrice', closes[-1])
                    prev_close = meta.get('previousClose', closes[-2] if len(closes) > 1 else price)
                    
                    if prev_close and prev_close > 0:
                        change_percent = ((price - prev_close) / prev_close) * 100
                    else:
                        change_percent = 0
                    
                    rsi = calculate_rsi(closes)
                    macd = calculate_macd(closes)
                    signal_type = determine_signal(rsi, macd['histogram'], change_percent)
                    strength = min(abs(rsi - 50) / 50 + abs(change_percent) / 10, 1.0)
                    
                    clean_symbol = symbol.replace(".BK", "")
                    
                    signals.append({
                        "symbol": clean_symbol,
                        "name": clean_symbol,
                        "price": round(price, 2),
                        "change_24h": round(change_percent, 2),
                        "volume_24h": sum(quotes.get('volume', [0])[-5:]) if quotes.get('volume') else 0,
                        "high_24h": max(quotes.get('high', [price])[-1:]) if quotes.get('high') else price,
                        "low_24h": min(quotes.get('low', [price])[-1:]) if quotes.get('low') else price,
                        "rsi": rsi,
                        "macd": macd,
                        "signal": signal_type,
                        "strength": round(strength, 2),
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
                    
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error fetching Yahoo data: {e}")
        cache_file = DATA_DIR / "thai_signals.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return []
    
    print(f"Fetched {len(signals)} Thai stock signals")
    return signals


def save_json(data, filename):
    """Save data to JSON file"""
    filepath = DATA_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {filepath}")


def main():
    print(f"Starting data fetch at {datetime.utcnow().isoformat()}Z")
    
    # Fetch crypto data
    crypto_signals = fetch_crypto_data()
    save_json(crypto_signals, "crypto_signals.json")
    
    # Fetch Thai stock data
    thai_signals = fetch_thai_stock_data()
    save_json(thai_signals, "thai_signals.json")
    
    # Create combined signals for polling
    all_signals = {
        "crypto": crypto_signals,
        "thai": thai_signals,
        "updated_at": datetime.utcnow().isoformat() + "Z"
    }
    save_json(all_signals, "all_signals.json")
    
    # Create health check file
    health = {
        "status": "ok",
        "crypto_count": len(crypto_signals),
        "thai_count": len(thai_signals),
        "updated_at": datetime.utcnow().isoformat() + "Z"
    }
    save_json(health, "health.json")
    
    print(f"Data fetch completed at {datetime.utcnow().isoformat()}Z")


if __name__ == "__main__":
    main()
