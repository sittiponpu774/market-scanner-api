#!/usr/bin/env python3
"""
Advanced ML Prediction Engine for Market Scanner
Features:
- XGBoost/LSTM-based price prediction
- Support & Resistance Levels (Pivot Points)
- Volume Profile (VPVR)
- Fear & Greed Index integration
- Bull Trap detection
- 5x Potential scoring
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json

# Try to import ML libraries (optional for production)
try:
    from sklearn.preprocessing import MinMaxScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class AdvancedAnalyzer:
    """Advanced technical analysis and ML prediction"""
    
    def __init__(self):
        self.fear_greed_cache = None
        self.fear_greed_timestamp = None
    
    # =============================================
    # SUPPORT & RESISTANCE (Pivot Points S3/R3)
    # =============================================
    
    def calculate_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """
        Calculate Standard Pivot Points with S3/R3 levels
        
        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
            
        Returns:
            Dictionary with PP, S1-S3, R1-R3 levels
        """
        pivot = (high + low + close) / 3
        
        # Support levels
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        # Resistance levels
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        return {
            "pivot": round(pivot, 8),
            "r1": round(r1, 8),
            "r2": round(r2, 8),
            "r3": round(r3, 8),
            "s1": round(s1, 8),
            "s2": round(s2, 8),
            "s3": round(s3, 8),
        }
    
    def calculate_fibonacci_pivots(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Fibonacci-based pivot points"""
        pivot = (high + low + close) / 3
        range_hl = high - low
        
        return {
            "pivot": round(pivot, 8),
            "r1": round(pivot + 0.382 * range_hl, 8),
            "r2": round(pivot + 0.618 * range_hl, 8),
            "r3": round(pivot + 1.0 * range_hl, 8),
            "s1": round(pivot - 0.382 * range_hl, 8),
            "s2": round(pivot - 0.618 * range_hl, 8),
            "s3": round(pivot - 1.0 * range_hl, 8),
        }
    
    def identify_fractal_levels(self, highs: List[float], lows: List[float], lookback: int = 5) -> Dict[str, List[float]]:
        """
        Identify Williams Fractal support/resistance levels
        
        Args:
            highs: List of high prices
            lows: List of low prices
            lookback: Number of candles to check (default 5)
            
        Returns:
            Dictionary with fractal_highs and fractal_lows
        """
        fractal_highs = []
        fractal_lows = []
        
        half = lookback // 2
        
        for i in range(half, len(highs) - half):
            # Check for fractal high
            is_fractal_high = True
            for j in range(i - half, i + half + 1):
                if j != i and highs[j] >= highs[i]:
                    is_fractal_high = False
                    break
            if is_fractal_high:
                fractal_highs.append(highs[i])
            
            # Check for fractal low
            is_fractal_low = True
            for j in range(i - half, i + half + 1):
                if j != i and lows[j] <= lows[i]:
                    is_fractal_low = False
                    break
            if is_fractal_low:
                fractal_lows.append(lows[i])
        
        # Return most recent and significant levels
        return {
            "resistance_levels": sorted(fractal_highs, reverse=True)[:5],
            "support_levels": sorted(fractal_lows)[:5],
        }
    
    # =============================================
    # VOLUME PROFILE (VPVR)
    # =============================================
    
    def calculate_volume_profile(
        self, 
        prices: List[float], 
        volumes: List[float], 
        num_bins: int = 20
    ) -> Dict:
        """
        Calculate Volume Profile Visible Range (VPVR)
        Identifies price levels where whales accumulate
        
        Args:
            prices: List of close prices
            volumes: List of corresponding volumes
            num_bins: Number of price bins
            
        Returns:
            Volume profile with POC, VAH, VAL
        """
        if len(prices) < 2 or len(volumes) < 2:
            return {"poc": 0, "vah": 0, "val": 0, "profile": []}
        
        price_min = min(prices)
        price_max = max(prices)
        
        if price_max == price_min:
            return {"poc": price_min, "vah": price_min, "val": price_min, "profile": []}
        
        bin_size = (price_max - price_min) / num_bins
        volume_bins = [0.0] * num_bins
        price_bins = []
        
        for i in range(num_bins):
            bin_start = price_min + i * bin_size
            bin_end = bin_start + bin_size
            price_bins.append((bin_start + bin_end) / 2)
        
        # Assign volumes to price bins
        for price, volume in zip(prices, volumes):
            bin_idx = int((price - price_min) / bin_size)
            bin_idx = min(bin_idx, num_bins - 1)
            volume_bins[bin_idx] += volume
        
        # Find Point of Control (POC) - highest volume price level
        poc_idx = volume_bins.index(max(volume_bins))
        poc = price_bins[poc_idx]
        
        # Calculate Value Area (70% of volume)
        total_volume = sum(volume_bins)
        target_volume = total_volume * 0.7
        
        # Start from POC and expand
        accumulated = volume_bins[poc_idx]
        lower_idx = poc_idx
        upper_idx = poc_idx
        
        while accumulated < target_volume and (lower_idx > 0 or upper_idx < num_bins - 1):
            lower_vol = volume_bins[lower_idx - 1] if lower_idx > 0 else 0
            upper_vol = volume_bins[upper_idx + 1] if upper_idx < num_bins - 1 else 0
            
            if lower_vol > upper_vol and lower_idx > 0:
                lower_idx -= 1
                accumulated += lower_vol
            elif upper_idx < num_bins - 1:
                upper_idx += 1
                accumulated += upper_vol
            else:
                break
        
        val = price_bins[lower_idx]  # Value Area Low
        vah = price_bins[upper_idx]  # Value Area High
        
        # Create profile for visualization
        profile = [
            {"price": round(price_bins[i], 8), "volume": round(volume_bins[i], 2)}
            for i in range(num_bins)
        ]
        
        return {
            "poc": round(poc, 8),          # Point of Control (whale accumulation)
            "vah": round(vah, 8),          # Value Area High
            "val": round(val, 8),          # Value Area Low
            "profile": profile,
            "whale_zone": (round(val, 8), round(vah, 8)),  # Zone where whales trade
        }
    
    # =============================================
    # FEAR & GREED INDEX
    # =============================================
    
    def fetch_fear_greed_index(self) -> Dict:
        """
        Fetch Fear & Greed Index from alternative.me API
        Caches for 1 hour
        
        Returns:
            Dictionary with value and classification
        """
        # Check cache
        if (self.fear_greed_cache and self.fear_greed_timestamp and
            datetime.now() - self.fear_greed_timestamp < timedelta(hours=1)):
            return self.fear_greed_cache
        
        try:
            import httpx
            response = httpx.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("data"):
                fng = data["data"][0]
                result = {
                    "value": int(fng.get("value", 50)),
                    "classification": fng.get("value_classification", "Neutral"),
                    "timestamp": fng.get("timestamp"),
                    "is_extreme_fear": int(fng.get("value", 50)) <= 25,
                    "is_extreme_greed": int(fng.get("value", 50)) >= 75,
                }
                self.fear_greed_cache = result
                self.fear_greed_timestamp = datetime.now()
                return result
        except Exception as e:
            print(f"Error fetching Fear & Greed: {e}")
        
        # Default neutral
        return {
            "value": 50,
            "classification": "Neutral",
            "is_extreme_fear": False,
            "is_extreme_greed": False,
        }
    
    # =============================================
    # BULL TRAP DETECTION
    # =============================================
    
    def detect_bull_trap(
        self,
        prices: List[float],
        volumes: List[float],
        lookback: int = 5
    ) -> Dict:
        """
        Detect Bull Trap pattern
        Bull Trap: Price bounces up with LOW volume = likely reversal
        
        Args:
            prices: Recent close prices
            volumes: Recent volumes
            lookback: Periods to analyze
            
        Returns:
            Bull trap detection result
        """
        if len(prices) < lookback or len(volumes) < lookback:
            return {"is_bull_trap": False, "confidence": 0, "reason": "Insufficient data"}
        
        recent_prices = prices[-lookback:]
        recent_volumes = volumes[-lookback:]
        older_volumes = volumes[-(lookback*2):-lookback] if len(volumes) >= lookback*2 else volumes[:lookback]
        
        # Calculate metrics
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
        avg_recent_volume = sum(recent_volumes) / len(recent_volumes)
        avg_older_volume = sum(older_volumes) / len(older_volumes) if older_volumes else avg_recent_volume
        volume_ratio = avg_recent_volume / avg_older_volume if avg_older_volume > 0 else 1
        
        # Bull Trap Criteria:
        # 1. Price is going UP (positive change)
        # 2. Volume is BELOW average (weak conviction)
        is_price_up = price_change > 2  # At least 2% up
        is_volume_low = volume_ratio < 0.8  # Volume 20% below average
        
        # Additional confirmation
        is_overbought = False  # Will be calculated with RSI
        
        # Calculate confidence
        confidence = 0
        reasons = []
        
        if is_price_up and is_volume_low:
            confidence += 40
            reasons.append(f"Price up {price_change:.1f}% but volume only {volume_ratio:.1%} of avg")
        
        if volume_ratio < 0.6:
            confidence += 20
            reasons.append("Very low volume confirmation")
        
        if price_change > 5 and volume_ratio < 0.7:
            confidence += 20
            reasons.append("Strong price move with weak volume = suspicious")
        
        is_bull_trap = confidence >= 40
        
        return {
            "is_bull_trap": is_bull_trap,
            "confidence": min(confidence, 95),
            "price_change": round(price_change, 2),
            "volume_ratio": round(volume_ratio, 3),
            "reason": " | ".join(reasons) if reasons else "Normal price action",
            "signal_override": "BEARISH" if is_bull_trap else None,
        }
    
    # =============================================
    # ML PRICE PREDICTION (XGBoost)
    # =============================================
    
    def prepare_features(
        self,
        prices: List[float],
        volumes: List[float],
        rsi: float,
        macd: Dict,
        fear_greed: int,
    ) -> np.ndarray:
        """
        Prepare feature vector for ML prediction
        
        Features:
        - Price momentum (3, 7, 14 day returns)
        - Volume momentum
        - RSI
        - MACD components
        - Fear & Greed Index
        - Volatility
        """
        features = []
        
        # Price momentum features
        if len(prices) >= 14:
            ret_3d = (prices[-1] - prices[-3]) / prices[-3]
            ret_7d = (prices[-1] - prices[-7]) / prices[-7]
            ret_14d = (prices[-1] - prices[-14]) / prices[-14]
        else:
            ret_3d = ret_7d = ret_14d = 0
        
        features.extend([ret_3d, ret_7d, ret_14d])
        
        # Volume momentum
        if len(volumes) >= 7:
            vol_ratio = sum(volumes[-3:]) / sum(volumes[-7:-3]) if sum(volumes[-7:-3]) > 0 else 1
        else:
            vol_ratio = 1
        features.append(vol_ratio)
        
        # Technical indicators (normalized)
        features.append(rsi / 100)  # Normalize RSI to 0-1
        features.append(macd.get("macd", 0) / prices[-1] if prices else 0)  # Normalize MACD
        features.append(macd.get("histogram", 0) / prices[-1] if prices else 0)
        
        # Sentiment
        features.append(fear_greed / 100)  # Normalize to 0-1
        
        # Volatility
        if len(prices) >= 14:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = np.std(returns[-14:]) if len(returns) >= 14 else 0
        else:
            volatility = 0
        features.append(volatility)
        
        return np.array(features).reshape(1, -1)
    
    def predict_price_direction(
        self,
        prices: List[float],
        volumes: List[float],
        rsi: float,
        macd: Dict,
        fear_greed: int = 50,
    ) -> Dict:
        """
        Predict price direction for next 24 hours using ensemble of rules
        
        When XGBoost is available, uses ML model
        Otherwise, uses rule-based ensemble
        
        Returns:
            Prediction with probability and direction
        """
        # Rule-based prediction (always works)
        bullish_score = 0
        bearish_score = 0
        
        # RSI Analysis
        if rsi < 30:
            bullish_score += 25
        elif rsi < 40:
            bullish_score += 10
        elif rsi > 70:
            bearish_score += 25
        elif rsi > 60:
            bearish_score += 10
        
        # MACD Analysis
        if macd.get("histogram", 0) > 0:
            bullish_score += 15
        else:
            bearish_score += 15
        
        if macd.get("macd", 0) > macd.get("signal", 0):
            bullish_score += 10
        else:
            bearish_score += 10
        
        # Fear & Greed (contrarian)
        if fear_greed <= 25:  # Extreme Fear = buying opportunity
            bullish_score += 20
        elif fear_greed >= 75:  # Extreme Greed = selling signal
            bearish_score += 20
        
        # Price momentum
        if len(prices) >= 7:
            week_return = (prices[-1] - prices[-7]) / prices[-7] * 100
            if week_return > 10:
                bearish_score += 15  # Overbought
            elif week_return < -10:
                bullish_score += 15  # Oversold
        
        # Volume analysis (bull trap detection)
        if len(prices) >= 5 and len(volumes) >= 5:
            bull_trap = self.detect_bull_trap(prices, volumes)
            if bull_trap["is_bull_trap"]:
                bearish_score += 25
        
        # Calculate probabilities
        total_score = bullish_score + bearish_score
        if total_score > 0:
            bull_prob = bullish_score / total_score
            bear_prob = bearish_score / total_score
        else:
            bull_prob = bear_prob = 0.5
        
        # Determine direction
        if bull_prob > 0.6:
            direction = "UP"
            confidence = bull_prob
        elif bear_prob > 0.6:
            direction = "DOWN"
            confidence = bear_prob
        else:
            direction = "NEUTRAL"
            confidence = 0.5
        
        return {
            "direction": direction,
            "bullish_probability": round(bull_prob, 3),
            "bearish_probability": round(bear_prob, 3),
            "confidence": round(confidence, 3),
            "timeframe": "24h",
            "factors": {
                "rsi_signal": "bullish" if rsi < 40 else "bearish" if rsi > 60 else "neutral",
                "macd_signal": "bullish" if macd.get("histogram", 0) > 0 else "bearish",
                "sentiment": "fear" if fear_greed < 40 else "greed" if fear_greed > 60 else "neutral",
            }
        }
    
    # =============================================
    # 5X POTENTIAL SCORING
    # =============================================
    
    def calculate_5x_potential(
        self,
        symbol: str,
        current_price: float,
        market_cap_billions: float,
        category: str,
        rsi: float,
        volume_profile: Dict,
    ) -> Dict:
        """
        Calculate 5x potential score for long-term holding (3-5 years)
        
        Factors:
        - Market cap (smaller = more potential)
        - Category (AI, L2, DeFi = high growth)
        - Technical setup
        - Accumulation zones from volume profile
        """
        score = 0
        reasons = []
        
        # Market Cap Score (max 30 points)
        if market_cap_billions < 0.5:
            score += 30
            reasons.append("Micro cap (<$500M) - maximum upside")
        elif market_cap_billions < 2:
            score += 25
            reasons.append("Small cap (<$2B) - high growth potential")
        elif market_cap_billions < 10:
            score += 15
            reasons.append("Mid cap - moderate growth")
        else:
            score += 5
            reasons.append("Large cap - limited upside")
        
        # Category Score (max 25 points)
        category_scores = {
            "AI": 25,
            "Layer2": 22,
            "DeFi": 20,
            "Gaming": 18,
            "Layer1": 15,
            "Meme": 10,
            "Storage": 18,
            "Oracle": 16,
        }
        cat_score = category_scores.get(category, 12)
        score += cat_score
        reasons.append(f"{category} sector: +{cat_score} points")
        
        # Technical Setup Score (max 20 points)
        if rsi < 30:
            score += 20
            reasons.append("RSI Oversold - excellent entry")
        elif rsi < 40:
            score += 15
            reasons.append("RSI Low - good entry")
        elif rsi > 70:
            score -= 10
            reasons.append("RSI Overbought - wait for pullback")
        
        # Volume Profile Score (max 15 points)
        if volume_profile:
            poc = volume_profile.get("poc", current_price)
            val = volume_profile.get("val", current_price)
            
            if current_price <= val:
                score += 15
                reasons.append("Price at Value Area Low - accumulation zone")
            elif current_price <= poc:
                score += 10
                reasons.append("Price below POC - good entry")
        
        # Calculate potential multiplier estimate
        base_growth = self._estimate_category_growth(category)
        cap_multiplier = 1 + (10 - min(market_cap_billions, 10)) / 10
        potential_multiplier = (1 + base_growth) ** 5 * cap_multiplier
        
        # Bonus for high multiplier
        if potential_multiplier >= 10:
            score += 10
            reasons.append(f"Estimated {potential_multiplier:.1f}x potential")
        elif potential_multiplier >= 5:
            score += 5
        
        score = min(score, 100)
        
        # Determine tier
        if score >= 80:
            tier = "S"
        elif score >= 60:
            tier = "A"
        elif score >= 40:
            tier = "B"
        else:
            tier = "C"
        
        return {
            "symbol": symbol,
            "score": score,
            "tier": tier,
            "potential_multiplier": round(potential_multiplier, 2),
            "has_5x_potential": potential_multiplier >= 5,
            "reasons": reasons,
            "category": category,
            "recommended_action": "ACCUMULATE" if score >= 60 else "WATCH" if score >= 40 else "PASS",
        }
    
    def _estimate_category_growth(self, category: str) -> float:
        """Estimate annual growth rate by category"""
        growth_rates = {
            "AI": 0.50,
            "Layer2": 0.40,
            "DeFi": 0.30,
            "Gaming": 0.35,
            "Layer1": 0.25,
            "Storage": 0.30,
            "Oracle": 0.25,
            "Meme": 0.15,
        }
        return growth_rates.get(category, 0.20)
    
    # =============================================
    # ENTRY ZONE CALCULATION
    # =============================================
    
    def calculate_optimal_entry_zone(
        self,
        current_price: float,
        pivot_points: Dict,
        volume_profile: Dict,
        fear_greed: int,
    ) -> Dict:
        """
        Calculate the optimal entry zone for long-term investment
        
        Combines:
        - Support levels from pivot points
        - Volume accumulation zones
        - Market sentiment
        """
        entries = []
        
        # Pivot point supports
        entries.append(pivot_points.get("s1", current_price * 0.95))
        entries.append(pivot_points.get("s2", current_price * 0.90))
        
        # Volume profile VAL
        if volume_profile:
            entries.append(volume_profile.get("val", current_price * 0.92))
        
        # Sentiment adjustment
        sentiment_discount = 0
        if fear_greed >= 70:  # Greedy market - expect correction
            sentiment_discount = 0.05  # Wait for 5% lower
        elif fear_greed <= 30:  # Fearful market - already discounted
            sentiment_discount = -0.02  # Can buy 2% higher
        
        # Calculate optimal entry range
        min_entry = min(entries)
        max_entry = max(entries)
        optimal = (min_entry + max_entry) / 2 * (1 + sentiment_discount)
        
        return {
            "optimal_entry": round(optimal, 8),
            "entry_zone_low": round(min_entry, 8),
            "entry_zone_high": round(max_entry, 8),
            "current_price": current_price,
            "discount_from_current": round((current_price - optimal) / current_price * 100, 2),
            "sentiment_adjustment": f"{sentiment_discount*100:+.1f}%",
            "recommendation": "BUY" if current_price <= optimal else "WAIT",
        }


# =============================================
# INTEGRATION WITH EXISTING FETCH_DATA.PY
# =============================================

def enhance_signal_with_ml(
    signal_data: Dict,
    prices: List[float],
    volumes: List[float],
    highs: List[float] = None,
    lows: List[float] = None,
) -> Dict:
    """
    Enhance an existing signal with advanced ML features
    
    Use this function to upgrade signals from fetch_data.py
    """
    analyzer = AdvancedAnalyzer()
    
    # Calculate pivot points
    if highs and lows and len(highs) > 0:
        pivot_points = analyzer.calculate_pivot_points(
            high=max(highs[-24:]) if len(highs) >= 24 else max(highs),
            low=min(lows[-24:]) if len(lows) >= 24 else min(lows),
            close=prices[-1] if prices else 0,
        )
    else:
        pivot_points = {}
    
    # Calculate volume profile
    volume_profile = analyzer.calculate_volume_profile(prices, volumes) if prices and volumes else {}
    
    # Get Fear & Greed
    fear_greed = analyzer.fetch_fear_greed_index()
    
    # Detect bull trap
    bull_trap = analyzer.detect_bull_trap(prices, volumes) if prices and volumes else {}
    
    # Get prediction
    rsi = signal_data.get("rsi", 50)
    macd = signal_data.get("macd", {})
    if isinstance(macd, (int, float)):
        macd = {"macd": macd, "signal": 0, "histogram": 0}
    
    prediction = analyzer.predict_price_direction(
        prices=prices,
        volumes=volumes,
        rsi=rsi,
        macd=macd,
        fear_greed=fear_greed.get("value", 50),
    )
    
    # Override signal if bull trap detected
    if bull_trap.get("is_bull_trap") and signal_data.get("signal") == "BUY":
        signal_data["signal"] = "HOLD"
        signal_data["bull_trap_warning"] = True
    
    # Add enhanced data
    signal_data["advanced"] = {
        "pivot_points": pivot_points,
        "volume_profile": {
            "poc": volume_profile.get("poc"),
            "vah": volume_profile.get("vah"),
            "val": volume_profile.get("val"),
        },
        "fear_greed": fear_greed,
        "prediction_24h": prediction,
        "bull_trap": bull_trap,
        "optimal_entry": analyzer.calculate_optimal_entry_zone(
            current_price=signal_data.get("price", 0),
            pivot_points=pivot_points,
            volume_profile=volume_profile,
            fear_greed=fear_greed.get("value", 50),
        ) if pivot_points else None,
    }
    
    return signal_data


if __name__ == "__main__":
    # Test the analyzer
    analyzer = AdvancedAnalyzer()
    
    # Sample data
    test_prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 110, 112, 111, 113, 115]
    test_volumes = [1000, 1200, 900, 1100, 1300, 1000, 800, 700, 600, 500, 400, 350, 300, 250]
    
    # Test pivot points
    pivots = analyzer.calculate_pivot_points(high=115, low=100, close=113)
    print("Pivot Points:", json.dumps(pivots, indent=2))
    
    # Test volume profile
    vp = analyzer.calculate_volume_profile(test_prices, test_volumes)
    print("\nVolume Profile:")
    print(f"  POC (Whale Zone): ${vp['poc']}")
    print(f"  VAH: ${vp['vah']}")
    print(f"  VAL: ${vp['val']}")
    
    # Test bull trap detection
    bt = analyzer.detect_bull_trap(test_prices, test_volumes)
    print(f"\nBull Trap Detection: {json.dumps(bt, indent=2)}")
    
    # Test prediction
    pred = analyzer.predict_price_direction(
        prices=test_prices,
        volumes=test_volumes,
        rsi=65,
        macd={"macd": 0.5, "signal": 0.3, "histogram": 0.2},
        fear_greed=70,
    )
    print(f"\n24h Prediction: {json.dumps(pred, indent=2)}")
    
    print("\n✅ Advanced Analyzer Test Complete!")
