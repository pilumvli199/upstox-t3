#!/usr/bin/env python3
"""
HYBRID TRADING BOT v31.0 - COMPACT PROFESSIONAL
================================================
✅ COMPACT TELEGRAM ALERTS (Less confusing)
✅ OI PROPERLY CALCULATED (Price + OI trend analysis)
✅ WHITE BACKGROUND CHARTS (Professional look)
✅ YELLOW DIAMOND HIGHLIGHT (Last candle)
✅ COMPRESSED OHLC (Sent to DeepSeek)
✅ OI IN LOGS & TELEGRAM (Complete tracking)
✅ REDIS OI TRACKING (Every scan saved)
"""

import os
import asyncio
import requests
import urllib.parse
from datetime import datetime, timedelta, time
import pytz
import time as time_sleep
from telegram import Bot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import traceback
import re

# Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Configuration
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('hybrid_v31.log')]
)
logger = logging.getLogger(__name__)

# API Keys
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

BASE_URL = "https://api.upstox.com"
SCAN_INTERVAL = 900
REDIS_EXPIRY = 86400

# Symbols
INDICES = {
    "NSE_INDEX|Nifty 50": {"name": "NIFTY", "display_name": "NIFTY 50", "expiry_day": 3},
    "NSE_INDEX|Nifty Bank": {"name": "BANKNIFTY", "display_name": "BANK NIFTY", "expiry_day": 2},
    "NSE_INDEX|NIFTY MID SELECT": {"name": "MIDCPNIFTY", "display_name": "MIDCAP NIFTY", "expiry_day": 0}
}

FO_STOCKS = {
    "NSE_EQ|INE467B01029": {"name": "TATAMOTORS", "display_name": "TATA MOTORS"},
    "NSE_EQ|INE585B01010": {"name": "MARUTI", "display_name": "MARUTI SUZUKI"},
    "NSE_EQ|INE101A01026": {"name": "M&M", "display_name": "M&M"},
    "NSE_EQ|INE917I01010": {"name": "BAJAJ-AUTO", "display_name": "BAJAJ AUTO"},
    "NSE_EQ|INE040A01034": {"name": "HDFCBANK", "display_name": "HDFC BANK"},
    "NSE_EQ|INE090A01021": {"name": "ICICIBANK", "display_name": "ICICI BANK"},
    "NSE_EQ|INE062A01020": {"name": "SBIN", "display_name": "STATE BANK"},
    "NSE_EQ|INE238A01034": {"name": "AXISBANK", "display_name": "AXIS BANK"},
    "NSE_EQ|INE237A01028": {"name": "KOTAKBANK", "display_name": "KOTAK BANK"},
    "NSE_EQ|INE009A01021": {"name": "INFY", "display_name": "INFOSYS"},
    "NSE_EQ|INE854D01024": {"name": "TCS", "display_name": "TCS"},
    "NSE_EQ|INE002A01018": {"name": "RELIANCE", "display_name": "RELIANCE"},
    "NSE_EQ|INE397D01024": {"name": "BHARTIARTL", "display_name": "BHARTI AIRTEL"},
    "NSE_EQ|INE296A01024": {"name": "BAJFINANCE", "display_name": "BAJAJ FINANCE"}
}

ALL_SYMBOLS = {**INDICES, **FO_STOCKS}

CONFIDENCE_MIN = 75
SCORE_MIN = 70
ALIGNMENT_MIN = 18

# ==================== DATA CLASSES ====================
@dataclass
class StrikeData:
    strike: int
    ce_oi: int
    pe_oi: int
    ce_volume: int
    pe_volume: int

@dataclass
class OIAnalysis:
    """✅ ENHANCED: Price + OI Trend Analysis"""
    pcr: float
    support_strike: int
    resistance_strike: int
    ce_oi_change_pct: float
    pe_oi_change_pct: float
    ce_volume_change_pct: float
    pe_volume_change_pct: float
    overall_sentiment: str
    price_trend: str  # UP/DOWN/SIDEWAYS
    oi_trend: str  # INCREASING/DECREASING/STABLE
    market_signal: str  # LONG_BUILDUP/SHORT_BUILDUP/SHORT_COVERING/LONG_UNWINDING/NEUTRAL
    signal_strength: str  # STRONG/WEAK
    timestamp: datetime

@dataclass
class MultiTimeframeData:
    df_5m: pd.DataFrame
    df_15m: pd.DataFrame
    df_1h: pd.DataFrame
    current_price: float
    trend_1h: str
    pattern_15m: str

@dataclass
class AIAnalysis:
    opportunity: str
    confidence: int
    chart_score: int
    oi_score: int
    alignment_score: int
    total_score: int
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: str
    recommended_strike: int
    pattern_signal: str
    oi_flow_signal: str
    support_levels: List[float]
    resistance_levels: List[float]
    risk_factors: List[str]
    tf_1h_trend: str
    tf_15m_pattern: str
    tf_alignment: str
    ai_reasoning: str

# ==================== REDIS WITH OI TRACKING ====================
class RedisOITracker:
    def __init__(self):
        self.redis_client = None
        self.connected = False
        
        if not REDIS_AVAILABLE:
            logger.warning("⚠️ Redis not available")
            return
        
        try:
            logger.info("🔄 Connecting to Redis...")
            self.redis_client = redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            self.redis_client.ping()
            self.connected = True
            logger.info("✅ Redis connected!")
        except Exception as e:
            logger.error(f"❌ Redis failed: {e}")
    
    def save_oi_snapshot(self, symbol: str, expiry: str, spot_price: float,
                        total_ce_oi: int, total_pe_oi: int,
                        total_ce_volume: int, total_pe_volume: int):
        """✅ Save OI snapshot every scan"""
        if not self.redis_client or not self.connected:
            return
        
        try:
            timestamp = datetime.now(IST)
            key = f"oi:{symbol}:{expiry}:{timestamp.strftime('%Y%m%d_%H%M')}"
            
            data = {
                "spot_price": spot_price,
                "ce_oi": total_ce_oi,
                "pe_oi": total_pe_oi,
                "ce_volume": total_ce_volume,
                "pe_volume": total_pe_volume,
                "pcr": total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0,
                "timestamp": timestamp.isoformat()
            }
            
            self.redis_client.setex(key, REDIS_EXPIRY, json.dumps(data))
            logger.info(f"  💾 OI saved: {symbol} | CE:{total_ce_oi} PE:{total_pe_oi}")
        except Exception as e:
            logger.error(f"  ❌ Redis save error: {e}")
    
    def get_previous_oi(self, symbol: str, expiry: str) -> Optional[Dict]:
        """✅ Get previous scan OI (15 min ago)"""
        if not self.redis_client or not self.connected:
            return None
        
        try:
            now = datetime.now(IST)
            prev_time = now - timedelta(minutes=15)
            
            # Round to 15-min interval
            prev_time = prev_time.replace(
                minute=(prev_time.minute // 15) * 15,
                second=0,
                microsecond=0
            )
            
            key = f"oi:{symbol}:{expiry}:{prev_time.strftime('%Y%m%d_%H%M')}"
            data = self.redis_client.get(key)
            
            if data:
                parsed = json.loads(data)
                logger.info(f"  ⏰ Previous OI: {prev_time.strftime('%H:%M')} | PCR:{parsed['pcr']:.2f}")
                return parsed
            else:
                logger.info(f"  ⚠️ No previous OI (first scan)")
                return None
        except Exception as e:
            logger.error(f"  ❌ Redis get error: {e}")
            return None

# ==================== OI ANALYZER (ENHANCED) ====================
class OIAnalyzer:
    @staticmethod
    def analyze_price_oi_trend(current_price: float, prev_price: float,
                               current_oi: int, prev_oi: int,
                               current_volume: int, prev_volume: int) -> OIAnalysis:
        """
        ✅ PRICE + OI TREND ANALYSIS
        
        Market Trend Rules:
        1. Price ⬆️ + OI ⬆️ = LONG BUILDUP (Strong Bullish)
        2. Price ⬇️ + OI ⬆️ = SHORT BUILDUP (Strong Bearish)
        3. Price ⬆️ + OI ⬇️ = SHORT COVERING (Weak Bullish)
        4. Price ⬇️ + OI ⬇️ = LONG UNWINDING (Weak Bearish)
        """
        
        # Price trend
        price_change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
        
        if price_change_pct > 0.5:
            price_trend = "UP"
        elif price_change_pct < -0.5:
            price_trend = "DOWN"
        else:
            price_trend = "SIDEWAYS"
        
        # OI trend
        oi_change_pct = ((current_oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0
        
        if oi_change_pct > 5:
            oi_trend = "INCREASING"
        elif oi_change_pct < -5:
            oi_trend = "DECREASING"
        else:
            oi_trend = "STABLE"
        
        # Market signal
        if price_trend == "UP" and oi_trend == "INCREASING":
            market_signal = "LONG_BUILDUP"
            signal_strength = "STRONG"
            sentiment = "BULLISH"
        elif price_trend == "DOWN" and oi_trend == "INCREASING":
            market_signal = "SHORT_BUILDUP"
            signal_strength = "STRONG"
            sentiment = "BEARISH"
        elif price_trend == "UP" and oi_trend == "DECREASING":
            market_signal = "SHORT_COVERING"
            signal_strength = "WEAK"
            sentiment = "BULLISH"
        elif price_trend == "DOWN" and oi_trend == "DECREASING":
            market_signal = "LONG_UNWINDING"
            signal_strength = "WEAK"
            sentiment = "BEARISH"
        else:
            market_signal = "NEUTRAL"
            signal_strength = "WEAK"
            sentiment = "NEUTRAL"
        
        # Volume analysis
        volume_change_pct = ((current_volume - prev_volume) / prev_volume * 100) if prev_volume > 0 else 0
        
        return {
            "price_trend": price_trend,
            "price_change_pct": price_change_pct,
            "oi_trend": oi_trend,
            "oi_change_pct": oi_change_pct,
            "volume_change_pct": volume_change_pct,
            "market_signal": market_signal,
            "signal_strength": signal_strength,
            "sentiment": sentiment
        }

# ==================== EXPIRY CALCULATOR ====================
class ExpiryCalculator:
    @staticmethod
    def get_all_expiries_from_api(instrument_key: str) -> List[str]:
        try:
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
            }
            
            encoded_key = urllib.parse.quote(instrument_key, safe='')
            url = f"{BASE_URL}/v2/option/contract?instrument_key={encoded_key}"
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                contracts = response.json().get('data', [])
                expiries = sorted(list(set(c['expiry'] for c in contracts if 'expiry' in c)))
                return expiries
            return []
        except:
            return []
    
    @staticmethod
    def calculate_monthly_expiry(symbol_name: str, expiry_day: int = 3) -> str:
        today = datetime.now(IST).date()
        current_time = datetime.now(IST).time()
        
        last_day = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        days_to_subtract = (last_day.weekday() - expiry_day) % 7
        expiry = last_day - timedelta(days=days_to_subtract)
        
        if expiry < today or (expiry == today and current_time >= time(15, 30)):
            next_month = (today.replace(day=28) + timedelta(days=4))
            last_day = (next_month.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            days_to_subtract = (last_day.weekday() - expiry_day) % 7
            expiry = last_day - timedelta(days=days_to_subtract)
        
        return expiry.strftime('%Y-%m-%d')
    
    @staticmethod
    def get_best_expiry(instrument_key: str, symbol_info: Dict) -> str:
        expiry_day = symbol_info.get('expiry_day', 3)
        
        # Try API
        expiries = ExpiryCalculator.get_all_expiries_from_api(instrument_key)
        
        if expiries:
            today = datetime.now(IST).date()
            now_time = datetime.now(IST).time()
            
            future_expiries = []
            for exp_str in expiries:
                try:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                    if exp_date > today or (exp_date == today and now_time < time(15, 30)):
                        future_expiries.append(exp_str)
                except:
                    continue
            
            if future_expiries:
                return min(future_expiries)
        
        # Fallback
        return ExpiryCalculator.calculate_monthly_expiry(symbol_info.get('name', ''), expiry_day)
    
    @staticmethod
    def get_display_expiry(expiry_str: str) -> str:
        try:
            dt = datetime.strptime(expiry_str, '%Y-%m-%d')
            return dt.strftime('%d%b%y').upper()
        except:
            return expiry_str

# ==================== DATA FETCHER ====================
class UpstoxDataFetcher:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
    
    def get_spot_price(self, instrument_key: str) -> float:
        for attempt in range(3):
            try:
                encoded_key = urllib.parse.quote(instrument_key, safe='')
                url = f"{BASE_URL}/v2/market-quote/ltp?instrument_key={encoded_key}"
                
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json().get('data', {})
                    if data:
                        ltp = list(data.values())[0].get('last_price', 0)
                        return float(ltp)
                
                time_sleep.sleep(2)
            except Exception as e:
                time_sleep.sleep(2)
        
        return 0.0
    
    def get_option_chain(self, instrument_key: str, expiry: str) -> List[StrikeData]:
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                encoded_key = urllib.parse.quote(instrument_key, safe='')
                url = f"{BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
                
                response = requests.get(url, headers=self.headers, timeout=20)
                
                if response.status_code == 200:
                    data = response.json()
                    strikes_raw = data.get('data', [])
                    
                    if not strikes_raw:
                        if attempt < max_retries - 1:
                            time_sleep.sleep(2 * (attempt + 1))
                            continue
                        return []
                    
                    strikes = []
                    for item in strikes_raw:
                        call_data = item.get('call_options', {}).get('market_data', {})
                        put_data = item.get('put_options', {}).get('market_data', {})
                        
                        ce_oi = call_data.get('oi', 0)
                        pe_oi = put_data.get('oi', 0)
                        
                        if ce_oi == 0 and pe_oi == 0:
                            continue
                        
                        strikes.append(StrikeData(
                            strike=int(item.get('strike_price', 0)),
                            ce_oi=ce_oi,
                            pe_oi=pe_oi,
                            ce_volume=call_data.get('volume', 0),
                            pe_volume=put_data.get('volume', 0)
                        ))
                    
                    if strikes:
                        return strikes
                
                if response.status_code == 429:
                    time_sleep.sleep(2 * (attempt + 2))
                    continue
                
                if response.status_code == 404:
                    return []
                
                if attempt < max_retries - 1:
                    time_sleep.sleep(2 * (attempt + 1))
            
            except Exception as e:
                if attempt < max_retries - 1:
                    time_sleep.sleep(2 * (attempt + 1))
        
        return []
    
    def get_multi_timeframe_data(self, instrument_key: str) -> Optional[MultiTimeframeData]:
        try:
            encoded_key = urllib.parse.quote(instrument_key, safe='')
            all_candles = []
            
            # Historical
            try:
                to_date = (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')
                from_date = (datetime.now(IST) - timedelta(days=10)).strftime('%Y-%m-%d')
                url = f"{BASE_URL}/v2/historical-candle/{encoded_key}/30minute/{to_date}/{from_date}"
                
                response = requests.get(url, headers=self.headers, timeout=20)
                
                if response.status_code == 200:
                    candles_30min = response.json().get('data', {}).get('candles', [])
                    all_candles.extend(candles_30min)
            except:
                pass
            
            # Intraday
            try:
                url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/1minute"
                response = requests.get(url, headers=self.headers, timeout=20)
                
                if response.status_code == 200:
                    candles_1min = response.json().get('data', {}).get('candles', [])
                    all_candles.extend(candles_1min)
            except:
                pass
            
            if not all_candles:
                return None
            
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').astype(float)
            df = df.sort_index()
            
            # Resample
            df_5m = df.resample('5min').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum', 'oi': 'last'
            }).dropna()
            
            df_15m = df.resample('15min').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum', 'oi': 'last'
            }).dropna()
            
            df_1h = df.resample('1H').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum', 'oi': 'last'
            }).dropna()
            
            current_price = df_15m['close'].iloc[-1] if len(df_15m) > 0 else 0
            
            # 1h trend
            trend_1h = "NEUTRAL"
            if len(df_1h) >= 20:
                ma20 = df_1h['close'].rolling(20).mean().iloc[-1]
                if current_price > ma20:
                    trend_1h = "BULLISH"
                elif current_price < ma20:
                    trend_1h = "BEARISH"
            
            return MultiTimeframeData(
                df_5m=df_5m,
                df_15m=df_15m,
                df_1h=df_1h,
                current_price=current_price,
                trend_1h=trend_1h,
                pattern_15m="ANALYZING"
            )
            
        except Exception as e:
            logger.error(f"Multi-TF error: {e}")
            return None

# ==================== CHART ANALYZER ====================
class ChartAnalyzer:
    @staticmethod
    def analyze_patterns(df: pd.DataFrame) -> Dict:
        try:
            if len(df) < 30:
                return {"pattern": "NONE", "signal": "NEUTRAL"}
            
            recent = df.tail(100)
            last_20 = recent.tail(20)
            patterns = []
            
            # Engulfing
            for i in range(1, len(last_20)):
                prev = last_20.iloc[i-1]
                curr = last_20.iloc[i]
                
                if (prev['close'] < prev['open'] and curr['close'] > curr['open'] and
                    curr['open'] < prev['close'] and curr['close'] > prev['open']):
                    patterns.append("BULLISH_ENGULFING")
                
                if (prev['close'] > prev['open'] and curr['close'] < curr['open'] and
                    curr['open'] > prev['close'] and curr['close'] < prev['open']):
                    patterns.append("BEARISH_ENGULFING")
            
            # Breakout
            high_20 = recent['high'].rolling(20).max().iloc[-1]
            low_20 = recent['low'].rolling(20).min().iloc[-1]
            current = recent['close'].iloc[-1]
            
            if current > high_20 * 0.999:
                patterns.append("BREAKOUT")
            elif current < low_20 * 1.001:
                patterns.append("BREAKDOWN")
            
            bullish = sum(1 for p in patterns if "BULLISH" in p or p == "BREAKOUT")
            bearish = sum(1 for p in patterns if "BEARISH" in p or p == "BREAKDOWN")
            
            if bullish > bearish:
                signal = "BULLISH"
            elif bearish > bullish:
                signal = "BEARISH"
            else:
                signal = "NEUTRAL"
            
            return {
                "pattern": ", ".join(patterns[:3]) if patterns else "NONE",
                "signal": signal
            }
        except:
            return {"pattern": "NONE", "signal": "NEUTRAL"}
    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame) -> Dict:
        try:
            if len(df) < 50:
                current = df['close'].iloc[-1]
                return {
                    'supports': [current * 0.98],
                    'resistances': [current * 1.02]
                }
            
            recent = df.tail(100)
            current = recent['close'].iloc[-1]
            
            highs = recent['high'].values
            lows = recent['low'].values
            
            resistance_levels = []
            support_levels = []
            
            window = 5
            for i in range(window, len(recent) - window):
                if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
                   all(highs[i] >= highs[i+j] for j in range(1, window+1)):
                    resistance_levels.append(highs[i])
                
                if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
                   all(lows[i] <= lows[i+j] for j in range(1, window+1)):
                    support_levels.append(lows[i])
            
            def cluster(levels):
                if not levels:
                    return []
                levels = sorted(levels)
                clustered = []
                current_cluster = [levels[0]]
                for level in levels[1:]:
                    if abs(level - current_cluster[-1]) / current_cluster[-1] < 0.005:
                        current_cluster.append(level)
                    else:
                        clustered.append(np.mean(current_cluster))
                        current_cluster = [level]
                clustered.append(np.mean(current_cluster))
                return clustered
            
            resistances = cluster(resistance_levels)
            supports = cluster(support_levels)
            
            resistances = [r for r in resistances if 0.001 <= (r - current)/current <= 0.05]
            supports = [s for s in supports if 0.001 <= (current - s)/current <= 0.05]
            
            return {
                'supports': supports[:3] if supports else [current * 0.98],
                'resistances': resistances[:3] if resistances else [current * 1.02]
            }
        except:
            current = df['close'].iloc[-1]
            return {
                'supports': [current * 0.98],
                'resistances': [current * 1.02]
            }

# ==================== DATA COMPRESSOR (OHLC) ====================
class DataCompressor:
    @staticmethod
    def compress_to_ohlc(df: pd.DataFrame, limit: int = None) -> str:
        """✅ Compress to O,H,L,C format"""
        if limit:
            df = df.tail(limit)
        
        ohlc_list = []
        for _, row in df.iterrows():
            ohlc_list.append(f"{row['open']:.1f},{row['high']:.1f},{row['low']:.1f},{row['close']:.1f}")
        
        return '|'.join(ohlc_list)

# ==================== DEEPSEEK AI ====================
class DeepSeekAnalyzer:
    @staticmethod
    def generate_prompt(symbol: str, mtf_data: MultiTimeframeData, spot_price: float,
                       oi_analysis: Dict, trend_1h: str, pattern_15m: Dict,
                       sr_levels: Dict) -> str:
        
        # ✅ Compressed OHLC (O,H,L,C format)
        ohlc_1h = DataCompressor.compress_to_ohlc(mtf_data.df_1h, limit=50)
        ohlc_15m = DataCompressor.compress_to_ohlc(mtf_data.df_15m, limit=100)
        ohlc_5m = DataCompressor.compress_to_ohlc(mtf_data.df_5m, limit=50)
        
        prompt = f"""Expert F&O trader. Analyze {symbol} using multi-timeframe + OI confluence.

**SPOT:** ₹{spot_price:.2f}

**1H TREND:** {trend_1h}
**1H DATA (O,H,L,C):** {ohlc_1h}

**15M PATTERN:** {pattern_15m['pattern']}
**15M DATA (O,H,L,C):** {ohlc_15m}
**Support:** {', '.join([f"₹{s:.0f}" for s in sr_levels['supports'][:2]])}
**Resistance:** {', '.join([f"₹{r:.0f}" for r in sr_levels['resistances'][:2]])}

**5M DATA (O,H,L,C):** {ohlc_5m}

**OI ANALYSIS (CRITICAL):**
Price Trend: {oi_analysis['price_trend']} ({oi_analysis['price_change_pct']:+.1f}%)
OI Trend: {oi_analysis['oi_trend']} ({oi_analysis['oi_change_pct']:+.1f}%)
Market Signal: {oi_analysis['market_signal']} ({oi_analysis['signal_strength']})
Sentiment: {oi_analysis['sentiment']}

**OI INTERPRETATION RULES:**
1. Price⬆️ + OI⬆️ = LONG BUILDUP (Strong Bullish) = Score +30
2. Price⬇️ + OI⬆️ = SHORT BUILDUP (Strong Bearish) = Score +30
3. Price⬆️ + OI⬇️ = SHORT COVERING (Weak Bullish) = Score +15
4. Price⬇️ + OI⬇️ = LONG UNWINDING (Weak Bearish) = Score +15

**SCORING (/125):**
- Chart: /50 (1H trend + 15M patterns + S/R)
- OI: /50 (Price+OI analysis + Strength)
- TF Alignment: /25 (1H + 15M + 5M aligned)

**THRESHOLDS:**
- Total Score: ≥70
- Confidence: ≥75%
- TF Alignment: ≥18

**OUTPUT JSON:**
{{
  "opportunity": "CE_BUY/PE_BUY/WAIT",
  "confidence": 80,
  "chart_score": 38,
  "oi_score": 35,
  "alignment_score": 20,
  "total_score": 93,
  "entry_price": {spot_price:.2f},
  "stop_loss": 0.0,
  "target_1": 0.0,
  "target_2": 0.0,
  "risk_reward": "1:2",
  "recommended_strike": {int(spot_price)},
  "pattern_signal": "Pattern details",
  "oi_flow_signal": "OI signal details",
  "support_levels": {sr_levels['supports'][:2]},
  "resistance_levels": {sr_levels['resistances'][:2]},
  "risk_factors": ["Risk1", "Risk2"],
  "tf_1h_trend": "{trend_1h}",
  "tf_15m_pattern": "{pattern_15m['pattern']}",
  "tf_alignment": "STRONG/MODERATE/WEAK",
  "ai_reasoning": "Chart(38/50): Details. OI(35/50): Details. Total: 93/125"
}}

Reply JSON only. If Score <70, return "WAIT"."""
        
        return prompt
    
    @staticmethod
    def parse_response(content: str) -> Optional[AIAnalysis]:
        try:
            content = content.replace('```json', '').replace('```', '').strip()
            
            try:
                data = json.loads(content)
            except:
                match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', content, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                else:
                    return None
            
            return AIAnalysis(
                opportunity=data.get('opportunity', 'WAIT'),
                confidence=data.get('confidence', 0),
                chart_score=data.get('chart_score', 0),
                oi_score=data.get('oi_score', 0),
                alignment_score=data.get('alignment_score', 0),
                total_score=data.get('total_score', 0),
                entry_price=data.get('entry_price', 0),
                stop_loss=data.get('stop_loss', 0),
                target_1=data.get('target_1', 0),
                target_2=data.get('target_2', 0),
                risk_reward=data.get('risk_reward', '0:0'),
                recommended_strike=data.get('recommended_strike', 0),
                pattern_signal=data.get('pattern_signal', 'None'),
                oi_flow_signal=data.get('oi_flow_signal', 'Neutral'),
                support_levels=data.get('support_levels', []),
                resistance_levels=data.get('resistance_levels', []),
                risk_factors=data.get('risk_factors', []),
                tf_1h_trend=data.get('tf_1h_trend', 'NEUTRAL'),
                tf_15m_pattern=data.get('tf_15m_pattern', 'NONE'),
                tf_alignment=data.get('tf_alignment', 'WEAK'),
                ai_reasoning=data.get('ai_reasoning', 'No reasoning')
            )
        except:
            return None
    
    @staticmethod
    def analyze(symbol: str, mtf_data: MultiTimeframeData, spot_price: float,
               oi_analysis: Dict, trend_1h: str, pattern_15m: Dict,
               sr_levels: Dict) -> Optional[AIAnalysis]:
        try:
            prompt = DeepSeekAnalyzer.generate_prompt(
                symbol, mtf_data, spot_price, oi_analysis, trend_1h, pattern_15m, sr_levels
            )
            
            logger.info(f"  🤖 Calling DeepSeek V3...")
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "Expert F&O trader. JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000
                },
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=90
            )
            
            if response.status_code != 200:
                logger.error(f"  ❌ DeepSeek error: {response.status_code}")
                return None
            
            ai_content = response.json()['choices'][0]['message']['content']
            analysis = DeepSeekAnalyzer.parse_response(ai_content)
            
            if analysis:
                logger.info(f"  🤖 AI: {analysis.opportunity} | Score: {analysis.total_score}/125 | Conf: {analysis.confidence}%")
            
            return analysis
            
        except Exception as e:
            logger.error(f"  ❌ DeepSeek error: {e}")
            return None

# ==================== CHART GENERATOR (WHITE BG) ====================
class ChartGenerator:
    @staticmethod
    def create_chart(symbol: str, df: pd.DataFrame, analysis: AIAnalysis,
                    spot: float, oi_analysis: Dict, path: str):
        """✅ WHITE BACKGROUND + YELLOW HIGHLIGHT"""
        
        # Colors (Professional)
        BG = '#FFFFFF'  # White background
        GRID = '#E0E0E0'
        TEXT = '#2C3E50'
        GREEN = '#26a69a'
        RED = '#ef5350'
        YELLOW = '#FFD700'  # Gold yellow
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       facecolor=BG)
        
        ax1.set_facecolor(BG)
        df_plot = df.tail(150).reset_index(drop=True)
        
        # Candlesticks
        for idx, row in df_plot.iterrows():
            color = GREEN if row['close'] > row['open'] else RED
            
            # Wick
            ax1.plot([idx+0.3, idx+0.3], [row['low'], row['high']],
                    color=color, linewidth=1.2, alpha=0.8)
            
            # Body
            ax1.add_patch(Rectangle(
                (idx, min(row['open'], row['close'])),
                0.6,
                abs(row['close'] - row['open']) if abs(row['close'] - row['open']) > 0 else spot * 0.0001,
                facecolor=color,
                edgecolor=color,
                alpha=0.85
            ))
        
        # ✅ YELLOW DIAMOND HIGHLIGHT (Last candle)
        last_idx = len(df_plot) - 1
        last_close = df_plot.iloc[-1]['close']
        ax1.scatter([last_idx + 0.3], [last_close],
                   color=YELLOW, s=250, marker='D', zorder=10,
                   edgecolors=TEXT, linewidths=2, alpha=0.9)
        
        # Support/Resistance
        for sup in analysis.support_levels[:2]:
            if sup > 0:
                ax1.axhline(sup, color=GREEN, linestyle='--', linewidth=1.8, alpha=0.7)
                ax1.text(2, sup, f' S: ₹{sup:.1f}',
                        color=GREEN, fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                 alpha=0.8, edgecolor=GREEN, linewidth=1.5))
        
        for res in analysis.resistance_levels[:2]:
            if res > 0:
                ax1.axhline(res, color=RED, linestyle='--', linewidth=1.8, alpha=0.7)
                ax1.text(2, res, f' R: ₹{res:.1f}',
                        color=RED, fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                 alpha=0.8, edgecolor=RED, linewidth=1.5))
        
        # Trade levels
        if analysis.opportunity != "WAIT":
            ax1.axhline(analysis.entry_price, color='#FF9800', linewidth=2.5, linestyle=':', alpha=0.8)
            ax1.axhline(analysis.stop_loss, color=RED, linewidth=2, linestyle=':', alpha=0.7)
            ax1.axhline(analysis.target_1, color=GREEN, linewidth=2, linestyle=':', alpha=0.7)
            ax1.axhline(analysis.target_2, color=GREEN, linewidth=1.5, linestyle=':', alpha=0.6)
        
        # Info box
        signal_emoji = "🟢" if analysis.opportunity == "CE_BUY" else ("🔴" if analysis.opportunity == "PE_BUY" else "⏸️")
        score_emoji = "🔥" if analysis.total_score >= 85 else ("✅" if analysis.total_score >= 70 else "⚠️")
        
        info = f"""{signal_emoji} {analysis.opportunity}

{score_emoji} SCORE: {analysis.total_score}/125
├─ Chart: {analysis.chart_score}/50
├─ OI: {analysis.oi_score}/50
└─ Align: {analysis.alignment_score}/25

Confidence: {analysis.confidence}%
TF: {analysis.tf_alignment}

1H: {analysis.tf_1h_trend}
15M: {analysis.tf_15m_pattern[:20]}

OI Signal: {oi_analysis['market_signal']}
Strength: {oi_analysis['signal_strength']}

Entry: ₹{analysis.entry_price:.1f}
SL: ₹{analysis.stop_loss:.1f}
T1: ₹{analysis.target_1:.1f}
R:R: {analysis.risk_reward}"""
        
        ax1.text(0.01, 0.99, info, transform=ax1.transAxes,
                fontsize=8, va='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                         alpha=0.95, edgecolor=TEXT, linewidth=2),
                color=TEXT)
        
        title = f"{score_emoji} {symbol} | 15M | Score: {analysis.total_score}/125"
        ax1.set_title(title, color=TEXT, fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, color=GRID, alpha=0.4, linestyle='-', linewidth=0.8)
        ax1.tick_params(colors=TEXT, labelsize=10)
        ax1.set_ylabel('Price (₹)', color=TEXT, fontsize=11, fontweight='bold')
        ax1.spines['top'].set_color(GRID)
        ax1.spines['right'].set_color(GRID)
        ax1.spines['bottom'].set_color(GRID)
        ax1.spines['left'].set_color(GRID)
        
        # Volume
        ax2.set_facecolor(BG)
        colors = [GREEN if df_plot.iloc[i]['close'] > df_plot.iloc[i]['open'] else RED
                 for i in range(len(df_plot))]
        ax2.bar(range(len(df_plot)), df_plot['volume'], color=colors, alpha=0.7, width=0.8)
        ax2.set_ylabel('Volume', color=TEXT, fontsize=10, fontweight='bold')
        ax2.tick_params(colors=TEXT, labelsize=9)
        ax2.grid(True, color=GRID, alpha=0.3)
        ax2.spines['top'].set_color(GRID)
        ax2.spines['right'].set_color(GRID)
        ax2.spines['bottom'].set_color(GRID)
        ax2.spines['left'].set_color(GRID)
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, facecolor=BG, edgecolor='none')
        plt.close()
        logger.info(f"  📊 Chart saved (white bg)")

# ==================== TELEGRAM NOTIFIER (COMPACT) ====================
class TelegramNotifier:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
    
    async def send_startup(self, redis_connected: bool):
        msg = f"""🚀 **BOT v31.0 STARTED**

⏰ {datetime.now(IST).strftime('%d-%b %H:%M IST')}

✅ Multi-TF: 1H + 15M + 5M
✅ OI Analysis: Price + OI Trend
✅ DeepSeek V3 AI
✅ Redis: {'🟢 Connected' if redis_connected else '🔴 Off'}

📊 Monitoring: {len(ALL_SYMBOLS)} symbols
🔄 Scan: Every 15 min

🟢 **BOT ACTIVE**"""
        
        await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
    
    async def send_alert(self, symbol: str, display_name: str, analysis: AIAnalysis,
                        oi_analysis: Dict, chart_path: str, expiry: str):
        """✅ COMPACT ALERT"""
        try:
            # Send chart
            with open(chart_path, 'rb') as photo:
                await self.bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo)
            
            signal_emoji = "🟢" if analysis.opportunity == "CE_BUY" else "🔴"
            score_emoji = "🔥" if analysis.total_score >= 85 else "✅"
            
            # ✅ COMPACT MESSAGE
            msg = f"""━━━━━━━━━━━━━━━━━━━━━━
{signal_emoji} **{display_name} {analysis.opportunity}**
━━━━━━━━━━━━━━━━━━━━━━

{score_emoji} **SCORE: {analysis.total_score}/125**
Chart: {analysis.chart_score} | OI: {analysis.oi_score} | Align: {analysis.alignment_score}
**Confidence: {analysis.confidence}%**

━━━━━━━━━━━━━━━━━━━━━━
📊 **MULTI-TIMEFRAME**
━━━━━━━━━━━━━━━━━━━━━━

**1H:** {analysis.tf_1h_trend}
**15M:** {analysis.tf_15m_pattern[:30]}
**Alignment:** {analysis.tf_alignment}

━━━━━━━━━━━━━━━━━━━━━━
⛓️ **OI ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━━

**Signal:** {oi_analysis['market_signal']}
**Strength:** {oi_analysis['signal_strength']}

Price: {oi_analysis['price_trend']} ({oi_analysis['price_change_pct']:+.1f}%)
OI: {oi_analysis['oi_trend']} ({oi_analysis['oi_change_pct']:+.1f}%)

**Interpretation:**"""

            # Add OI interpretation
            if oi_analysis['market_signal'] == 'LONG_BUILDUP':
                msg += "\n✅ Price⬆️ + OI⬆️ = Strong Bullish"
            elif oi_analysis['market_signal'] == 'SHORT_BUILDUP':
                msg += "\n❌ Price⬇️ + OI⬆️ = Strong Bearish"
            elif oi_analysis['market_signal'] == 'SHORT_COVERING':
                msg += "\n⚠️ Price⬆️ + OI⬇️ = Weak Bullish (Short Covering)"
            elif oi_analysis['market_signal'] == 'LONG_UNWINDING':
                msg += "\n⚠️ Price⬇️ + OI⬇️ = Weak Bearish (Long Unwinding)"
            else:
                msg += "\n⏸️ No clear OI signal"
            
            msg += f"""

━━━━━━━━━━━━━━━━━━━━━━
💰 **TRADE SETUP**
━━━━━━━━━━━━━━━━━━━━━━

**Entry:** ₹{analysis.entry_price:.2f}
**Stop Loss:** ₹{analysis.stop_loss:.2f}
**Target 1:** ₹{analysis.target_1:.2f}
**Target 2:** ₹{analysis.target_2:.2f}

**R:R:** {analysis.risk_reward}
**Strike:** {analysis.recommended_strike}

**Support:** {', '.join([f"₹{s:.1f}" for s in analysis.support_levels[:2]])}
**Resistance:** {', '.join([f"₹{r:.1f}" for r in analysis.resistance_levels[:2]])}

━━━━━━━━━━━━━━━━━━━━━━
🧠 **AI REASONING**
━━━━━━━━━━━━━━━━━━━━━━

{analysis.ai_reasoning}

━━━━━━━━━━━━━━━━━━━━━━
⚠️ **RISKS**
━━━━━━━━━━━━━━━━━━━━━━
"""
            
            for i, risk in enumerate(analysis.risk_factors[:2], 1):
                msg += f"\n{i}. {risk[:80]}"
            
            msg += f"""

━━━━━━━━━━━━━━━━━━━━━━

📅 Expiry: {expiry}
🕐 {datetime.now(IST).strftime('%d-%b %H:%M IST')}

━━━━━━━━━━━━━━━━━━━━━━"""
            
            await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
            logger.info(f"  ✅ Compact alert sent")
            
        except Exception as e:
            logger.error(f"Alert error: {e}")

# ==================== MAIN BOT ====================
class HybridBot:
    def __init__(self):
        logger.info("🔄 Initializing v31.0...")
        
        self.fetcher = UpstoxDataFetcher()
        self.redis = RedisOITracker()
        self.notifier = TelegramNotifier()
        self.processed = set()
        
        logger.info(f"✅ Bot ready | Redis: {self.redis.connected}")
    
    def is_market_open(self) -> bool:
        now = datetime.now(IST)
        if now.weekday() >= 5:
            return False
        current_time = now.time()
        return time(9, 15) <= current_time <= time(15, 30)
    
    async def analyze_symbol(self, instrument_key: str, symbol_info: Dict):
        try:
            symbol_name = symbol_info.get('name', '')
            display_name = symbol_info.get('display_name', symbol_name)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 {display_name}")
            logger.info(f"{'='*70}")
            
            # Expiry
            expiry_api = ExpiryCalculator.get_best_expiry(instrument_key, symbol_info)
            expiry_display = ExpiryCalculator.get_display_expiry(expiry_api)
            logger.info(f"  📅 Expiry: {expiry_display}")
            
            # Multi-TF data
            mtf_data = self.fetcher.get_multi_timeframe_data(instrument_key)
            if not mtf_data:
                logger.warning(f"  ❌ No TF data")
                return
            
            logger.info(f"  📊 TF: 1H({len(mtf_data.df_1h)}) 15M({len(mtf_data.df_15m)}) 5M({len(mtf_data.df_5m)})")
            
            # Spot
            spot_price = self.fetcher.get_spot_price(instrument_key)
            if spot_price == 0:
                spot_price = mtf_data.current_price
            
            logger.info(f"  💹 Spot: ₹{spot_price:.2f}")
            
            # Option chain
            all_strikes = self.fetcher.get_option_chain(instrument_key, expiry_api)
            if not all_strikes:
                logger.warning(f"  ⚠️ No OI data")
                return
            
            # ATM strikes
            atm = round(spot_price / 100) * 100
            atm_range = range(atm - 700, atm + 800, 100)
            top_strikes = sorted(
                [s for s in all_strikes if s.strike in atm_range],
                key=lambda x: (x.ce_oi + x.pe_oi),
                reverse=True
            )[:15]
            
            if not top_strikes:
                logger.warning(f"  ⚠️ No ATM strikes")
                return
            
            # Calculate OI totals
            total_ce_oi = sum(s.ce_oi for s in top_strikes)
            total_pe_oi = sum(s.pe_oi for s in top_strikes)
            total_ce_volume = sum(s.ce_volume for s in top_strikes)
            total_pe_volume = sum(s.pe_volume for s in top_strikes)
            
            # ✅ SAVE CURRENT OI
            self.redis.save_oi_snapshot(
                symbol_name, expiry_display, spot_price,
                total_ce_oi, total_pe_oi,
                total_ce_volume, total_pe_volume
            )
            
            # ✅ GET PREVIOUS OI
            prev_oi = self.redis.get_previous_oi(symbol_name, expiry_display)
            
            # ✅ ANALYZE PRICE + OI TREND
            if prev_oi:
                oi_analysis = OIAnalyzer.analyze_price_oi_trend(
                    spot_price, prev_oi['spot_price'],
                    total_ce_oi + total_pe_oi, prev_oi['ce_oi'] + prev_oi['pe_oi'],
                    total_ce_volume + total_pe_volume, prev_oi['ce_volume'] + prev_oi['pe_volume']
                )
            else:
                # First scan
                oi_analysis = {
                    'price_trend': 'SIDEWAYS',
                    'price_change_pct': 0,
                    'oi_trend': 'STABLE',
                    'oi_change_pct': 0,
                    'volume_change_pct': 0,
                    'market_signal': 'NEUTRAL',
                    'signal_strength': 'WEAK',
                    'sentiment': 'NEUTRAL'
                }
            
            # ✅ LOG OI ANALYSIS
            logger.info(f"  📊 OI ANALYSIS:")
            logger.info(f"     Price: {oi_analysis['price_trend']} ({oi_analysis['price_change_pct']:+.1f}%)")
            logger.info(f"     OI: {oi_analysis['oi_trend']} ({oi_analysis['oi_change_pct']:+.1f}%)")
            logger.info(f"     Signal: {oi_analysis['market_signal']} ({oi_analysis['signal_strength']})")
            logger.info(f"     Sentiment: {oi_analysis['sentiment']}")
            
            # Chart analysis
            pattern_15m = ChartAnalyzer.analyze_patterns(mtf_data.df_15m)
            sr_levels = ChartAnalyzer.calculate_support_resistance(mtf_data.df_15m)
            
            logger.info(f"  🕐 1H: {mtf_data.trend_1h}")
            logger.info(f"  ⏰ 15M: {pattern_15m['signal']} | {pattern_15m['pattern'][:40]}")
            
            # AI Analysis
            analysis = DeepSeekAnalyzer.analyze(
                display_name, mtf_data, spot_price,
                oi_analysis, mtf_data.trend_1h, pattern_15m, sr_levels
            )
            
            if not analysis:
                logger.info(f"  ⏸️ No AI analysis")
                return
            
            # Filters
            if analysis.opportunity == "WAIT":
                logger.info(f"  ⏸️ AI: WAIT")
                return
            
            if analysis.total_score < SCORE_MIN:
                logger.info(f"  ⏸️ Score {analysis.total_score} < {SCORE_MIN}")
                return
            
            if analysis.confidence < CONFIDENCE_MIN:
                logger.info(f"  ⏸️ Conf {analysis.confidence}% < {CONFIDENCE_MIN}%")
                return
            
            if analysis.alignment_score < ALIGNMENT_MIN:
                logger.info(f"  ⏸️ Align {analysis.alignment_score} < {ALIGNMENT_MIN}")
                return
            
            # Check duplicate
            signal_key = f"{symbol_name}_{analysis.opportunity}_{datetime.now(IST).strftime('%Y%m%d_%H')}"
            if signal_key in self.processed:
                logger.info(f"  ⏭️ Already alerted")
                return
            
            logger.info(f"  🚨 ALERT! Score: {analysis.total_score}/125")
            
            # Generate chart
            chart_path = f"/tmp/{symbol_name}_v31.png"
            ChartGenerator.create_chart(
                display_name, mtf_data.df_15m, analysis,
                spot_price, oi_analysis, chart_path
            )
            
            # Send alert
            await self.notifier.send_alert(
                symbol_name, display_name, analysis,
                oi_analysis, chart_path, expiry_display
            )
            
            self.processed.add(signal_key)
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            traceback.print_exc()
    
    async def run_scan(self):
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 SCAN - {datetime.now(IST).strftime('%H:%M:%S IST')}")
        logger.info(f"{'='*80}")
        
        for idx, (key, info) in enumerate(ALL_SYMBOLS.items(), 1):
            logger.info(f"\n[{idx}/{len(ALL_SYMBOLS)}]")
            await self.analyze_symbol(key, info)
            if idx < len(ALL_SYMBOLS):
                await asyncio.sleep(3)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ SCAN COMPLETE")
        logger.info(f"{'='*80}\n")
    
    async def run(self):
        logger.info("="*80)
        logger.info("HYBRID BOT v31.0 - COMPACT PROFESSIONAL")
        logger.info("="*80)
        
        if not all([UPSTOX_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DEEPSEEK_API_KEY]):
            logger.error("❌ Missing credentials!")
            return
        
        await self.notifier.send_startup(self.redis.connected)
        
        logger.info("="*80)
        logger.info(f"🟢 RUNNING | Redis: {self.redis.connected}")
        logger.info("="*80)
        
        while True:
            try:
                if not self.is_market_open():
                    logger.info("⏸️ Market closed. Waiting...")
                    await asyncio.sleep(300)
                    continue
                
                await self.run_scan()
                
                logger.info(f"⏳ Next scan in 15 min...")
                await asyncio.sleep(SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopped")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                traceback.print_exc()
                await asyncio.sleep(60)

# ==================== ENTRY POINT ====================
async def main():
    try:
        bot = HybridBot()
        await bot.run()
    except Exception as e:
        logger.error(f"Fatal: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("STARTING HYBRID BOT v31.0 - COMPACT PROFESSIONAL")
    logger.info("="*80)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n✅ Shutdown complete")
    except Exception as e:
        logger.error(f"\n❌ Critical: {e}")
        traceback.print_exc()
