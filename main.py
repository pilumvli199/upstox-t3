#!/usr/bin/env python3
"""
HYBRID TRADING BOT v30.0 - ULTIMATE EDITION
============================================
✅ MULTI-TIMEFRAME: 5m/15m/1h (from Code 1)
✅ DEEPSEEK V3 AI: 20+ patterns, confluence-based (from Code 2)
✅ FIXED DATA FETCHING: Robust retry logic with proper URL encoding
✅ SMART EXPIRY: API-first with calculated fallback
✅ 400+ CANDLES: Historical + Intraday combined
✅ REDIS OI TRACKING: 2-hour comparison
✅ NEWS INTEGRATION: Finnhub sentiment analysis
✅ PROFESSIONAL CHARTS: Clean candlesticks with levels
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
from typing import Dict, List, Optional, Tuple
import traceback
import re

# Redis with fallback
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - running without OI tracking")

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('hybrid_bot_v30.log')]
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
SCAN_INTERVAL = 900  # 15 minutes
REDIS_EXPIRY = 86400  # 24 hours

# ==================== SYMBOLS CONFIGURATION ====================
INDICES = {
    "NSE_INDEX|Nifty 50": {"name": "NIFTY", "display_name": "NIFTY 50", "expiry_day": 3},
    "NSE_INDEX|Nifty Bank": {"name": "BANKNIFTY", "display_name": "BANK NIFTY", "expiry_day": 2},
    "NSE_INDEX|NIFTY MID SELECT": {"name": "MIDCPNIFTY", "display_name": "MIDCAP NIFTY", "expiry_day": 0},
    "BSE_INDEX|SENSEX": {"name": "SENSEX", "display_name": "SENSEX", "expiry_day": 4}
}

FO_STOCKS = {
    # Auto
    "NSE_EQ|INE467B01029": {"name": "TATAMOTORS", "display_name": "TATA MOTORS"},
    "NSE_EQ|INE585B01010": {"name": "MARUTI", "display_name": "MARUTI SUZUKI"},
    "NSE_EQ|INE101A01026": {"name": "M&M", "display_name": "MAHINDRA & MAHINDRA"},
    "NSE_EQ|INE917I01010": {"name": "BAJAJ-AUTO", "display_name": "BAJAJ AUTO"},
    
    # Banks
    "NSE_EQ|INE040A01034": {"name": "HDFCBANK", "display_name": "HDFC BANK"},
    "NSE_EQ|INE090A01021": {"name": "ICICIBANK", "display_name": "ICICI BANK"},
    "NSE_EQ|INE062A01020": {"name": "SBIN", "display_name": "STATE BANK"},
    "NSE_EQ|INE238A01034": {"name": "AXISBANK", "display_name": "AXIS BANK"},
    "NSE_EQ|INE237A01028": {"name": "KOTAKBANK", "display_name": "KOTAK BANK"},
    
    # IT
    "NSE_EQ|INE009A01021": {"name": "INFY", "display_name": "INFOSYS"},
    "NSE_EQ|INE075A01022": {"name": "WIPRO", "display_name": "WIPRO"},
    "NSE_EQ|INE854D01024": {"name": "TCS", "display_name": "TCS"},
    "NSE_EQ|INE047A01021": {"name": "HCLTECH", "display_name": "HCL TECH"},
    
    # Others
    "NSE_EQ|INE002A01018": {"name": "RELIANCE", "display_name": "RELIANCE IND"},
    "NSE_EQ|INE397D01024": {"name": "BHARTIARTL", "display_name": "BHARTI AIRTEL"},
    "NSE_EQ|INE296A01024": {"name": "BAJFINANCE", "display_name": "BAJAJ FINANCE"},
    "NSE_EQ|INE044A01036": {"name": "SUNPHARMA", "display_name": "SUN PHARMA"},
    "NSE_EQ|INE154A01025": {"name": "ITC", "display_name": "ITC LTD"}
}

ALL_SYMBOLS = {**INDICES, **FO_STOCKS}

# Thresholds
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
    ce_iv: float = 0.0
    pe_iv: float = 0.0

@dataclass
class OIData:
    pcr: float
    support_strike: int
    resistance_strike: int
    strikes_data: List[StrikeData]
    timestamp: datetime
    ce_oi_change_pct: float = 0.0
    pe_oi_change_pct: float = 0.0
    ce_volume_change_pct: float = 0.0
    pe_volume_change_pct: float = 0.0
    overall_sentiment: str = "NEUTRAL"

@dataclass
class NewsData:
    headline: str
    sentiment: str
    impact_score: int
    source: str
    datetime_ts: int

@dataclass
class MultiTimeframeData:
    df_5m: pd.DataFrame
    df_15m: pd.DataFrame
    df_1h: pd.DataFrame
    current_price: float
    trend_1h: str
    pattern_15m: str
    entry_5m: float

@dataclass
class AIAnalysis:
    opportunity: str
    confidence: int
    chart_score: int
    oi_score: int
    news_score: int
    alignment_score: int
    total_score: int
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: str
    recommended_strike: int
    chart_bias: str
    market_structure: str
    pattern_signal: str
    oi_flow_signal: str
    support_levels: List[float]
    resistance_levels: List[float]
    risk_factors: List[str]
    monitoring_checklist: List[str]
    tf_1h_trend: str
    tf_15m_pattern: str
    tf_5m_entry: float
    tf_alignment: str
    ai_reasoning: str

# ==================== REDIS MANAGER ====================
class RedisCache:
    def __init__(self):
        self.redis_client = None
        self.connected = False
        
        if not REDIS_AVAILABLE:
            logger.warning("⚠️ Redis module not installed")
            return
        
        try:
            logger.info("🔄 Connecting to Redis...")
            self.redis_client = redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                retry_on_timeout=True
            )
            self.redis_client.ping()
            self.connected = True
            logger.info("✅ Redis connected successfully!")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None
            self.connected = False
    
    def save_oi(self, symbol: str, expiry: str, oi_data: OIData):
        if not self.redis_client or not self.connected:
            return
        try:
            key = f"oi:{symbol}:{expiry}:{oi_data.timestamp.strftime('%Y-%m-%d_%H:%M')}"
            data = {
                "pcr": oi_data.pcr,
                "support": oi_data.support_strike,
                "resistance": oi_data.resistance_strike,
                "ce_oi_change_pct": oi_data.ce_oi_change_pct,
                "pe_oi_change_pct": oi_data.pe_oi_change_pct,
                "sentiment": oi_data.overall_sentiment,
                "strikes": [
                    {
                        "strike": s.strike,
                        "ce_oi": s.ce_oi,
                        "pe_oi": s.pe_oi,
                        "ce_volume": s.ce_volume,
                        "pe_volume": s.pe_volume
                    } for s in oi_data.strikes_data
                ]
            }
            self.redis_client.setex(key, REDIS_EXPIRY, json.dumps(data))
            logger.info(f"  💾 Redis: OI saved for {symbol}")
        except Exception as e:
            logger.error(f"  ❌ Redis save error: {e}")
    
    def get_comparison_oi(self, symbol: str, expiry: str, current_time: datetime) -> Optional[OIData]:
        if not self.redis_client or not self.connected:
            return None
        
        try:
            two_hours_ago = current_time - timedelta(hours=2)
            comparison_time = two_hours_ago.replace(
                minute=(two_hours_ago.minute // 15) * 15,
                second=0,
                microsecond=0
            )
            
            key = f"oi:{symbol}:{expiry}:{comparison_time.strftime('%Y-%m-%d_%H:%M')}"
            data = self.redis_client.get(key)
            
            if data:
                parsed = json.loads(data)
                logger.info(f"  ⏰ OI comparison: 2h ago ({comparison_time.strftime('%H:%M')})")
                return OIData(
                    pcr=parsed['pcr'],
                    support_strike=parsed['support'],
                    resistance_strike=parsed['resistance'],
                    ce_oi_change_pct=parsed.get('ce_oi_change_pct', 0),
                    pe_oi_change_pct=parsed.get('pe_oi_change_pct', 0),
                    overall_sentiment=parsed.get('sentiment', 'NEUTRAL'),
                    strikes_data=[
                        StrikeData(
                            strike=s['strike'],
                            ce_oi=s['ce_oi'],
                            pe_oi=s['pe_oi'],
                            ce_volume=s.get('ce_volume', 0),
                            pe_volume=s.get('pe_volume', 0)
                        ) for s in parsed['strikes']
                    ],
                    timestamp=comparison_time
                )
            else:
                logger.info(f"  ⚠️ No OI data from 2h ago, using fresh baseline")
                return None
        except Exception as e:
            logger.error(f"  ❌ Redis get error: {e}")
            return None

# ==================== EXPIRY CALCULATOR (FIXED) ====================
class ExpiryCalculator:
    @staticmethod
    def get_all_expiries_from_api(instrument_key: str) -> List[str]:
        """✅ FIXED: Robust API expiry fetching"""
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
                if expiries:
                    logger.info(f"     API returned {len(expiries)} expiries")
                return expiries
            else:
                logger.warning(f"     API expiry fetch failed: {response.status_code}")
            return []
        except Exception as e:
            logger.warning(f"     API expiry error: {e}")
            return []
    
    @staticmethod
    def calculate_monthly_expiry(symbol_name: str, expiry_day: int = 3) -> str:
        """Calculate monthly expiry - Returns YYYY-MM-DD"""
        today = datetime.now(IST).date()
        current_time = datetime.now(IST).time()
        
        # Get last day of current month
        last_day = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        
        # Calculate expiry (last occurrence of target weekday)
        days_to_subtract = (last_day.weekday() - expiry_day) % 7
        expiry = last_day - timedelta(days=days_to_subtract)
        
        # If expiry passed, calculate next month
        if expiry < today or (expiry == today and current_time >= time(15, 30)):
            next_month = (today.replace(day=28) + timedelta(days=4))
            last_day = (next_month.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            days_to_subtract = (last_day.weekday() - expiry_day) % 7
            expiry = last_day - timedelta(days=days_to_subtract)
        
        return expiry.strftime('%Y-%m-%d')
    
    @staticmethod
    def get_best_expiry(instrument_key: str, symbol_info: Dict) -> str:
        """✅ SMART: API-first, fallback to calculation"""
        symbol_name = symbol_info.get('name', '')
        expiry_day = symbol_info.get('expiry_day', 3)
        
        # Try API first
        expiries = ExpiryCalculator.get_all_expiries_from_api(instrument_key)
        
        if expiries:
            today = datetime.now(IST).date()
            now_time = datetime.now(IST).time()
            
            # Filter future expiries
            future_expiries = []
            for exp_str in expiries:
                try:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                    if exp_date > today or (exp_date == today and now_time < time(15, 30)):
                        future_expiries.append(exp_str)
                except:
                    continue
            
            if future_expiries:
                selected = min(future_expiries)
                logger.info(f"     ✅ Using API expiry: {selected}")
                return selected
        
        # Fallback to calculation
        calculated = ExpiryCalculator.calculate_monthly_expiry(symbol_name, expiry_day)
        logger.info(f"     📅 Using calculated expiry: {calculated}")
        return calculated
    
    @staticmethod
    def get_display_expiry(expiry_str: str) -> str:
        """Convert YYYY-MM-DD to DDMmmYY"""
        try:
            dt = datetime.strptime(expiry_str, '%Y-%m-%d')
            return dt.strftime('%d%b%y').upper()
        except:
            return expiry_str

# ==================== DATA FETCHER (FIXED & ENHANCED) ====================
class UpstoxDataFetcher:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
            "Accept": "application/json"
        }
    
    def get_spot_price(self, instrument_key: str) -> float:
        """Fetch current LTP with retry"""
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
                logger.error(f"     LTP error (attempt {attempt+1}): {e}")
                time_sleep.sleep(2)
        
        return 0.0
    
    def get_option_chain(self, instrument_key: str, expiry: str) -> List[StrikeData]:
        """✅ FIXED: Robust option chain fetching with retry"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                encoded_key = urllib.parse.quote(instrument_key, safe='')
                url = f"{BASE_URL}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
                
                response = requests.get(url, headers=self.headers, timeout=20)
                
                logger.info(f"     Option chain: {response.status_code} (Attempt {attempt+1}/{max_retries})")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'data' not in data:
                        logger.warning(f"     ⚠️ No 'data' key in response")
                        if attempt < max_retries - 1:
                            time_sleep.sleep(retry_delay * (attempt + 1))
                            continue
                        return []
                    
                    strikes_raw = data.get('data', [])
                    
                    if not strikes_raw:
                        logger.warning(f"     ⚠️ Empty strikes array")
                        if attempt < max_retries - 1:
                            time_sleep.sleep(retry_delay * (attempt + 1))
                            continue
                        return []
                    
                    # Parse strikes
                    strikes = []
                    for item in strikes_raw:
                        call_data = item.get('call_options', {}).get('market_data', {})
                        put_data = item.get('put_options', {}).get('market_data', {})
                        
                        ce_oi = call_data.get('oi', 0)
                        pe_oi = put_data.get('oi', 0)
                        
                        # Skip if no OI
                        if ce_oi == 0 and pe_oi == 0:
                            continue
                        
                        strikes.append(StrikeData(
                            strike=int(item.get('strike_price', 0)),
                            ce_oi=ce_oi,
                            pe_oi=pe_oi,
                            ce_volume=call_data.get('volume', 0),
                            pe_volume=put_data.get('volume', 0),
                            ce_iv=call_data.get('iv', 0.0),
                            pe_iv=put_data.get('iv', 0.0)
                        ))
                    
                    if strikes:
                        logger.info(f"     ✅ Parsed {len(strikes)} strikes with OI data")
                        return strikes
                    else:
                        logger.warning(f"     ⚠️ No strikes with OI > 0")
                        if attempt < max_retries - 1:
                            time_sleep.sleep(retry_delay * (attempt + 1))
                            continue
                        return []
                
                elif response.status_code == 429:
                    wait_time = retry_delay * (attempt + 2)
                    logger.warning(f"     ⚠️ Rate limit! Waiting {wait_time}s...")
                    time_sleep.sleep(wait_time)
                    continue
                
                elif response.status_code == 404:
                    logger.warning(f"     ⚠️ No options for expiry {expiry}")
                    return []
                
                else:
                    logger.error(f"     ❌ API Error: {response.status_code}")
                    if attempt < max_retries - 1:
                        time_sleep.sleep(retry_delay * (attempt + 1))
                        continue
                    return []
                
            except Exception as e:
                logger.error(f"     ❌ Exception (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time_sleep.sleep(retry_delay * (attempt + 1))
                    continue
                return []
        
        return []
    
    def get_multi_timeframe_data(self, instrument_key: str, symbol: str) -> Optional[MultiTimeframeData]:
        """✅ ENHANCED: Fetch 400+ candles and create 3 timeframes"""
        try:
            encoded_key = urllib.parse.quote(instrument_key, safe='')
            all_candles = []
            
            # 1️⃣ Historical data (30min, last 10 days)
            try:
                to_date = (datetime.now(IST) - timedelta(days=1)).strftime('%Y-%m-%d')
                from_date = (datetime.now(IST) - timedelta(days=10)).strftime('%Y-%m-%d')
                url = f"{BASE_URL}/v2/historical-candle/{encoded_key}/30minute/{to_date}/{from_date}"
                
                response = requests.get(url, headers=self.headers, timeout=20)
                
                if response.status_code == 200 and response.json().get('status') == 'success':
                    candles_30min = response.json().get('data', {}).get('candles', [])
                    all_candles.extend(candles_30min)
                    logger.info(f"  📊 Historical 30m: {len(candles_30min)} candles")
            except Exception as e:
                logger.error(f"  Historical error: {e}")
            
            # 2️⃣ Intraday data (1min, today)
            try:
                url = f"{BASE_URL}/v2/historical-candle/intraday/{encoded_key}/1minute"
                
                response = requests.get(url, headers=self.headers, timeout=20)
                
                if response.status_code == 200 and response.json().get('status') == 'success':
                    candles_1min = response.json().get('data', {}).get('candles', [])
                    all_candles.extend(candles_1min)
                    logger.info(f"  📊 Intraday 1m: {len(candles_1min)} candles")
            except Exception as e:
                logger.error(f"  Intraday error: {e}")
            
            if not all_candles:
                logger.warning(f"  ❌ No candle data for {symbol}")
                return None
            
            # 3️⃣ Convert to DataFrame
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').astype(float)
            df = df.sort_index()
            
            logger.info(f"  📊 Total candles: {len(df)}")
            
            # 4️⃣ Resample to 3 timeframes
            df_5m = df.resample('5min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'oi': 'last'
            }).dropna()
            
            df_15m = df.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'oi': 'last'
            }).dropna()
            
            df_1h = df.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'oi': 'last'
            }).dropna()
            
            logger.info(f"  📊 Resampled: 5m={len(df_5m)}, 15m={len(df_15m)}, 1h={len(df_1h)}")
            
            current_price = df_15m['close'].iloc[-1] if len(df_15m) > 0 else 0
            
            # Quick 1h trend
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
                pattern_15m="ANALYZING",
                entry_5m=current_price
            )
            
        except Exception as e:
            logger.error(f"Multi-TF data error: {e}")
            traceback.print_exc()
            return None

# ==================== NEWS FETCHER ====================
class NewsFetcher:
    @staticmethod
    def fetch_finnhub_news(symbol_name: str) -> Optional[NewsData]:
        """Fetch latest news from Finnhub"""
        if not FINNHUB_API_KEY:
            return None
        
        try:
            today = datetime.now(IST).date()
            yesterday = today - timedelta(days=1)
            
            response = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": symbol_name,
                    "from": yesterday.strftime('%Y-%m-%d'),
                    "to": today.strftime('%Y-%m-%d'),
                    "token": FINNHUB_API_KEY
                },
                timeout=10
            )
            
            if response.status_code == 200:
                news_list = response.json()
                if news_list:
                    latest = news_list[0]
                    sentiment = latest.get('sentiment', 'neutral').upper()
                    
                    if sentiment == 'POSITIVE':
                        impact = 10
                    elif sentiment == 'NEGATIVE':
                        impact = -10
                    else:
                        impact = 0
                    
                    return NewsData(
                        headline=latest.get('headline', 'No headline')[:150],
                        sentiment=sentiment,
                        impact_score=impact,
                        source='Finnhub',
                        datetime_ts=latest.get('datetime', 0)
                    )
        except Exception as e:
            logger.error(f"  📰 News fetch error: {e}")
        
        return None

# ==================== CHART ANALYZER ====================
class ChartAnalyzer:
    @staticmethod
    def analyze_1h_trend(df_1h: pd.DataFrame) -> Dict:
        """1H Trend Analysis"""
        try:
            if len(df_1h) < 20:
                return {"trend": "NEUTRAL", "strength": 0, "bias": "NONE"}
            
            recent = df_1h.tail(50)
            current = recent['close'].iloc[-1]
            
            ma20 = recent['close'].rolling(20).mean().iloc[-1]
            ma50 = recent['close'].rolling(50).mean().iloc[-1] if len(recent) >= 50 else ma20
            
            if current > ma20 > ma50:
                trend = "BULLISH"
                strength = 80
            elif current < ma20 < ma50:
                trend = "BEARISH"
                strength = 80
            elif current > ma20:
                trend = "BULLISH"
                strength = 60
            elif current < ma20:
                trend = "BEARISH"
                strength = 60
            else:
                trend = "NEUTRAL"
                strength = 40
            
            return {
                "trend": trend,
                "strength": strength,
                "bias": "LONG" if trend == "BULLISH" else "SHORT" if trend == "BEARISH" else "NONE",
                "ma20": ma20,
                "current": current
            }
        except:
            return {"trend": "NEUTRAL", "strength": 0, "bias": "NONE"}
    
    @staticmethod
    def analyze_15m_patterns(df_15m: pd.DataFrame) -> Dict:
        """15M Pattern Detection"""
        try:
            if len(df_15m) < 30:
                return {"pattern": "NONE", "signal": "NEUTRAL", "confidence": 0}
            
            recent = df_15m.tail(100)
            last_20 = recent.tail(20)
            patterns_found = []
            
            # Bullish Engulfing
            for i in range(1, len(last_20)):
                prev = last_20.iloc[i-1]
                curr = last_20.iloc[i]
                
                if (prev['close'] < prev['open'] and
                    curr['close'] > curr['open'] and
                    curr['open'] < prev['close'] and
                    curr['close'] > prev['open']):
                    patterns_found.append("BULLISH_ENGULFING")
            
            # Bearish Engulfing
            for i in range(1, len(last_20)):
                prev = last_20.iloc[i-1]
                curr = last_20.iloc[i]
                
                if (prev['close'] > prev['open'] and
                    curr['close'] < curr['open'] and
                    curr['open'] > prev['close'] and
                    curr['close'] < prev['open']):
                    patterns_found.append("BEARISH_ENGULFING")
            
            # Hammer/Doji
            last_candle = last_20.iloc[-1]
            body = abs(last_candle['close'] - last_candle['open'])
            total_range = last_candle['high'] - last_candle['low']
            
            if total_range > 0:
                if body / total_range < 0.1:
                    patterns_found.append("DOJI")
                elif (last_candle['low'] < min(last_candle['open'], last_candle['close']) - body * 2):
                    patterns_found.append("HAMMER")
            
            # Breakout detection
            high_20 = recent['high'].rolling(20).max().iloc[-1]
            low_20 = recent['low'].rolling(20).min().iloc[-1]
            current = recent['close'].iloc[-1]
            
            if current > high_20 * 0.999:
                patterns_found.append("BREAKOUT_HIGH")
            elif current < low_20 * 1.001:
                patterns_found.append("BREAKDOWN_LOW")
            
            # Signal determination
            bullish_patterns = ["BULLISH_ENGULFING", "HAMMER", "BREAKOUT_HIGH"]
            bearish_patterns = ["BEARISH_ENGULFING", "BREAKDOWN_LOW"]
            
            bullish_count = sum(1 for p in patterns_found if p in bullish_patterns)
            bearish_count = sum(1 for p in patterns_found if p in bearish_patterns)
            
            if bullish_count > bearish_count:
                signal = "BULLISH"
                confidence = min(bullish_count * 30, 90)
            elif bearish_count > bullish_count:
                signal = "BEARISH"
                confidence = min(bearish_count * 30, 90)
            else:
                signal = "NEUTRAL"
                confidence = 50
            
            pattern_str = ", ".join(patterns_found[:3]) if patterns_found else "NONE"
            
            return {
                "pattern": pattern_str,
                "signal": signal,
                "confidence": confidence,
                "patterns_found": patterns_found
            }
        except:
            return {"pattern": "NONE", "signal": "NEUTRAL", "confidence": 0}
    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame) -> Dict:
        """Calculate S/R from 15min data"""
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

# ==================== DATA COMPRESSOR ====================
class DataCompressor:
    @staticmethod
    def compress_to_ohlc(df: pd.DataFrame, limit: int = None) -> str:
        """Compress DataFrame to compact OHLC string"""
        if limit:
            df = df.tail(limit)
        
        ohlc_list = []
        for _, row in df.iterrows():
            ohlc_list.append(
                f"[O:{row['open']:.1f},H:{row['high']:.1f},L:{row['low']:.1f},"
                f"C:{row['close']:.1f},V:{int(row['volume'])}]"
            )
        
        return ','.join(ohlc_list)
    
    @staticmethod
    def compress_oi(current: OIData, prev: Optional[OIData]) -> str:
        """Compress OI data for AI prompt"""
        if not prev:
            return f"PCR:{current.pcr:.2f}|S:{current.support_strike}|R:{current.resistance_strike}"
        
        prev_map = {s.strike: s for s in prev.strikes_data}
        ce_builds, pe_builds = [], []
        
        for s in current.strikes_data[:10]:
            if s.strike in prev_map:
                ps = prev_map[s.strike]
                ce_chg = (s.ce_oi - ps.ce_oi) / ps.ce_oi if ps.ce_oi > 0 else 0
                pe_chg = (s.pe_oi - ps.pe_oi) / ps.pe_oi if ps.pe_oi > 0 else 0
                
                if ce_chg > 0.15:
                    ce_builds.append(s.strike)
                if pe_chg > 0.15:
                    pe_builds.append(s.strike)
        
        result = f"PCR:{current.pcr:.2f}|CE:{current.ce_oi_change_pct:+.1f}%|PE:{current.pe_oi_change_pct:+.1f}%"
        if ce_builds:
            result += f"|CE_BUILD:{','.join(map(str, ce_builds[:2]))}"
        if pe_builds:
            result += f"|PE_BUILD:{','.join(map(str, pe_builds[:2]))}"
        
        return result

# ==================== DEEPSEEK AI ANALYZER (ENHANCED) ====================
class DeepSeekAnalyzer:
    @staticmethod
    def generate_multi_tf_prompt(symbol: str, mtf_data: MultiTimeframeData,
                                 spot_price: float, atr: float,
                                 current_oi: OIData, prev_oi: Optional[OIData],
                                 trend_1h: Dict, pattern_15m: Dict,
                                 sr_levels: Dict, news: Optional[NewsData]) -> str:
        """Generate comprehensive multi-timeframe prompt"""
        
        ohlc_1h = DataCompressor.compress_to_ohlc(mtf_data.df_1h, limit=50)
        ohlc_15m = DataCompressor.compress_to_ohlc(mtf_data.df_15m, limit=100)
        ohlc_5m = DataCompressor.compress_to_ohlc(mtf_data.df_5m, limit=50)
        oi_summary = DataCompressor.compress_oi(current_oi, prev_oi)
        
        news_section = ""
        if news:
            news_section = f"""
**NEWS CONTEXT:**
Headline: {news.headline}
Sentiment: {news.sentiment}
Impact: {'+' if news.impact_score > 0 else ''}{news.impact_score} points
"""
        
        prompt = f"""You are an expert F&O multi-timeframe trader. Analyze {symbol} using institutional confluence methodology.

**INSTRUMENT:** {symbol}
**SPOT PRICE:** ₹{spot_price:.2f}
**ATR (14):** {atr:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**MULTI-TIMEFRAME DATA (Compressed OHLC):**

**1H TIMEFRAME (Last 50 candles - Trend Filter):**
{ohlc_1h}
→ Current Trend: {trend_1h['trend']} (Strength: {trend_1h['strength']}%)
→ MA20: ₹{trend_1h.get('ma20', spot_price):.2f}

**15M TIMEFRAME (Last 100 candles - Main Analysis):**
{ohlc_15m}
→ Patterns: {pattern_15m['pattern']}
→ Signal: {pattern_15m['signal']} ({pattern_15m['confidence']}% conf)
→ Support: {', '.join([f"₹{s:.0f}" for s in sr_levels['supports'][:3]])}
→ Resistance: {', '.join([f"₹{r:.0f}" for r in sr_levels['resistances'][:3]])}

**5M TIMEFRAME (Last 50 candles - Entry Precision):**
{ohlc_5m}
→ Current: ₹{mtf_data.current_price:.2f}

**OPTIONS DATA (2-Hour Comparison):**
{oi_summary}
→ Support Zone: {current_oi.support_strike}
→ Resistance Zone: {current_oi.resistance_strike}
→ Sentiment: {current_oi.overall_sentiment}

{news_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**MULTI-TIMEFRAME CONFLUENCE ANALYSIS:**

**STEP 1: 1H TREND (Mandatory Alignment)**
- Analyze overall trend direction using HH/HL or LH/LL
- Check price vs MA20 position
- Trend strength assessment
- **CRITICAL: Only trade WITH 1H trend**

**STEP 2: 15M PATTERNS (Main Signal)**
- Chart patterns: Flags, Triangles, H&S, Double Top/Bottom
- Candlestick patterns: Engulfing, Hammer, Doji (with volume)
- Support/Resistance respect (clean bounces = high score)
- Breakout/Breakdown confirmation

**STEP 3: 5M ENTRY (Precision)**
- Use for exact entry timing
- Look for pullbacks to support in uptrend
- Or resistance test in downtrend

**STEP 4: OI FLOW CONFLUENCE**
**CRITICAL OI RULES:**
- CE Unwinding (-ve) = Resistance weakening = BULLISH
- PE Unwinding (-ve) = Support weakening = BEARISH
- CE Building (+ve) = Resistance strengthening = BEARISH
- PE Building (+ve) = Support strengthening = BULLISH

**PCR Interpretation:**
- PCR > 1.2 = Bullish (put writers confident)
- PCR < 0.8 = Bearish (call writers confident)
- PCR 0.8-1.2 = Neutral

**Pattern-OI Confluence (HIGH PROBABILITY):**
- Bullish 15M Pattern + CE Unwinding + 1H Uptrend = STRONG BUY
- Bearish 15M Pattern + PE Unwinding + 1H Downtrend = STRONG SELL
- Pattern without OI support = REJECT

**STEP 5: SCORING SYSTEM (/125 points)**

**CHART SCORE (0-50):**
- 1H Trend Clarity: 0-15 pts (Strong=15, Weak=7, Sideways=0)
- 15M Pattern Quality: 0-20 pts (Textbook=20, Partial=10, Weak=3)
- S/R Respect: 0-10 pts (Clean=10, Choppy=5)
- Volume Confirmation: 0-5 pts (High volume on key levels=5)

**OI SCORE (0-50):**
- PCR Signal: 0-10 pts (Extreme PCR=10, Neutral=5)
- OI Change Magnitude: 0-20 pts (>15%=20, 10-15%=15, 5-10%=10)
- Pattern-OI Confluence: 0-20 pts (Perfect=20, Partial=10, Mismatch=-10)

**TF ALIGNMENT SCORE (0-25):**
- 1H + 15M + 5M aligned = 25 pts
- 1H + 15M aligned only = 18 pts
- Only 15M signal = 10 pts
- Conflicting TFs = 0 pts

**MINIMUM THRESHOLDS FOR TRADE:**
- Total Score: ≥ 70/125
- Confidence: ≥ 75%
- TF Alignment: ≥ 18/25
- Risk:Reward: ≥ 1:2

**REJECTION CRITERIA (Return "WAIT"):**
- No clear 1H trend
- 1H vs 15M conflict
- Pattern without OI confirmation
- Score < 70
- Choppy/indecision candles

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**OUTPUT FORMAT (JSON ONLY):**

{{
  "opportunity": "CE_BUY/PE_BUY/WAIT",
  "confidence": 85,
  "chart_score": 42,
  "oi_score": 40,
  "news_score": 5,
  "alignment_score": 22,
  "total_score": 109,
  "entry_price": {spot_price:.2f},
  "stop_loss": 0.0,
  "target_1": 0.0,
  "target_2": 0.0,
  "risk_reward": "1:2.5",
  "recommended_strike": {int(spot_price)},
  "chart_bias": "Bullish/Bearish/Neutral",
  "market_structure": "HH/HL forming",
  "pattern_signal": "Bullish Flag + Volume spike",
  "oi_flow_signal": "CE Unwinding at resistance",
  "support_levels": {sr_levels['supports'][:3]},
  "resistance_levels": {sr_levels['resistances'][:3]},
  "risk_factors": ["Risk 1", "Risk 2", "Risk 3"],
  "monitoring_checklist": ["Monitor 1H trend", "Watch 15M S/R", "5M entry trigger"],
  "tf_1h_trend": "{trend_1h['trend']}",
  "tf_15m_pattern": "{pattern_15m['pattern']}",
  "tf_5m_entry": {mtf_data.current_price:.2f},
  "tf_alignment": "STRONG/MODERATE/WEAK",
  "ai_reasoning": "Detailed breakdown: Chart(42/50): 1H bullish(14) + 15M flag(18) + Clean S/R(8) + Volume(2). OI(40/50): PCR 1.25 bullish(9) + CE unwind 12%(18) + Pattern match(13). TF Alignment(22/25): All aligned. News(+5). Total: 109/125 = HIGH PROBABILITY"
}}

**BE BRUTALLY HONEST:**
- If TF not aligned, return "WAIT"
- If pattern weak without OI support, deduct points
- All targets must satisfy 1:2 minimum RR
- News contradicting setup = deduct 5 points
"""
        
        return prompt
    
    @staticmethod
    def parse_ai_response(content: str) -> Optional[AIAnalysis]:
        """Parse DeepSeek JSON response"""
        try:
            content = content.replace('```json', '').replace('```', '').strip()
            
            try:
                analysis = json.loads(content)
            except:
                match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', content, re.DOTALL)
                if match:
                    analysis = json.loads(match.group(0))
                else:
                    return None
            
            return AIAnalysis(
                opportunity=analysis.get('opportunity', 'WAIT'),
                confidence=analysis.get('confidence', 0),
                chart_score=analysis.get('chart_score', 0),
                oi_score=analysis.get('oi_score', 0),
                news_score=analysis.get('news_score', 0),
                alignment_score=analysis.get('alignment_score', 0),
                total_score=analysis.get('total_score', 0),
                entry_price=analysis.get('entry_price', 0),
                stop_loss=analysis.get('stop_loss', 0),
                target_1=analysis.get('target_1', 0),
                target_2=analysis.get('target_2', 0),
                risk_reward=analysis.get('risk_reward', '0:0'),
                recommended_strike=analysis.get('recommended_strike', 0),
                chart_bias=analysis.get('chart_bias', 'Neutral'),
                market_structure=analysis.get('market_structure', 'Unknown'),
                pattern_signal=analysis.get('pattern_signal', 'None'),
                oi_flow_signal=analysis.get('oi_flow_signal', 'Neutral'),
                support_levels=analysis.get('support_levels', []),
                resistance_levels=analysis.get('resistance_levels', []),
                risk_factors=analysis.get('risk_factors', []),
                monitoring_checklist=analysis.get('monitoring_checklist', []),
                tf_1h_trend=analysis.get('tf_1h_trend', 'NEUTRAL'),
                tf_15m_pattern=analysis.get('tf_15m_pattern', 'NONE'),
                tf_5m_entry=analysis.get('tf_5m_entry', 0),
                tf_alignment=analysis.get('tf_alignment', 'WEAK'),
                ai_reasoning=analysis.get('ai_reasoning', 'No reasoning')
            )
        except Exception as e:
            logger.error(f"AI parse error: {e}")
            return None
    
    @staticmethod
    def analyze(symbol: str, mtf_data: MultiTimeframeData, spot_price: float,
               atr: float, current_oi: OIData, prev_oi: Optional[OIData],
               trend_1h: Dict, pattern_15m: Dict, sr_levels: Dict,
               news: Optional[NewsData]) -> Optional[AIAnalysis]:
        """Call DeepSeek V3 API"""
        try:
            prompt = DeepSeekAnalyzer.generate_multi_tf_prompt(
                symbol, mtf_data, spot_price, atr, current_oi, prev_oi,
                trend_1h, pattern_15m, sr_levels, news
            )
            
            logger.info(f"  🤖 Calling DeepSeek V3...")
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "Expert F&O trader. Respond ONLY in valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 3000
                },
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=120
            )
            
            if response.status_code != 200:
                logger.error(f"  ❌ DeepSeek error: {response.status_code}")
                return None
            
            ai_content = response.json()['choices'][0]['message']['content']
            analysis = DeepSeekAnalyzer.parse_ai_response(ai_content)
            
            if analysis:
                logger.info(f"  🤖 AI: {analysis.opportunity} | Score: {analysis.total_score}/125 | Conf: {analysis.confidence}%")
                logger.info(f"     Chart:{analysis.chart_score} OI:{analysis.oi_score} Align:{analysis.alignment_score} News:{analysis.news_score}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"  ❌ DeepSeek error: {e}")
            traceback.print_exc()
            return None

# ==================== CHART GENERATOR ====================
class ChartGenerator:
    @staticmethod
    def create_professional_chart(symbol: str, df: pd.DataFrame, analysis: AIAnalysis,
                                  spot: float, oi_data: OIData, path: str):
        """Generate professional trading chart"""
        BG, GRID, TEXT = '#131722', '#1e222d', '#d1d4dc'
        GREEN, RED, YELLOW = '#26a69a', '#ef5350', '#ffd700'
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       facecolor=BG)
        
        ax1.set_facecolor(BG)
        df_plot = df.tail(150).reset_index(drop=True)
        
        # Candlesticks
        for idx, row in df_plot.iterrows():
            color = GREEN if row['close'] > row['open'] else RED
            ax1.add_patch(Rectangle(
                (idx, min(row['open'], row['close'])),
                0.6,
                abs(row['close'] - row['open']),
                facecolor=color,
                alpha=0.8
            ))
            ax1.plot([idx+0.3, idx+0.3], [row['low'], row['high']],
                    color=color, linewidth=1, alpha=0.6)
        
        # Support levels
        for sup in analysis.support_levels[:3]:
            if sup > 0:
                ax1.axhline(sup, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.7)
                ax1.text(len(df_plot)*0.02, sup, f'S: ₹{sup:.1f}',
                        color=GREEN, fontsize=9,
                        bbox=dict(boxstyle='round', facecolor=BG, alpha=0.7))
        
        # Resistance levels
        for res in analysis.resistance_levels[:3]:
            if res > 0:
                ax1.axhline(res, color=RED, linestyle='--', linewidth=1.5, alpha=0.7)
                ax1.text(len(df_plot)*0.02, res, f'R: ₹{res:.1f}',
                        color=RED, fontsize=9,
                        bbox=dict(boxstyle='round', facecolor=BG, alpha=0.7))
        
        # Trade levels
        if analysis.opportunity != "WAIT":
            ax1.scatter([len(df_plot)-1], [analysis.entry_price],
                       color=YELLOW, s=300, marker='D', zorder=5,
                       edgecolors='white', linewidths=2)
            ax1.axhline(analysis.stop_loss, color=RED, linewidth=2.5, linestyle=':')
            ax1.axhline(analysis.target_1, color=GREEN, linewidth=2, linestyle=':')
            ax1.axhline(analysis.target_2, color=GREEN, linewidth=1.5, linestyle=':')
        
        # Info box
        score_color = GREEN if analysis.total_score >= 85 else (YELLOW if analysis.total_score >= 70 else RED)
        signal_emoji = "🟢" if analysis.opportunity == "CE_BUY" else ("🔴" if analysis.opportunity == "PE_BUY" else "⏸️")
        
        info = f"""{signal_emoji} {analysis.opportunity}

SCORE: {analysis.total_score}/125
├─ Chart: {analysis.chart_score}/50
├─ OI: {analysis.oi_score}/50
├─ Align: {analysis.alignment_score}/25
└─ News: {analysis.news_score}/10

Confidence: {analysis.confidence}%
TF: {analysis.tf_alignment}

1H: {analysis.tf_1h_trend}
15M: {analysis.tf_15m_pattern[:20]}
5M: ₹{analysis.tf_5m_entry:.1f}

OI: PCR {oi_data.pcr:.2f}
Sentiment: {oi_data.overall_sentiment}

Entry: ₹{analysis.entry_price:.1f}
SL: ₹{analysis.stop_loss:.1f}
T1: ₹{analysis.target_1:.1f}
T2: ₹{analysis.target_2:.1f}
R:R: {analysis.risk_reward}"""
        
        ax1.text(0.01, 0.99, info, transform=ax1.transAxes,
                fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor=GRID, alpha=0.95,
                         edgecolor=score_color, linewidth=2),
                color=TEXT, family='monospace')
        
        score_emoji = "🔥" if analysis.total_score >= 85 else ("✅" if analysis.total_score >= 70 else "⚠️")
        title = f"{score_emoji} {symbol} | 15M | Score: {analysis.total_score}/125 | {analysis.pattern_signal[:40]}"
        
        ax1.set_title(title, color=TEXT, fontsize=13, fontweight='bold', pad=15)
        ax1.grid(True, color=GRID, alpha=0.3)
        ax1.tick_params(colors=TEXT)
        ax1.set_ylabel('Price (₹)', color=TEXT, fontsize=11)
        
        # Volume
        ax2.set_facecolor(BG)
        colors = [GREEN if df_plot.iloc[i]['close'] > df_plot.iloc[i]['open'] else RED
                 for i in range(len(df_plot))]
        ax2.bar(range(len(df_plot)), df_plot['volume'], color=colors, alpha=0.6)
        ax2.set_ylabel('Volume', color=TEXT, fontsize=10)
        ax2.tick_params(colors=TEXT)
        ax2.grid(True, color=GRID, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, facecolor=BG)
        plt.close()
        logger.info(f"  📊 Chart saved: {path}")

# ==================== TELEGRAM NOTIFIER ====================
class TelegramNotifier:
    def __init__(self):
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
    
    async def send_startup_message(self, redis_connected: bool):
        """Send bot startup notification"""
        redis_status = "🟢 Connected" if redis_connected else "🔴 Disconnected"
        
        message = f"""
🚀 **HYBRID BOT v30.0 - ULTIMATE EDITION**

⏰ **Started:** {datetime.now(IST).strftime('%d-%b-%Y %H:%M:%S IST')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 **STRATEGY FEATURES**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Multi-Timeframe: 1H + 15M + 5M
✅ 400+ Candles (Historical + Intraday)
✅ DeepSeek V3 AI (20+ Patterns)
✅ Confluence-Based Scoring
✅ OI Flow Analysis (2h comparison)
✅ News Integration (Finnhub)
✅ Professional Charts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **SCORING SYSTEM**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Chart Analysis: /50
OI Analysis: /50
TF Alignment: /25
News Impact: /10
────────────────
**Total: /125**

**Minimum Threshold: 70/125**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 **SYSTEM STATUS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📦 Redis: {redis_status}
🔄 Scan Interval: 15 minutes
📊 Monitoring: {len(ALL_SYMBOLS)} symbols
⏰ Market Hours: 09:15 - 15:30 IST

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🟢 **BOT ACTIVE & MONITORING**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        await self.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message,
            parse_mode='Markdown'
        )
        logger.info("✅ Startup message sent")
    
    async def send_alert(self, symbol: str, display_name: str, analysis: AIAnalysis,
                        oi_data: OIData, chart_path: str, news: Optional[NewsData],
                        expiry: str):
        """Send comprehensive trading alert"""
        try:
            # Send chart first
            with open(chart_path, 'rb') as photo:
                await self.bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=photo
                )
            
            # Calculate R:R
            risk = abs(analysis.entry_price - analysis.stop_loss)
            reward = abs(analysis.target_1 - analysis.entry_price)
            rr = reward / risk if risk > 0 else 0
            
            signal_emoji = "🟢" if analysis.opportunity == "CE_BUY" else "🔴"
            score_emoji = "🔥" if analysis.total_score >= 85 else "✅"
            
            message = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{signal_emoji} **{display_name} {analysis.opportunity}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{score_emoji} **SCORE: {analysis.total_score}/125**

📊 **Breakdown:**
├─ Chart: **{analysis.chart_score}/50**
├─ OI: **{analysis.oi_score}/50**
├─ TF Alignment: **{analysis.alignment_score}/25**
└─ News: **{analysis.news_score}/10**

**Confidence: {analysis.confidence}%**
**TF Alignment: {analysis.tf_alignment}**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 **MULTI-TIMEFRAME VIEW**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**1H Trend:** {analysis.tf_1h_trend}
**15M Pattern:** {analysis.tf_15m_pattern}
**5M Entry:** ₹{analysis.tf_5m_entry:.2f}

**Market Structure:** {analysis.market_structure}
**Chart Bias:** {analysis.chart_bias}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⛓️ **OPTIONS ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**PCR:** {oi_data.pcr:.2f}
**Sentiment:** {oi_data.overall_sentiment}

**CE OI Change:** {oi_data.ce_oi_change_pct:+.1f}%
**PE OI Change:** {oi_data.pe_oi_change_pct:+.1f}%

**Support Strike:** {oi_data.support_strike}
**Resistance Strike:** {oi_data.resistance_strike}

**Signal:** {analysis.oi_flow_signal}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 **TRADE SETUP**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Entry:** ₹{analysis.entry_price:.2f}
**Stop Loss:** ₹{analysis.stop_loss:.2f}
**Target 1:** ₹{analysis.target_1:.2f} 🎯
**Target 2:** ₹{analysis.target_2:.2f} 🎯🎯

**Risk:Reward:** 1:{rr:.1f}
**Risk Amount:** ₹{risk:.2f}
**Potential Reward:** ₹{reward:.2f}

**Recommended Strike:** {analysis.recommended_strike}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **SUPPORT & RESISTANCE**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Support Levels:**"""
            
            for i, sup in enumerate(analysis.support_levels[:3], 1):
                message += f"\n{i}. ₹{sup:.2f}"
            
            message += "\n\n**Resistance Levels:**"
            for i, res in enumerate(analysis.resistance_levels[:3], 1):
                message += f"\n{i}. ₹{res:.2f}"
            
            message += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 **AI REASONING**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{analysis.ai_reasoning}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ **RISK FACTORS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            for i, risk in enumerate(analysis.risk_factors[:3], 1):
                message += f"\n{i}. {risk}"
            
            message += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ **MONITORING CHECKLIST**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            for i, check in enumerate(analysis.monitoring_checklist[:3], 1):
                message += f"\n{i}. {check}"
            
            if news:
                message += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📰 **NEWS UPDATE** ({news.source})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Headline:** {news.headline}

**Sentiment:** {news.sentiment}
**Impact:** {'+' if news.impact_score > 0 else ''}{news.impact_score} points
"""
            
            message += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 **Expiry:** {expiry}
🕐 **Alert Time:** {datetime.now(IST).strftime('%d-%b-%Y %H:%M:%S IST')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='Markdown'
            )
            
            logger.info(f"  ✅ Alert sent for {display_name}")
            
        except Exception as e:
            logger.error(f"Telegram alert error: {e}")
            traceback.print_exc()
    
    async def send_cycle_summary(self, total: int, alerts: int):
        """Send scan cycle summary"""
        message = f"""
📊 **SCAN CYCLE COMPLETE**

Instruments Analyzed: {total}
Alerts Sent: {alerts}

Strategy: Multi-TF Confluence
Next Scan: 15 minutes

⏰ {datetime.now(IST).strftime('%H:%M:%S IST')}
"""
        
        await self.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message,
            parse_mode='Markdown'
        )

# ==================== MAIN BOT ====================
class HybridBot:
    def __init__(self):
        logger.info("🔄 Initializing Hybrid Bot v30.0...")
        
        self.fetcher = UpstoxDataFetcher()
        self.redis = RedisCache()
        self.notifier = TelegramNotifier()
        self.processed_signals = set()
        
        logger.info(f"✅ Bot initialized | Redis: {self.redis.connected}")
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now(IST)
        
        # Weekend check
        if now.weekday() >= 5:
            return False
        
        # Market hours
        current_time = now.time()
        return time(9, 15) <= current_time <= time(15, 30)
    
    async def analyze_symbol(self, instrument_key: str, symbol_info: Dict):
        """Deep multi-timeframe analysis"""
        try:
            symbol_name = symbol_info.get('name', '')
            display_name = symbol_info.get('display_name', symbol_name)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 {display_name} ({symbol_name})")
            logger.info(f"{'='*70}")
            
            # Get expiry
            expiry_api = ExpiryCalculator.get_best_expiry(instrument_key, symbol_info)
            expiry_display = ExpiryCalculator.get_display_expiry(expiry_api)
            
            logger.info(f"  📅 Expiry: {expiry_display} (API: {expiry_api})")
            
            # Get multi-timeframe data
            mtf_data = self.fetcher.get_multi_timeframe_data(instrument_key, symbol_name)
            if not mtf_data:
                logger.warning(f"  ❌ No multi-TF data")
                return
            
            logger.info(f"  📊 TF Data: 1H({len(mtf_data.df_1h)}) 15M({len(mtf_data.df_15m)}) 5M({len(mtf_data.df_5m)})")
            
            # Get spot price
            spot_price = self.fetcher.get_spot_price(instrument_key)
            if spot_price == 0:
                spot_price = mtf_data.current_price
            
            # Calculate ATR
            mtf_data.df_15m['tr'] = mtf_data.df_15m[['high', 'low', 'close']].apply(
                lambda x: max(x['high']-x['low'], abs(x['high']-x['close']), abs(x['low']-x['close'])),
                axis=1
            )
            atr = mtf_data.df_15m['tr'].rolling(14).mean().iloc[-1]
            
            logger.info(f"  💹 Spot: ₹{spot_price:.2f} | ATR: {atr:.2f}")
            
            # Get option chain
            all_strikes = self.fetcher.get_option_chain(instrument_key, expiry_api)
            
            if not all_strikes:
                logger.warning(f"  ⚠️ No option data - skipping")
                return
            
            # Filter ATM strikes
            atm = round(spot_price / 100) * 100
            atm_range = range(atm - 700, atm + 800, 100)
            top_15 = sorted(
                [s for s in all_strikes if s.strike in atm_range],
                key=lambda x: (x.ce_oi + x.pe_oi),
                reverse=True
            )[:15]
            
            if not top_15:
                logger.warning(f"  ⚠️ No ATM strikes")
                return
            
            logger.info(f"  📊 OI: {len(top_15)} ATM strikes")
            
            # Analyze OI
            total_ce = sum(s.ce_oi for s in top_15)
            total_pe = sum(s.pe_oi for s in top_15)
            pcr = total_pe / total_ce if total_ce > 0 else 0
            
            max_ce_strike = max(top_15, key=lambda x: x.ce_oi).strike
            max_pe_strike = max(top_15, key=lambda x: x.pe_oi).strike
            
            # Get comparison OI
            prev_oi = self.redis.get_comparison_oi(symbol_name, expiry_display, datetime.now(IST))
            
            ce_change_pct = 0.0
            pe_change_pct = 0.0
            if prev_oi:
                prev_ce = sum(s.ce_oi for s in prev_oi.strikes_data)
                prev_pe = sum(s.pe_oi for s in prev_oi.strikes_data)
                if prev_ce > 0:
                    ce_change_pct = ((total_ce - prev_ce) / prev_ce) * 100
                if prev_pe > 0:
                    pe_change_pct = ((total_pe - prev_pe) / prev_pe) * 100
            
            # Determine sentiment
            sentiment = "NEUTRAL"
            if pe_change_pct > 5 and pe_change_pct > ce_change_pct:
                sentiment = "BULLISH"
            elif ce_change_pct > 5 and ce_change_pct > pe_change_pct:
                sentiment = "BEARISH"
            elif pcr > 1.3:
                sentiment = "BULLISH"
            elif pcr < 0.7:
                sentiment = "BEARISH"
            
            current_oi = OIData(
                pcr=pcr,
                support_strike=max_pe_strike,
                resistance_strike=max_ce_strike,
                strikes_data=top_15,
                timestamp=datetime.now(IST),
                ce_oi_change_pct=ce_change_pct,
                pe_oi_change_pct=pe_change_pct,
                overall_sentiment=sentiment
            )
            
            logger.info(f"  📊 PCR: {pcr:.2f} | Sentiment: {sentiment}")
            logger.info(f"     CE: {ce_change_pct:+.1f}% | PE: {pe_change_pct:+.1f}%")
            
            # Save OI
            self.redis.save_oi(symbol_name, expiry_display, current_oi)
            
            # Timeframe analysis
            trend_1h = ChartAnalyzer.analyze_1h_trend(mtf_data.df_1h)
            pattern_15m = ChartAnalyzer.analyze_15m_patterns(mtf_data.df_15m)
            sr_levels = ChartAnalyzer.calculate_support_resistance(mtf_data.df_15m)
            
            logger.info(f"  🕐 1H: {trend_1h['trend']} ({trend_1h['strength']}%)")
            logger.info(f"  ⏰ 15M: {pattern_15m['signal']} | {pattern_15m['pattern']}")
            
            # Fetch news
            news_data = NewsFetcher.fetch_finnhub_news(symbol_name)
            if news_data:
                logger.info(f"  📰 {news_data.headline[:60]}... [{news_data.sentiment}]")
            
            # AI Analysis
            analysis = DeepSeekAnalyzer.analyze(
                display_name,
                mtf_data,
                spot_price,
                atr,
                current_oi,
                prev_oi,
                trend_1h,
                pattern_15m,
                sr_levels,
                news_data
            )
            
            if not analysis:
                logger.info(f"  ⏸️ No AI analysis")
                return
            
            # Check thresholds
            if analysis.opportunity == "WAIT":
                logger.info(f"  ⏸️ AI says WAIT")
                return
            
            if analysis.total_score < SCORE_MIN:
                logger.info(f"  ⏸️ Score {analysis.total_score} < {SCORE_MIN}")
                return
            
            if analysis.confidence < CONFIDENCE_MIN:
                logger.info(f"  ⏸️ Confidence {analysis.confidence}% < {CONFIDENCE_MIN}%")
                return
            
            if analysis.alignment_score < ALIGNMENT_MIN:
                logger.info(f"  ⏸️ Alignment {analysis.alignment_score} < {ALIGNMENT_MIN}")
                return
            
            # Check if already alerted
            signal_key = f"{symbol_name}_{analysis.opportunity}_{datetime.now(IST).strftime('%Y%m%d_%H')}"
            
            if signal_key in self.processed_signals:
                logger.info(f"  ⏭️ Already alerted this hour")
                return
            
            logger.info(f"  🚨 ALERT! Score: {analysis.total_score}/125")
            
            # Generate chart
            chart_path = f"/tmp/{symbol_name}_v30.png"
            ChartGenerator.create_professional_chart(
                display_name,
                mtf_data.df_15m,
                analysis,
                spot_price,
                current_oi,
                chart_path
            )
            
            # Send alert
            await self.notifier.send_alert(
                symbol_name,
                display_name,
                analysis,
                current_oi,
                chart_path,
                news_data,
                expiry_display
            )
            
            self.processed_signals.add(signal_key)
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            traceback.print_exc()
    
    async def run_scan_cycle(self):
        """Run complete scan cycle"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 SCAN START - {datetime.now(IST).strftime('%H:%M:%S IST')}")
        logger.info(f"{'='*80}")
        
        total_analyzed = 0
        alerts_sent_before = len(self.processed_signals)
        
        for idx, (instrument_key, symbol_info) in enumerate(ALL_SYMBOLS.items(), 1):
            logger.info(f"\n[{idx}/{len(ALL_SYMBOLS)}] Processing...")
            await self.analyze_symbol(instrument_key, symbol_info)
            total_analyzed += 1
            
            # Rate limiting
            if idx < len(ALL_SYMBOLS):
                await asyncio.sleep(3)
        
        alerts_sent = len(self.processed_signals) - alerts_sent_before
        
        await self.notifier.send_cycle_summary(total_analyzed, alerts_sent)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ SCAN COMPLETE | Alerts: {alerts_sent}")
        logger.info(f"{'='*80}\n")
    
    async def run(self):
        """Main bot loop"""
        logger.info("="*80)
        logger.info("HYBRID BOT v30.0 - ULTIMATE EDITION")
        logger.info("="*80)
        
        # Check credentials
        if not all([UPSTOX_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DEEPSEEK_API_KEY]):
            logger.error("❌ Missing API credentials!")
            return
        
        await self.notifier.send_startup_message(self.redis.connected)
        
        logger.info("="*80)
        logger.info(f"🟢 Bot RUNNING | Redis: {self.redis.connected}")
        logger.info("="*80)
        
        while True:
            try:
                if not self.is_market_open():
                    logger.info("⏸️ Market closed. Waiting...")
                    await asyncio.sleep(300)
                    continue
                
                await self.run_scan_cycle()
                
                logger.info(f"⏳ Next scan in 15 minutes...")
                await asyncio.sleep(SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopped by user")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                traceback.print_exc()
                await asyncio.sleep(60)

# ==================== ENTRY POINT ====================
async def main():
    """Entry point"""
    try:
        bot = HybridBot()
        await bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("STARTING HYBRID BOT v30.0 - ULTIMATE EDITION")
    logger.info("="*80)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n✅ Shutdown complete")
    except Exception as e:
        logger.error(f"\n❌ Critical error: {e}")
        traceback.print_exc()
