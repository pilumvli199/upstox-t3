#!/usr/bin/env python3
"""
HYBRID TRADING BOT v25.0 - MINIMAL FIX (ONLY DATA FETCHING)
==============================================================
✅ ONLY FIXED: Option chain fetching (empty data issue)
✅ ONLY FIXED: Auto expiry selection
✅ ONLY REMOVED: Holiday blocking
✅ KEPT: ALL original DeepSeek analysis logic
✅ KEPT: ALL original strategies & patterns
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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import traceback
import redis

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('deepseek_bot.log')]
)
logger = logging.getLogger(__name__)

# API Keys & Environment Variables
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'your_token')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'your_key')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', 'your_key')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'your_token')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'your_chat_id')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

try:
    redis_client = redis.from_url(
        REDIS_URL, 
        decode_responses=True, 
        socket_connect_timeout=5,
        socket_keepalive=True,
        retry_on_timeout=True
    )
    redis_client.ping()
    logger.info("✅ Redis connected successfully")
except Exception as e:
    logger.error(f"❌ Redis connection failed: {e}")
    redis_client = None

# ==================== SYMBOLS CONFIG (UNCHANGED) ====================
INDICES = {
    "NSE_INDEX|Nifty Bank": {"name": "BANKNIFTY", "display_name": "BANK NIFTY", "type": "index"},
    "NSE_INDEX|Nifty Midcap Select": {"name": "MIDCPNIFTY", "display_name": "MIDCAP NIFTY", "type": "index"}
}

FO_STOCKS = {
    "NSE_EQ|INE467B01029": {"name": "TATAMOTORS", "display_name": "TATA MOTORS", "type": "stock"},
    "NSE_EQ|INE585B01010": {"name": "MARUTI", "display_name": "MARUTI SUZUKI", "type": "stock"},
    "NSE_EQ|INE101A01026": {"name": "M&M", "display_name": "M&M", "type": "stock"},
    "NSE_EQ|INE917I01010": {"name": "BAJAJ-AUTO", "display_name": "BAJAJ AUTO", "type": "stock"},
    "NSE_EQ|INE040A01034": {"name": "HDFCBANK", "display_name": "HDFC BANK", "type": "stock"},
    "NSE_EQ|INE090A01021": {"name": "ICICIBANK", "display_name": "ICICI BANK", "type": "stock"},
    "NSE_EQ|INE062A01020": {"name": "SBIN", "display_name": "STATE BANK", "type": "stock"},
    "NSE_EQ|INE238A01034": {"name": "AXISBANK", "display_name": "AXIS BANK", "type": "stock"},
    "NSE_EQ|INE237A01028": {"name": "KOTAKBANK", "display_name": "KOTAK BANK", "type": "stock"},
    "NSE_EQ|INE009A01021": {"name": "INFY", "display_name": "INFOSYS", "type": "stock"},
    "NSE_EQ|INE075A01022": {"name": "WIPRO", "display_name": "WIPRO", "type": "stock"},
    "NSE_EQ|INE467B01029": {"name": "TCS", "display_name": "TCS", "type": "stock"},
    "NSE_EQ|INE047A01021": {"name": "HCLTECH", "display_name": "HCL TECH", "type": "stock"},
    "NSE_EQ|INE002A01018": {"name": "RELIANCE", "display_name": "RELIANCE IND", "type": "stock"},
    "NSE_EQ|INE044A01036": {"name": "SUNPHARMA", "display_name": "SUN PHARMA", "type": "stock"},
    "NSE_EQ|INE154A01025": {"name": "ITC", "display_name": "ITC LTD", "type": "stock"},
    "NSE_EQ|INE397D01024": {"name": "BHARTIARTL", "display_name": "BHARTI AIRTEL", "type": "stock"},
    "NSE_EQ|INE296A01024": {"name": "BAJFINANCE", "display_name": "BAJAJ FINANCE", "type": "stock"}
}

ALL_SYMBOLS = {**INDICES, **FO_STOCKS}

# ==================== DATA CLASSES (UNCHANGED) ====================
@dataclass
class StrikeData:
    strike: int
    ce_oi: int
    pe_oi: int
    ce_volume: int
    pe_volume: int

@dataclass
class OIData:
    pcr: float
    support_strike: int
    resistance_strike: int
    strikes_data: List[StrikeData]
    timestamp: datetime
    ce_oi_change_pct: float = 0.0
    pe_oi_change_pct: float = 0.0

@dataclass
class NewsData:
    headline: str
    sentiment: str
    impact_score: int
    source: str
    datetime: int

@dataclass
class AIAnalysis:
    opportunity: str
    confidence: int
    chart_score: int
    oi_score: int
    news_score: int
    total_score: int
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: str
    chart_bias: str
    market_structure: str
    pattern_signal: str
    oi_flow_signal: str
    support_levels: List[float]
    resistance_levels: List[float]
    risk_factors: List[str]
    ai_reasoning: str

# ==================== EXPIRY CALCULATOR (FIXED - ONLY THIS CHANGED) ====================
class ExpiryCalculator:
    @staticmethod
    def get_all_expiries(instrument_key: str) -> List[str]:
        """✅ FIXED: Fetch expiries from API"""
        try:
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}"
            }
            
            encoded_key = urllib.parse.quote(instrument_key, safe='')
            url = f"https://api.upstox.com/v2/option/contract?instrument_key={encoded_key}"
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                contracts = response.json().get('data', [])
                expiries = sorted(list(set(c['expiry'] for c in contracts if 'expiry' in c)))
                return expiries
            return []
        except:
            return []
    
    @staticmethod
    def get_monthly_expiry(symbol_name: str) -> str:
        """Calculate monthly expiry - RETURNS YYYY-MM-DD for API"""
        today = datetime.now(IST).date()
        current_time = datetime.now(IST).time()
        
        EXPIRY_DAY = {"BANKNIFTY": 2, "MIDCPNIFTY": 0}
        target_weekday = EXPIRY_DAY.get(symbol_name, 3)
        
        last_day = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        days_to_subtract = (last_day.weekday() - target_weekday) % 7
        expiry = last_day - timedelta(days=days_to_subtract)
        
        if expiry < today or (expiry == today and current_time >= time(15, 30)):
            next_month = (today.replace(day=28) + timedelta(days=4))
            last_day = (next_month.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            days_to_subtract = (last_day.weekday() - target_weekday) % 7
            expiry = last_day - timedelta(days=days_to_subtract)
        
        return expiry.strftime('%Y-%m-%d')
    
    @staticmethod
    def get_display_expiry(symbol_name: str) -> str:
        """Display format: DDMmmYY"""
        expiry_date = ExpiryCalculator.get_monthly_expiry(symbol_name)
        dt = datetime.strptime(expiry_date, '%Y-%m-%d')
        return dt.strftime('%d%b%y').upper()
    
    @staticmethod
    def get_best_expiry(instrument_key: str, symbol_name: str) -> str:
        """✅ NEW: Try API first, fallback to calculation"""
        # Try API
        expiries = ExpiryCalculator.get_all_expiries(instrument_key)
        if expiries:
            today = datetime.now(IST).date()
            now_time = datetime.now(IST).time()
            
            future_expiries = []
            for exp_str in expiries:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                if exp_date > today or (exp_date == today and now_time < time(15, 30)):
                    future_expiries.append(exp_str)
            
            if future_expiries:
                return min(future_expiries)
        
        # Fallback
        return ExpiryCalculator.get_monthly_expiry(symbol_name)

# ==================== REDIS OI MANAGER (UNCHANGED) ====================
class RedisOIManager:
    @staticmethod
    def save_oi(symbol: str, expiry: str, oi_data: OIData):
        if not redis_client:
            return
        try:
            key = f"oi:{symbol}:{expiry}:{oi_data.timestamp.strftime('%Y-%m-%d_%H:%M')}"
            data = {
                "pcr": oi_data.pcr,
                "support": oi_data.support_strike,
                "resistance": oi_data.resistance_strike,
                "ce_oi_change_pct": oi_data.ce_oi_change_pct,
                "pe_oi_change_pct": oi_data.pe_oi_change_pct,
                "strikes": [{"strike": s.strike, "ce_oi": s.ce_oi, "pe_oi": s.pe_oi} for s in oi_data.strikes_data]
            }
            redis_client.setex(key, 259200, json.dumps(data))
            logger.info(f"  💾 Redis: OI saved for {symbol}")
        except Exception as e:
            logger.error(f"  ❌ Redis save error: {e}")
    
    @staticmethod
    def get_comparison_oi(symbol: str, expiry: str, current_time: datetime) -> Optional[OIData]:
        if not redis_client:
            return None
        
        try:
            two_hours_ago = current_time - timedelta(hours=2)
            comparison_time = two_hours_ago.replace(
                minute=(two_hours_ago.minute // 15) * 15, 
                second=0, 
                microsecond=0
            )
            
            key = f"oi:{symbol}:{expiry}:{comparison_time.strftime('%Y-%m-%d_%H:%M')}"
            data = redis_client.get(key)
            
            if data:
                parsed = json.loads(data)
                logger.info(f"  ⏰ OI comparison: 2 hours ago ({comparison_time.strftime('%H:%M')})")
                return OIData(
                    pcr=parsed['pcr'], 
                    support_strike=parsed['support'], 
                    resistance_strike=parsed['resistance'],
                    ce_oi_change_pct=parsed.get('ce_oi_change_pct', 0), 
                    pe_oi_change_pct=parsed.get('pe_oi_change_pct', 0),
                    strikes_data=[StrikeData(s['strike'], s['ce_oi'], s['pe_oi'], 0, 0) for s in parsed['strikes']],
                    timestamp=comparison_time
                )
            else:
                logger.info(f"  ⚠️ No OI data from 2h ago, using fresh baseline")
                return None
        except Exception as e:
            logger.error(f"  ❌ Redis get error: {e}")
            return None

# ==================== NEWS FETCHER (UNCHANGED) ====================
class NewsFetcher:
    @staticmethod
    def fetch_finnhub_news(symbol_name: str) -> Optional[NewsData]:
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
                        impact = 25
                    elif sentiment == 'NEGATIVE':
                        impact = -25
                    else:
                        impact = 0
                    
                    return NewsData(
                        headline=latest.get('headline', 'No headline')[:150],
                        sentiment=sentiment,
                        impact_score=impact,
                        source='Finnhub',
                        datetime=latest.get('datetime', 0)
                    )
        except Exception as e:
            logger.error(f"  📰 News fetch error: {e}")
        return None

# ==================== MULTI-TIMEFRAME PROCESSOR (UNCHANGED) ====================
class MultiTimeframeProcessor:
    @staticmethod
    def resample(df_1m: pd.DataFrame, tf: str) -> pd.DataFrame:
        df = df_1m.copy()
        df.set_index('timestamp', inplace=True)
        resampled = df.resample(tf).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()
        return resampled

# ==================== DATA COMPRESSOR (UNCHANGED) ====================
class DataCompressor:
    @staticmethod
    def compress_to_ohlc(df: pd.DataFrame, limit: int = None) -> str:
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
        if not prev:
            return f"PCR:{current.pcr:.2f}|S:{current.support_strike}|R:{current.resistance_strike}"
        
        prev_map = {s.strike: s for s in prev.strikes_data}
        ce_builds, pe_builds, ce_unwinds, pe_unwinds = [], [], [], []
        
        for s in current.strikes_data[:10]:
            if s.strike in prev_map:
                ps = prev_map[s.strike]
                ce_chg = (s.ce_oi - ps.ce_oi) / ps.ce_oi if ps.ce_oi > 0 else 0
                pe_chg = (s.pe_oi - ps.pe_oi) / ps.pe_oi if ps.pe_oi > 0 else 0
                
                if ce_chg > 0.15: ce_builds.append(s.strike)
                if pe_chg > 0.15: pe_builds.append(s.strike)
                if ce_chg < -0.10: ce_unwinds.append(s.strike)
                if pe_chg < -0.10: pe_unwinds.append(s.strike)
        
        result = f"PCR:{current.pcr:.2f}|CE_CHG:{current.ce_oi_change_pct:+.1f}%|PE_CHG:{current.pe_oi_change_pct:+.1f}%"
        if ce_builds: result += f"|CE_BUILD:{','.join(map(str, ce_builds[:2]))}"
        if pe_builds: result += f"|PE_BUILD:{','.join(map(str, pe_builds[:2]))}"
        if ce_unwinds: result += f"|CE_UNWIND:{','.join(map(str, ce_unwinds[:2]))}"
        if pe_unwinds: result += f"|PE_UNWIND:{','.join(map(str, pe_unwinds[:2]))}"
        
        return result

# ==================== DEEPSEEK V3 AI ANALYZER (UNCHANGED - YOUR ORIGINAL STRATEGY) ====================
class DeepSeekAnalyzer:
    @staticmethod
    def generate_professional_prompt(symbol: str, df_1h: pd.DataFrame, df_15m: pd.DataFrame,
                                    df_5m: pd.DataFrame, spot_price: float, atr: float,
                                    current_oi: OIData, prev_oi: Optional[OIData],
                                    news: Optional[NewsData]) -> str:
        
        ohlc_1h = DataCompressor.compress_to_ohlc(df_1h, limit=50)
        ohlc_15m = DataCompressor.compress_to_ohlc(df_15m, limit=500)
        ohlc_5m = DataCompressor.compress_to_ohlc(df_5m, limit=100)
        oi_summary = DataCompressor.compress_oi(current_oi, prev_oi)
        
        news_section = ""
        if news:
            news_section = f"""
**NEWS CONTEXT (Finnhub):**
Headline: {news.headline}
Sentiment: {news.sentiment}
Impact: {"Positive bias (+25pts)" if news.impact_score > 0 else "Negative bias (-25pts)" if news.impact_score < 0 else "Neutral"}
"""
        
        prompt = f"""You are an expert F&O price action trader specializing in Indian markets. Analyze using institutional-grade confluence.

**INSTRUMENT:** {symbol}
**SPOT PRICE:** ₹{spot_price:.2f}
**ATR:** {atr:.2f}

---

**DATA PROVIDED (OHLC Short Format):**

**1H TIMEFRAME (Last 50 candles - Trend Identification):**
{ohlc_1h}

**15M TIMEFRAME (Last 500 candles - Main Analysis):**
{ohlc_15m}

**5M TIMEFRAME (Last 100 candles - Entry/Exit Precision):**
{ohlc_5m}

**OI DATA (2-Hour Comparison):**
{oi_summary}
Support Zone: {current_oi.support_strike}
Resistance Zone: {current_oi.resistance_strike}

{news_section}

---

**ANALYSIS FRAMEWORK (Confluence-Based):**

**STEP 1: 1H TREND ANALYSIS (Mandatory Filter)**
- Overall trend direction (uptrend/downtrend/sideways)
- Trend strength using HH/HL or LH/LL
- Moving average position (simulate SMA20)
- Recent break of structure?

**STEP 2: 15M CHART PATTERN RECOGNITION**

**A) CHART PATTERNS (Continuation + Reversal):**
- Ascending/Descending Triangle
- Symmetrical Triangle
- Bull/Bear Flag
- Pennant
- Head & Shoulders / Inverse H&S
- Double Top/Bottom
- Triple Top/Bottom
- Rising/Falling Wedge
- Channel patterns

**B) CANDLESTICK PATTERNS (Research-Based Rules):**

**Single Candle Patterns:**
- Hammer (Long lower shadow, small body, bullish at support)
- Shooting Star (Long upper shadow, bearish at resistance)
- Doji (Indecision, needs confirmation)
- Dragonfly Doji (Bullish reversal)
- Gravestone Doji (Bearish reversal)
- Spinning Top (Indecision)
- Marubozu (Strong trend continuation)

**Double Candle Patterns:**
- Bullish/Bearish Engulfing (70%+ success with volume)
- Tweezer Top/Bottom (55% win rate, needs confirmation)
- Harami (Indecision after trend)
- Piercing Line / Dark Cloud Cover

**Triple Candle Patterns:**
- Morning Star / Evening Star (Reversal)
- Three White Soldiers / Three Black Crows (Strong trend)
- Abandoned Baby (Rare but powerful)

**CONFLUENCE RULES (From Research):**
1. Pattern at S/R level = +20% reliability (StockCharts.com)
2. Pattern + Volume spike = +25% success (TradingView)
3. Multi-timeframe alignment = +30% probability (XS.com)
4. Pattern + OI confirmation = +35% win rate (Strike.money)

**STEP 3: SUPPORT & RESISTANCE (15M Chart)**
- Identify minimum 3-4 key levels (2-3 tests minimum)
- Psychological levels (round numbers)
- Current price position relative to S/R
- Order blocks / demand-supply zones

**STEP 4: MARKET STRUCTURE**
- Higher highs + higher lows = Bullish structure
- Lower highs + lower lows = Bearish structure
- Break of structure (BOS) detected?
- Change of character (CHoCH)?

**STEP 5: VOLUME ANALYSIS**
- Volume at key S/R levels
- Climax volume bars (>1.5× average = significant)
- Volume divergence with price (bearish if price up, volume down)
- Volume confirmation on pattern breakout

**STEP 6: OI FLOW ANALYSIS (CRITICAL)**

**OI INTERPRETATION RULES:**
- **CE Unwinding (-ve):** Resistance weakening = BULLISH signal
- **PE Unwinding (-ve):** Support weakening = BEARISH signal
- **CE Building (+ve):** Resistance strengthening = BEARISH signal
- **PE Building (+ve):** Support strengthening = BULLISH signal

**PCR Interpretation:**
- PCR > 1.2 = Bullish sentiment (Put writers confident)
- PCR < 0.8 = Bearish sentiment (Call writers confident)
- PCR 0.8-1.2 = Neutral

**Pattern-OI Confluence (Professional Rule):**
- Bullish Pattern + CE Unwinding = HIGH PROBABILITY BUY (Score +30)
- Bullish Pattern + PE Building = MODERATE BUY (Score +20)
- Bearish Pattern + PE Unwinding = HIGH PROBABILITY SELL (Score +30)
- Bearish Pattern + CE Building = MODERATE SELL (Score +20)
- Pattern without OI support = REJECT (Score -20)

**STEP 7: MULTI-TIMEFRAME CONFLUENCE**
- 1H trend MUST align with 15M setup
- 5M used ONLY for entry/exit precision
- Counter-trend trades = REJECTED

**STEP 8: SCORING SYSTEM (Charts: 45 + OI: 45 + News: 10 = 100)**

**CHART SCORE (0-45 points):**
- Trend clarity (1H): 0-10 pts (Strong trend = 10, Weak = 5, Sideways = 0)
- Pattern strength (15M): 0-15 pts (Textbook pattern = 15, Partial = 8, Weak = 3)
- Support/Resistance respect: 0-10 pts (Clean bounces = 10, Choppy = 5)
- Volume confirmation: 0-10 pts (High volume on key levels = 10, Low = 3)

**OI SCORE (0-45 points):**
- PCR interpretation: 0-10 pts (Extreme PCR = 10, Neutral = 5)
- OI change magnitude: 0-15 pts (>15% change = 15, 10-15% = 10, 5-10% = 5)
- Pattern-OI confluence: 0-20 pts (Perfect match = 20, Partial = 10, Mismatch = -10)

**NEWS SCORE (0-10 points):**
- Positive news + Bullish setup = +10 pts
- Negative news + Bearish setup = +10 pts
- Neutral news = +5 pts
- Contradicting news = -5 pts

**TOTAL SCORE INTERPRETATION:**
- 85-100: VERY HIGH PROBABILITY (Aggressive position)
- 70-84: HIGH PROBABILITY (Standard position)
- 50-69: MODERATE (Reduced size or WAIT)
- <50: LOW (WAIT for better setup)

**MINIMUM THRESHOLD FOR TRADE:**
- Total Score: ≥ 70
- Confidence: ≥ 75%
- Risk:Reward: ≥ 1:2
- Entry: Use 5M chart for precise entry (current spot or breakout level)
- Stop Loss: Beyond recent swing low/high on 5M + 0.3× ATR
- Target 1: Minimum 1:2 Risk:Reward
- Target 2: Minimum 1:3.5 Risk:Reward
- Recommended Strike: Nearest ATM/ITM

**REJECTION CRITERIA:**
- No clear 1H trend = WAIT
- Conflicting 1H vs 15M = WAIT
- Pattern without OI confirmation = WAIT
- Choppy/unclear price action = WAIT
- Risk:Reward < 1:1.5 = WAIT

**STEP 9: RISK FACTORS**
- Nearby major S/R blocking move?
- News sentiment contradicting setup?
- Low volume on key levels?
- Multiple Doji candles (indecision)?

---

**OUTPUT FORMAT (JSON ONLY, NO MARKDOWN):**

{{
  "opportunity": "CE_BUY/PE_BUY/WAIT",
  "confidence": 85,
  "chart_score": 38,
  "oi_score": 40,
  "news_score": 8,
  "total_score": 86,
  "entry_price": {spot_price:.2f},
  "stop_loss": 0.0,
  "target_1": 0.0,
  "target_2": 0.0,
  "risk_reward": "1:2.5",
  "chart_bias": "Bullish/Bearish/Neutral",
  "market_structure": "HH/HL forming",
  "pattern_signal": "Bullish Flag + Hammer",
  "oi_flow_signal": "CE Unwinding at resistance",
  "support_levels": [0.0, 0.0, 0.0],
  "resistance_levels": [0.0, 0.0, 0.0],
  "risk_factors": ["Risk 1", "Risk 2"],
  "ai_reasoning": "Chart Score (38/45): 1H strong uptrend (9/10) + 15M bullish flag (14/15) + Clean support bounces (8/10) + Volume spike (7/10). OI Score (40/45): PCR 1.25 bullish (9/10) + CE unwinding 12% (12/15) + Pattern-OI perfect match (19/20). News Score (8/10): Positive earnings aligned with bullish setup. TOTAL: 86/100 = VERY HIGH PROBABILITY."
}}

**IMPORTANT:**
- Be brutally honest with scoring
- If total score < 70, return "WAIT"
- Pattern without OI confirmation = Deduct 10 points from OI score
- News contradicting setup = Deduct 5 points from news score
- All targets must satisfy Risk:Reward ≥ 1:2
"""
        
        return prompt
    
    @staticmethod
    def parse_ai_response(content: str) -> Optional[AIAnalysis]:
        try:
            content = content.replace('```json', '').replace('```', '').strip()
            
            try:
                analysis = json.loads(content)
            except:
                import re
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
                total_score=analysis.get('total_score', 0),
                entry_price=analysis.get('entry_price', 0),
                stop_loss=analysis.get('stop_loss', 0),
                target_1=analysis.get('target_1', 0),
                target_2=analysis.get('target_2', 0),
                risk_reward=analysis.get('risk_reward', '0:0'),
                chart_bias=analysis.get('chart_bias', 'Neutral'),
                market_structure=analysis.get('market_structure', 'Unknown'),
                pattern_signal=analysis.get('pattern_signal', 'None'),
                oi_flow_signal=analysis.get('oi_flow_signal', 'Neutral'),
                support_levels=analysis.get('support_levels', []),
                resistance_levels=analysis.get('resistance_levels', []),
                risk_factors=analysis.get('risk_factors', []),
                ai_reasoning=analysis.get('ai_reasoning', 'No reasoning provided')
            )
        except Exception as e:
            logger.error(f"AI parse error: {e}")
            return None
    
    @staticmethod
    def analyze(symbol: str, df_1h: pd.DataFrame, df_15m: pd.DataFrame, df_5m: pd.DataFrame,
               spot_price: float, atr: float, current_oi: OIData, prev_oi: Optional[OIData],
               news: Optional[NewsData]) -> Optional[AIAnalysis]:
        try:
            prompt = DeepSeekAnalyzer.generate_professional_prompt(
                symbol, df_1h, df_15m, df_5m, spot_price, atr, current_oi, prev_oi, news
            )
            
            logger.info(f"  🤖 Calling DeepSeek V3...")
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You are an expert F&O trader. Analyze data and respond ONLY in JSON format."},
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
                logger.error(f"  ❌ DeepSeek API error: {response.status_code}")
                return None
            
            ai_content = response.json()['choices'][0]['message']['content']
            analysis = DeepSeekAnalyzer.parse_ai_response(ai_content)
            
            if analysis:
                logger.info(f"  🤖 AI: {analysis.opportunity} | Score: {analysis.total_score}/100 " +
                          f"(Chart:{analysis.chart_score} OI:{analysis.oi_score} News:{analysis.news_score})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"  ❌ DeepSeek analysis error: {e}")
            traceback.print_exc()
            return None

# ==================== CHART GENERATOR (UNCHANGED) ====================
class ChartGenerator:
    @staticmethod
    def create_professional_chart(symbol: str, df: pd.DataFrame, analysis: AIAnalysis,
                                  spot: float, path: str):
        BG, GRID, TEXT = '#131722', '#1e222d', '#d1d4dc'
        GREEN, RED, YELLOW = '#26a69a', '#ef5350', '#ffd700'
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]}, facecolor=BG)
        
        ax1.set_facecolor(BG)
        df_plot = df.tail(150).reset_index(drop=True)
        
        for idx, row in df_plot.iterrows():
            color = GREEN if row['close'] > row['open'] else RED
            ax1.add_patch(Rectangle((idx, min(row['open'], row['close'])), 0.6,
                                   abs(row['close'] - row['open']), facecolor=color, alpha=0.8))
            ax1.plot([idx+0.3, idx+0.3], [row['low'], row['high']], color=color, linewidth=1, alpha=0.6)
        
        for sup in analysis.support_levels[:3]:
            if sup > 0:
                ax1.axhline(sup, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.7)
                ax1.text(len(df_plot)*0.02, sup, f'S: ₹{sup:.1f}', color=GREEN, fontsize=9, 
                        bbox=dict(boxstyle='round', facecolor=BG, alpha=0.7))
        
        for res in analysis.resistance_levels[:3]:
            if res > 0:
                ax1.axhline(res, color=RED, linestyle='--', linewidth=1.5, alpha=0.7)
                ax1.text(len(df_plot)*0.02, res, f'R: ₹{res:.1f}', color=RED, fontsize=9,
                        bbox=dict(boxstyle='round', facecolor=BG, alpha=0.7))
        
        if analysis.opportunity != "WAIT":
            ax1.scatter([len(df_plot)-1], [analysis.entry_price], color=YELLOW, s=300, 
                       marker='D', zorder=5, edgecolors='white', linewidths=2)
            ax1.axhline(analysis.stop_loss, color=RED, linewidth=2.5, linestyle=':')
            ax1.axhline(analysis.target_1, color=GREEN, linewidth=2, linestyle=':')
            ax1.axhline(analysis.target_2, color=GREEN, linewidth=1.5, linestyle=':')
            
            ax1.text(len(df_plot)*0.98, analysis.entry_price, f'ENTRY: ₹{analysis.entry_price:.2f}  ',
                    color=YELLOW, fontsize=10, ha='right', va='center',
                    bbox=dict(boxstyle='round', facecolor=BG, edgecolor=YELLOW, linewidth=2))
        
        score_color = GREEN if analysis.total_score >= 85 else (YELLOW if analysis.total_score >= 70 else RED)
        
        info = f"""{'🟢 BUY' if analysis.opportunity=='CE_BUY' else '🔴 SELL' if analysis.opportunity=='PE_BUY' else '⏸️ WAIT'}

SCORE: {analysis.total_score}/100
├─ Chart: {analysis.chart_score}/45
├─ OI: {analysis.oi_score}/45
└─ News: {analysis.news_score}/10

Confidence: {analysis.confidence}%

Bias: {analysis.chart_bias}
Structure: {analysis.market_structure}
Pattern: {analysis.pattern_signal[:25]}

OI: {analysis.oi_flow_signal[:25]}

Entry: ₹{analysis.entry_price:.1f}
SL: ₹{analysis.stop_loss:.1f}
T1: ₹{analysis.target_1:.1f}
T2: ₹{analysis.target_2:.1f}
R:R: {analysis.risk_reward}"""
        
        ax1.text(0.01, 0.99, info, transform=ax1.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor=GRID, alpha=0.95, edgecolor=score_color, linewidth=2),
                color=TEXT, family='monospace')
        
        score_emoji = "🔥" if analysis.total_score >= 85 else ("✅" if analysis.total_score >= 70 else "⚠️")
        title = f"{score_emoji} {symbol} | 15M | DeepSeek V3 | Score: {analysis.total_score}/100"
        if analysis.pattern_signal:
            title += f" | {analysis.pattern_signal[:40]}"
        
        ax1.set_title(title, color=TEXT, fontsize=13, fontweight='bold', pad=15)
        ax1.grid(True, color=GRID, alpha=0.3)
        ax1.tick_params(colors=TEXT)
        ax1.set_ylabel('Price (₹)', color=TEXT, fontsize=11)
        
        ax2.set_facecolor(BG)
        colors = [GREEN if df_plot.iloc[i]['close']>df_plot.iloc[i]['open'] else RED 
                 for i in range(len(df_plot))]
        ax2.bar(range(len(df_plot)), df_plot['volume'], color=colors, alpha=0.6)
        ax2.set_ylabel('Volume', color=TEXT, fontsize=10)
        ax2.tick_params(colors=TEXT)
        ax2.grid(True, color=GRID, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, facecolor=BG)
        plt.close()
        logger.info(f"  📊 Chart saved: {path}")

# ==================== DATA FETCHER (FIXED - ONLY THIS CHANGED) ====================
class UpstoxDataFetcher:
    def __init__(self, token: str):
        self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    
    def get_historical(self, key: str, interval: str, days: int = 10) -> pd.DataFrame:
        try:
            to_date = datetime.now(IST)
            from_date = to_date - timedelta(days=days)
            
            url = f"https://api.upstox.com/v2/historical-candle/{key}/{interval}/{to_date.strftime('%Y-%m-%d')}/{from_date.strftime('%Y-%m-%d')}"
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'candles' in data['data']:
                    df = pd.DataFrame(data['data']['candles'],
                                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df.sort_values('timestamp').reset_index(drop=True)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Historical data error: {e}")
            return pd.DataFrame()
    
    def get_ltp(self, key: str) -> float:
        try:
            response = requests.get("https://api.upstox.com/v2/market-quote/ltp",
                                  headers=self.headers, params={"instrument_key": key}, timeout=10)
            if response.status_code == 200:
                return response.json()['data'][key]['last_price']
            return 0.0
        except:
            return 0.0
    
    def get_option_chain(self, key: str, expiry: str) -> List[StrikeData]:
        """✅ FIXED: Proper retry logic and URL encoding"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # ✅ FIX: Proper URL encoding
                encoded_key = urllib.parse.quote(key, safe='')
                url = f"https://api.upstox.com/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
                
                response = requests.get(url, headers=self.headers, timeout=20)
                
                logger.info(f"     API Response: {response.status_code} (Attempt {attempt + 1}/{max_retries})")
                
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
                        logger.warning(f"     ⚠️ Empty 'data' array")
                        if attempt < max_retries - 1:
                            time_sleep.sleep(retry_delay * (attempt + 1))
                            continue
                        return []
                    
                    strikes = []
                    for item in strikes_raw:
                        call = item.get('call_options', {}).get('market_data', {})
                        put = item.get('put_options', {}).get('market_data', {})
                        
                        ce_oi = call.get('oi', 0)
                        pe_oi = put.get('oi', 0)
                        
                        if ce_oi == 0 and pe_oi == 0:
                            continue
                        
                        strikes.append(StrikeData(
                            strike=int(item.get('strike_price', 0)),
                            ce_oi=ce_oi,
                            pe_oi=pe_oi,
                            ce_volume=call.get('volume', 0),
                            pe_volume=put.get('volume', 0)
                        ))
                    
                    if strikes:
                        logger.info(f"     ✅ Fetched {len(strikes)} strikes with OI data")
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
                    logger.warning(f"     ⚠️ No options available for expiry {expiry}")
                    return []
                
                else:
                    logger.error(f"     ❌ API Error: {response.status_code} - {response.text[:200]}")
                    if attempt < max_retries - 1:
                        time_sleep.sleep(retry_delay * (attempt + 1))
                        continue
                    return []
                    
            except Exception as e:
                logger.error(f"     ❌ Exception (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time_sleep.sleep(retry_delay * (attempt + 1))
                    continue
                return []
        
        return []

# ==================== MAIN BOT (MINIMAL CHANGES) ====================
class HybridBot:
    def __init__(self):
        self.data_fetcher = UpstoxDataFetcher(UPSTOX_ACCESS_TOKEN)
        self.telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.processed_signals = set()
    
    async def send_startup_message(self):
        redis_status = "✅ Connected" if redis_client else "❌ Disconnected"
        
        message = f"""
🚀 **HYBRID BOT v25.0 STARTED - MINIMAL FIX**

⏰ **Start Time:** {datetime.now(IST).strftime('%d-%b-%Y %H:%M:%S IST')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 **FIXES APPLIED**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Option chain fetching (empty data fixed)
✅ Auto expiry selection with API fallback
✅ Retry logic (3 attempts, exponential backoff)
✅ Holiday check removed (only weekends)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **ORIGINAL FEATURES KEPT**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤖 DeepSeek V3 AI (ALL strategies intact)
📈 Multi-Timeframe: 1H + 15M + 5M
📊 20+ Chart Patterns
📊 10+ Candlestick Patterns
📊 OI Analysis (2-hour comparison)
📰 News Integration
🎯 Scoring: 70/100 threshold

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📡 **SYSTEM STATUS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📦 Redis: {redis_status}
🔄 Scan: Every 15 minutes
📊 Monitoring: {len(ALL_SYMBOLS)} symbols
⏰ Market: 09:15 - 15:30 IST

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🟢 **BOT ACTIVE & SCANNING**

Next scan starts at market open...
"""
        
        await self.telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
        logger.info("✅ Startup message sent to Telegram")
    
    async def send_alert(self, symbol: str, analysis: AIAnalysis, chart_path: str,
                        oi_summary: str, news: Optional[NewsData], expiry: str):
        try:
            with open(chart_path, 'rb') as photo:
                await self.telegram_bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo)
            
            risk = abs(analysis.entry_price - analysis.stop_loss)
            reward = abs(analysis.target_1 - analysis.entry_price)
            rr = reward / risk if risk > 0 else 0
            
            signal_emoji = "🟢" if analysis.opportunity == "CE_BUY" else "🔴"
            score_emoji = "🔥" if analysis.total_score >= 85 else "✅"
            
            message = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{signal_emoji} **{symbol} {analysis.opportunity}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{score_emoji} **SCORE: {analysis.total_score}/100**

📊 **Breakdown:**
├─ Chart: **{analysis.chart_score}/45**
├─ OI: **{analysis.oi_score}/45**
└─ News: **{analysis.news_score}/10**

**Confidence: {analysis.confidence}%**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 **MARKET ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Chart View:**
├─ Bias: {analysis.chart_bias}
├─ Structure: {analysis.market_structure}
└─ Pattern: {analysis.pattern_signal}

**OI Flow:**
{oi_summary}
└─ Signal: {analysis.oi_flow_signal}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 **TRADE SETUP**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Entry:** ₹{analysis.entry_price:.2f}
**Stop Loss:** ₹{analysis.stop_loss:.2f}
**Target 1:** ₹{analysis.target_1:.2f} 🎯
**Target 2:** ₹{analysis.target_2:.2f} 🎯🎯

**Risk:Reward:** 1:{rr:.1f}
**Risk Amount:** ₹{risk:.2f}
**Reward (T1):** ₹{reward:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 **AI REASONING**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{analysis.ai_reasoning}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ **RISK FACTORS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
            
            for i, risk_factor in enumerate(analysis.risk_factors[:3], 1):
                message += f"{i}. {risk_factor}\n"
            
            if news:
                message += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📰 **NEWS UPDATE** ({news.source})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Headline:**
{news.headline}

**Sentiment:** {news.sentiment}
**Impact:** {"Positive +" if news.impact_score > 0 else "Negative " if news.impact_score < 0 else "Neutral ±"}{abs(news.impact_score)} points

"""
            
            message += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Expiry:** {expiry}
🕐 **Alert Time:** {datetime.now(IST).strftime('%d-%b-%Y %H:%M:%S IST')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            await self.telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
            logger.info(f"  ✅ Professional alert sent for {symbol}")
        except Exception as e:
            logger.error(f"Telegram alert error: {e}")
            traceback.print_exc()
    
    async def analyze_symbol(self, instrument_key: str, symbol_info: Dict):
        try:
            symbol_name = symbol_info['name']
            display_name = symbol_info['display_name']
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 {display_name} ({symbol_name})")
            logger.info(f"{'='*70}")
            
            # ✅ FIXED: Auto expiry with API + fallback
            expiry_api = ExpiryCalculator.get_best_expiry(instrument_key, symbol_name)
            expiry_display = ExpiryCalculator.get_display_expiry(symbol_name)
            
            logger.info(f"  📅 Expiry: {expiry_display} (API: {expiry_api})")
            logger.info(f"     Instrument Key: {instrument_key}")
            
            # Fetch 1-min data
            df_1m = self.data_fetcher.get_historical(instrument_key, "1minute", days=15)
            if df_1m.empty:
                logger.warning(f"  ⚠️ No data")
                return
            
            # Resample
            df_1h = MultiTimeframeProcessor.resample(df_1m, '1H')
            df_15m = MultiTimeframeProcessor.resample(df_1m, '15T')
            df_5m = MultiTimeframeProcessor.resample(df_1m, '5T')
            
            logger.info(f"  📊 Data: 1H({len(df_1h)}) | 15M({len(df_15m)}) | 5M({len(df_5m)})")
            
            # Spot & ATR
            spot_price = self.data_fetcher.get_ltp(instrument_key)
            if spot_price == 0:
                spot_price = df_15m['close'].iloc[-1]
            
            df_15m['tr'] = df_15m[['high', 'low', 'close']].apply(
                lambda x: max(x['high']-x['low'], abs(x['high']-x['close']), abs(x['low']-x['close'])), axis=1
            )
            atr = df_15m['tr'].rolling(14).mean().iloc[-1]
            logger.info(f"  💹 Spot: ₹{spot_price:.2f} | ATR: {atr:.2f}")
            
            # ✅ FIXED: Option chain with retry
            all_strikes = self.data_fetcher.get_option_chain(instrument_key, expiry_api)
            
            if not all_strikes:
                logger.warning(f"  ⚠️ No OI data available - skipping")
                return
            
            # Filter ATM strikes
            atm = round(spot_price / 100) * 100
            atm_range = range(atm - 700, atm + 800, 100)
            top_15 = sorted([s for s in all_strikes if s.strike in atm_range],
                          key=lambda x: (x.ce_oi + x.pe_oi), reverse=True)[:15]
            
            if len(top_15) == 0:
                logger.warning(f"  ⚠️ No strikes in ATM range")
                return
            
            logger.info(f"  📊 Selected {len(top_15)} ATM strikes")
            
            # OI Analysis
            total_ce = sum(s.ce_oi for s in top_15)
            total_pe = sum(s.pe_oi for s in top_15)
            pcr = total_pe / total_ce if total_ce > 0 else 0
            
            max_ce_strike = max(top_15, key=lambda x: x.ce_oi).strike
            max_pe_strike = max(top_15, key=lambda x: x.pe_oi).strike
            
            prev_oi = RedisOIManager.get_comparison_oi(symbol_name, expiry_display, datetime.now(IST))
            
            ce_change_pct = 0.0
            pe_change_pct = 0.0
            if prev_oi:
                prev_ce_total = sum(s.ce_oi for s in prev_oi.strikes_data)
                prev_pe_total = sum(s.pe_oi for s in prev_oi.strikes_data)
                if prev_ce_total > 0:
                    ce_change_pct = ((total_ce - prev_ce_total) / prev_ce_total) * 100
                if prev_pe_total > 0:
                    pe_change_pct = ((total_pe - prev_pe_total) / prev_pe_total) * 100
            
            current_oi = OIData(
                pcr=pcr, support_strike=max_pe_strike, resistance_strike=max_ce_strike,
                strikes_data=top_15, timestamp=datetime.now(IST),
                ce_oi_change_pct=ce_change_pct, pe_oi_change_pct=pe_change_pct
            )
            
            logger.info(f"  📊 PCR: {pcr:.2f} | S: {max_pe_strike} | R: {max_ce_strike}")
            logger.info(f"     CE: {ce_change_pct:+.1f}% | PE: {pe_change_pct:+.1f}%")
            
            # Save OI
            RedisOIManager.save_oi(symbol_name, expiry_display, current_oi)
            
            # Fetch News
            news_data = NewsFetcher.fetch_finnhub_news(symbol_name)
            if news_data:
                logger.info(f"  📰 {news_data.headline[:60]}... [{news_data.sentiment}]")
            
            # DeepSeek AI Analysis (YOUR ORIGINAL STRATEGY)
            analysis = DeepSeekAnalyzer.analyze(
                symbol=display_name,
                df_1h=df_1h,
                df_15m=df_15m,
                df_5m=df_5m,
                spot_price=spot_price,
                atr=atr,
                current_oi=current_oi,
                prev_oi=prev_oi,
                news=news_data
            )
            
            if not analysis:
                logger.info(f"  ⏸️ No AI analysis")
                return
            
            # Threshold check (70% minimum)
            if analysis.opportunity != "WAIT" and analysis.total_score >= 70 and analysis.confidence >= 75:
                signal_key = f"{symbol_name}_{analysis.opportunity}_{datetime.now(IST).strftime('%Y%m%d_%H')}"
                
                if signal_key not in self.processed_signals:
                    logger.info(f"  🚨 ALERT! Score: {analysis.total_score}/100 (≥70 threshold)")
                    
                    # Generate chart
                    chart_path = f"/tmp/{symbol_name}_deepseek.png"
                    ChartGenerator.create_professional_chart(display_name, df_15m, analysis, spot_price, chart_path)
                    
                    # OI summary
                    oi_summary = DataCompressor.compress_oi(current_oi, prev_oi)
                    
                    # Send alert
                    await self.send_alert(display_name, analysis, chart_path, oi_summary, news_data, expiry_display)
                    self.processed_signals.add(signal_key)
                else:
                    logger.info(f"  ⏭️ Already alerted today")
            else:
                threshold_msg = f"Score {analysis.total_score}/100 (<70)" if analysis.total_score < 70 else f"Conf {analysis.confidence}% (<75%)"
                logger.info(f"  ⏸️ {analysis.opportunity} | {threshold_msg} | Below threshold")
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            traceback.print_exc()
    
    async def run_scanner(self):
        logger.info("\n" + "="*80)
        logger.info("🚀 HYBRID BOT v25.0 - MINIMAL FIX VERSION")
        logger.info("="*80)
        
        await self.send_startup_message()
        
        while True:
            try:
                now = datetime.now(IST)
                current_time = now.time()
                
                # ✅ REMOVED: Holiday check (only weekends now)
                if now.weekday() >= 5:
                    logger.info(f"📅 Weekend. Next check in 1 hour...")
                    await asyncio.sleep(3600)
                    continue
                
                # Market hours check
                if current_time < time(9, 15) or current_time > time(15, 30):
                    logger.info(f"⏸️ Market closed. Waiting 5 minutes...")
                    await asyncio.sleep(300)
                    continue
                
                logger.info(f"\n🔄 Scan started: {now.strftime('%H:%M:%S')}")
                
                # Scan all symbols
                total_symbols = len(ALL_SYMBOLS)
                for idx, (instrument_key, symbol_info) in enumerate(ALL_SYMBOLS.items(), 1):
                    logger.info(f"\n[{idx}/{total_symbols}] Scanning...")
                    await self.analyze_symbol(instrument_key, symbol_info)
                    
                    # Rate limiting: 3 sec between symbols
                    if idx < total_symbols:
                        await asyncio.sleep(3)
                
                logger.info(f"\n✅ Scan complete ({total_symbols} symbols). Next in 15 min...")
                await asyncio.sleep(900)
                
            except Exception as e:
                logger.error(f"Scanner error: {e}")
                traceback.print_exc()
                await asyncio.sleep(60)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    bot = HybridBot()
    asyncio.run(bot.run_scanner())
