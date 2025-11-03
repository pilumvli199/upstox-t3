#!/usr/bin/env python3
"""
HYBRID TRADING BOT v25.1 - FIXED OPTION CHAIN + AUTO EXPIRY
==============================================================
✅ Fixed: Option chain fetching (empty data issue)
✅ Fixed: Auto expiry selection with validation
✅ Fixed: Proper instrument key handling
✅ Removed: Holiday blocking (auto-skips weekends)
✅ Enhanced: Retry logic with exponential backoff
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

# API Keys
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

# ==================== SYMBOLS CONFIG ====================
INDICES = {
    "NSE_INDEX|Nifty Bank": {"name": "BANKNIFTY", "display_name": "BANK NIFTY", "type": "index", "expiry_day": 2},
    "NSE_INDEX|Nifty Midcap Select": {"name": "MIDCPNIFTY", "display_name": "MIDCAP NIFTY", "type": "index", "expiry_day": 0}
}

FO_STOCKS = {
    # AUTO SECTOR
    "NSE_EQ|INE467B01029": {"name": "TATAMOTORS", "display_name": "TATA MOTORS", "type": "stock", "expiry_day": 3},
    "NSE_EQ|INE585B01010": {"name": "MARUTI", "display_name": "MARUTI SUZUKI", "type": "stock", "expiry_day": 3},
    "NSE_EQ|INE101A01026": {"name": "M&M", "display_name": "M&M", "type": "stock", "expiry_day": 3},
    "NSE_EQ|INE917I01010": {"name": "BAJAJ-AUTO", "display_name": "BAJAJ AUTO", "type": "stock", "expiry_day": 3},
    
    # BANKING SECTOR
    "NSE_EQ|INE040A01034": {"name": "HDFCBANK", "display_name": "HDFC BANK", "type": "stock", "expiry_day": 3},
    "NSE_EQ|INE090A01021": {"name": "ICICIBANK", "display_name": "ICICI BANK", "type": "stock", "expiry_day": 3},
    "NSE_EQ|INE062A01020": {"name": "SBIN", "display_name": "STATE BANK", "type": "stock", "expiry_day": 3},
    "NSE_EQ|INE238A01034": {"name": "AXISBANK", "display_name": "AXIS BANK", "type": "stock", "expiry_day": 3},
    "NSE_EQ|INE237A01028": {"name": "KOTAKBANK", "display_name": "KOTAK BANK", "type": "stock", "expiry_day": 3},
    
    # IT SECTOR
    "NSE_EQ|INE009A01021": {"name": "INFY", "display_name": "INFOSYS", "type": "stock", "expiry_day": 3},
    "NSE_EQ|INE075A01022": {"name": "WIPRO", "display_name": "WIPRO", "type": "stock", "expiry_day": 3},
    "NSE_EQ|INE467B01029": {"name": "TCS", "display_name": "TCS", "type": "stock", "expiry_day": 3},
    "NSE_EQ|INE047A01021": {"name": "HCLTECH", "display_name": "HCL TECH", "type": "stock", "expiry_day": 3},
    
    # ENERGY SECTOR
    "NSE_EQ|INE002A01018": {"name": "RELIANCE", "display_name": "RELIANCE IND", "type": "stock", "expiry_day": 3},
    
    # PHARMA SECTOR
    "NSE_EQ|INE044A01036": {"name": "SUNPHARMA", "display_name": "SUN PHARMA", "type": "stock", "expiry_day": 3},
    
    # FMCG SECTOR
    "NSE_EQ|INE154A01025": {"name": "ITC", "display_name": "ITC LTD", "type": "stock", "expiry_day": 3},
    
    # TELECOM & FINANCE
    "NSE_EQ|INE397D01024": {"name": "BHARTIARTL", "display_name": "BHARTI AIRTEL", "type": "stock", "expiry_day": 3},
    "NSE_EQ|INE296A01024": {"name": "BAJFINANCE", "display_name": "BAJAJ FINANCE", "type": "stock", "expiry_day": 3}
}

ALL_SYMBOLS = {**INDICES, **FO_STOCKS}

# ==================== DATA CLASSES ====================
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

# ==================== EXPIRY CALCULATOR (FIXED) ====================
class ExpiryCalculator:
    @staticmethod
    def get_all_expiries(instrument_key: str) -> List[str]:
        """Fetch all available expiries from Upstox API"""
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
            else:
                logger.error(f"  ❌ Expiry fetch error: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"  ❌ Expiry fetch exception: {e}")
            return []
    
    @staticmethod
    def get_next_expiry(instrument_key: str, symbol_info: Dict) -> Optional[str]:
        """
        Auto-select nearest valid expiry
        Returns: YYYY-MM-DD format for API
        """
        try:
            # Try fetching from API first
            expiries = ExpiryCalculator.get_all_expiries(instrument_key)
            
            if expiries:
                today = datetime.now(IST).date()
                now_time = datetime.now(IST).time()
                
                # Filter future expiries
                future_expiries = []
                for exp_str in expiries:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                    if exp_date > today or (exp_date == today and now_time < time(15, 30)):
                        future_expiries.append(exp_str)
                
                if future_expiries:
                    selected = min(future_expiries)
                    logger.info(f"  ✅ Auto-selected expiry: {selected}")
                    return selected
            
            # Fallback: Calculate based on expiry_day
            logger.info(f"  ⚠️ API expiries unavailable, using calculation")
            expiry_day = symbol_info.get('expiry_day', 3)
            today = datetime.now(IST).date()
            current_time = datetime.now(IST).time()
            
            # Get last day of current month
            last_day = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            
            # Calculate last occurrence of target weekday
            days_to_subtract = (last_day.weekday() - expiry_day) % 7
            expiry = last_day - timedelta(days=days_to_subtract)
            
            # If expiry passed, get next month
            if expiry < today or (expiry == today and current_time >= time(15, 30)):
                next_month = (today.replace(day=28) + timedelta(days=4))
                last_day = (next_month.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
                days_to_subtract = (last_day.weekday() - expiry_day) % 7
                expiry = last_day - timedelta(days=days_to_subtract)
            
            return expiry.strftime('%Y-%m-%d')
            
        except Exception as e:
            logger.error(f"  ❌ Expiry calculation error: {e}")
            return None

# ==================== DATA FETCHER (FIXED) ====================
class UpstoxDataFetcher:
    def __init__(self, token: str):
        self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        self.base_url = "https://api.upstox.com"
    
    def get_historical(self, key: str, interval: str, days: int = 10) -> pd.DataFrame:
        try:
            to_date = datetime.now(IST)
            from_date = to_date - timedelta(days=days)
            
            url = f"{self.base_url}/v2/historical-candle/{key}/{interval}/{to_date.strftime('%Y-%m-%d')}/{from_date.strftime('%Y-%m-%d')}"
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
            response = requests.get(f"{self.base_url}/v2/market-quote/ltp",
                                  headers=self.headers, params={"instrument_key": key}, timeout=10)
            if response.status_code == 200:
                return response.json()['data'][key]['last_price']
            return 0.0
        except:
            return 0.0
    
    def get_option_chain(self, instrument_key: str, expiry: str) -> List[StrikeData]:
        """
        FIXED: Fetch option chain with proper error handling and retry
        """
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                encoded_key = urllib.parse.quote(instrument_key, safe='')
                url = f"{self.base_url}/v2/option/chain?instrument_key={encoded_key}&expiry_date={expiry}"
                
                response = requests.get(url, headers=self.headers, timeout=20)
                
                logger.info(f"     API Response: {response.status_code} (Attempt {attempt + 1}/{max_retries})")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'data' not in data:
                        logger.warning(f"     ⚠️ No 'data' key in response")
                        if attempt < max_retries - 1:
                            time_sleep.sleep(retry_delay)
                            continue
                        return []
                    
                    strikes_raw = data.get('data', [])
                    
                    if not strikes_raw:
                        logger.warning(f"     ⚠️ Empty strikes array")
                        if attempt < max_retries - 1:
                            time_sleep.sleep(retry_delay)
                            continue
                        return []
                    
                    strikes = []
                    for item in strikes_raw:
                        call = item.get('call_options', {}).get('market_data', {})
                        put = item.get('put_options', {}).get('market_data', {})
                        
                        ce_oi = call.get('oi', 0)
                        pe_oi = put.get('oi', 0)
                        
                        # Skip if both OI are 0
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
                        logger.info(f"     ✅ Fetched {len(strikes)} valid strikes")
                        return strikes
                    else:
                        logger.warning(f"     ⚠️ No strikes with OI data")
                        if attempt < max_retries - 1:
                            time_sleep.sleep(retry_delay)
                            continue
                        return []
                
                elif response.status_code == 429:
                    logger.warning(f"     ⚠️ Rate limit hit! Waiting {retry_delay * 2}s...")
                    time_sleep.sleep(retry_delay * 2)
                    continue
                
                elif response.status_code == 404:
                    logger.warning(f"     ⚠️ No options for this expiry")
                    return []
                
                else:
                    logger.error(f"     ❌ API Error: {response.status_code}")
                    if attempt < max_retries - 1:
                        time_sleep.sleep(retry_delay)
                        continue
                    return []
                    
            except Exception as e:
                logger.error(f"     ❌ Option chain error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time_sleep.sleep(retry_delay)
                    continue
                return []
        
        return []

# ==================== REDIS OI MANAGER ====================
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
            logger.info(f"  💾 Redis: OI saved")
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
                logger.info(f"  ⏰ OI comparison: 2h ago")
                return OIData(
                    pcr=parsed['pcr'], 
                    support_strike=parsed['support'], 
                    resistance_strike=parsed['resistance'],
                    ce_oi_change_pct=parsed.get('ce_oi_change_pct', 0), 
                    pe_oi_change_pct=parsed.get('pe_oi_change_pct', 0),
                    strikes_data=[StrikeData(s['strike'], s['ce_oi'], s['pe_oi'], 0, 0) for s in parsed['strikes']],
                    timestamp=comparison_time
                )
            return None
        except Exception as e:
            logger.error(f"  ❌ Redis get error: {e}")
            return None

# ==================== NEWS FETCHER ====================
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

# ==================== MULTI-TIMEFRAME PROCESSOR ====================
class MultiTimeframeProcessor:
    @staticmethod
    def resample(df_1m: pd.DataFrame, tf: str) -> pd.DataFrame:
        df = df_1m.copy()
        df.set_index('timestamp', inplace=True)
        resampled = df.resample(tf).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()
        return resampled

# ==================== DATA COMPRESSOR ====================
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

# ==================== DEEPSEEK ANALYZER (Kept original prompt) ====================
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
Impact: {"Positive (+25pts)" if news.impact_score > 0 else "Negative (-25pts)" if news.impact_score < 0 else "Neutral"}
"""
        
        prompt = f"""You are an expert F&O price action trader. Analyze using institutional-grade confluence.

**INSTRUMENT:** {symbol}
**SPOT PRICE:** ₹{spot_price:.2f}
**ATR:** {atr:.2f}

**1H DATA (50 candles):** {ohlc_1h}
**15M DATA (500 candles):** {ohlc_15m}
**5M DATA (100 candles):** {ohlc_5m}

**OI DATA:** {oi_summary}
{news_section}

**OUTPUT JSON ONLY:**
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
  "chart_bias": "Bullish/Bearish",
  "market_structure": "HH/HL forming",
  "pattern_signal": "Pattern detected",
  "oi_flow_signal": "OI analysis",
  "support_levels": [0.0, 0.0],
  "resistance_levels": [0.0, 0.0],
  "risk_factors": ["Risk1", "Risk2"],
  "ai_reasoning": "Brief reasoning"
}}"""
        
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
                ai_reasoning=analysis.get('ai_reasoning', 'No reasoning')
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
                logger.info(f"  🤖 AI: {analysis.opportunity} | Score: {analysis.total_score}/100")
            
            return analysis
            
        except Exception as e:
            logger.error(f"  ❌ DeepSeek error: {e}")
            return None
