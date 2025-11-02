#!/usr/bin/env python3
"""
HYBRID TRADING BOT v24.0 - GOLDEN SETUP FINDER
===============================================
✅ Multi-Timeframe: 1H (Trend) + 15M (Analysis) + 5M (Entry/Exit)
✅ Golden Setup Finder (4-Signal Confluence)
✅ Token-Optimized (Compressed Data)
✅ Corrected OI Logic
✅ Professional Chart Generation
"""

import os
import asyncio
import requests
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
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import traceback
import re
import redis

# ==================== CONFIGURATION ====================
IST = pytz.timezone('Asia/Kolkata')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('hybrid_bot.log')]
)
logger = logging.getLogger(__name__)

# API Keys
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN', 'your_token')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'your_key')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', 'your_key')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'your_token')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'your_chat_id')

# Redis
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=5)

# ==================== SYMBOLS CONFIG ====================
INDICES = {
    "NSE_INDEX|Nifty Bank": {"name": "BANKNIFTY", "display_name": "BANK NIFTY", "type": "index"},
    "NSE_INDEX|Nifty Midcap Select": {"name": "MIDCPNIFTY", "display_name": "MIDCAP NIFTY", "type": "index"}
}

FO_STOCKS = {
    # AUTO SECTOR
    "NSE_EQ|INE467B01029": {"name": "TATAMOTORS", "display_name": "TATA MOTORS", "type": "stock"},
    "NSE_EQ|INE585B01010": {"name": "MARUTI", "display_name": "MARUTI SUZUKI", "type": "stock"},
    "NSE_EQ|INE208A01029": {"name": "ASHOKLEY", "display_name": "ASHOK LEYLAND", "type": "stock"},
    "NSE_EQ|INE494B01023": {"name": "TVSMOTOR", "display_name": "TVS MOTOR", "type": "stock"},
    "NSE_EQ|INE101A01026": {"name": "M&M", "display_name": "M&M", "type": "stock"},
    "NSE_EQ|INE917I01010": {"name": "BAJAJ-AUTO", "display_name": "BAJAJ AUTO", "type": "stock"},
    
    # BANKING SECTOR
    "NSE_EQ|INE040A01034": {"name": "HDFCBANK", "display_name": "HDFC BANK", "type": "stock"},
    "NSE_EQ|INE090A01021": {"name": "ICICIBANK", "display_name": "ICICI BANK", "type": "stock"},
    "NSE_EQ|INE062A01020": {"name": "SBIN", "display_name": "STATE BANK", "type": "stock"},
    "NSE_EQ|INE028A01039": {"name": "BANKBARODA", "display_name": "BANK OF BARODA", "type": "stock"},
    "NSE_EQ|INE238A01034": {"name": "AXISBANK", "display_name": "AXIS BANK", "type": "stock"},
    "NSE_EQ|INE237A01028": {"name": "KOTAKBANK", "display_name": "KOTAK BANK", "type": "stock"},
    
    # METALS SECTOR
    "NSE_EQ|INE155A01022": {"name": "TATASTEEL", "display_name": "TATA STEEL", "type": "stock"},
    "NSE_EQ|INE205A01025": {"name": "HINDALCO", "display_name": "HINDALCO", "type": "stock"},
    "NSE_EQ|INE019A01038": {"name": "JSWSTEEL", "display_name": "JSW STEEL", "type": "stock"},
    
    # ENERGY SECTOR
    "NSE_EQ|INE002A01018": {"name": "RELIANCE", "display_name": "RELIANCE IND", "type": "stock"},
    "NSE_EQ|INE213A01029": {"name": "ONGC", "display_name": "ONGC", "type": "stock"},
    "NSE_EQ|INE242A01010": {"name": "IOC", "display_name": "INDIAN OIL", "type": "stock"},
    
    # IT SECTOR
    "NSE_EQ|INE009A01021": {"name": "INFY", "display_name": "INFOSYS", "type": "stock"},
    "NSE_EQ|INE075A01022": {"name": "WIPRO", "display_name": "WIPRO", "type": "stock"},
    "NSE_EQ|INE467B01029": {"name": "TCS", "display_name": "TCS", "type": "stock"},
    "NSE_EQ|INE047A01021": {"name": "HCLTECH", "display_name": "HCL TECH", "type": "stock"},
    
    # PHARMA SECTOR
    "NSE_EQ|INE044A01036": {"name": "SUNPHARMA", "display_name": "SUN PHARMA", "type": "stock"},
    "NSE_EQ|INE361B01024": {"name": "DIVISLAB", "display_name": "DIVI'S LAB", "type": "stock"},
    "NSE_EQ|INE089A01023": {"name": "DRREDDY", "display_name": "DR REDDY", "type": "stock"},
    
    # FMCG SECTOR
    "NSE_EQ|INE154A01025": {"name": "ITC", "display_name": "ITC LTD", "type": "stock"},
    "NSE_EQ|INE030A01027": {"name": "HUL", "display_name": "HINDUSTAN UNILEVER", "type": "stock"},
    "NSE_EQ|INE216A01030": {"name": "BRITANNIA", "display_name": "BRITANNIA", "type": "stock"},
    
    # INFRASTRUCTURE
    "NSE_EQ|INE742F01042": {"name": "ADANIPORTS", "display_name": "ADANI PORTS", "type": "stock"},
    "NSE_EQ|INE733E01010": {"name": "NTPC", "display_name": "NTPC", "type": "stock"},
    "NSE_EQ|INE018A01030": {"name": "LT", "display_name": "L&T", "type": "stock"},
    
    # RETAIL & CONSUMER
    "NSE_EQ|INE280A01028": {"name": "TITAN", "display_name": "TITAN", "type": "stock"},
    "NSE_EQ|INE849A01020": {"name": "TRENT", "display_name": "TRENT", "type": "stock"},
    "NSE_EQ|INE021A01026": {"name": "ASIANPAINT", "display_name": "ASIAN PAINTS", "type": "stock"},
    
    # TELECOM & FINANCE
    "NSE_EQ|INE397D01024": {"name": "BHARTIARTL", "display_name": "BHARTI AIRTEL", "type": "stock"},
    "NSE_EQ|INE296A01024": {"name": "BAJFINANCE", "display_name": "BAJAJ FINANCE", "type": "stock"}
}

ALL_SYMBOLS = {**INDICES, **FO_STOCKS}

NSE_HOLIDAYS_2025 = [
    '2025-01-26', '2025-03-14', '2025-03-31', '2025-04-10', '2025-04-14', '2025-04-18',
    '2025-05-01', '2025-08-15', '2025-10-02', '2025-12-25'
]

# ==================== DATA CLASSES ====================
@dataclass
class StrikeData:
    strike: int
    ce_oi: int
    pe_oi: int
    ce_volume: int
    pe_volume: int
    ce_price: float
    pe_price: float

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
class GoldenSetup:
    signal: str  # "HIGH_PROB_BUY" / "HIGH_PROB_SELL" / "WAIT"
    confidence: int
    zone_price: float
    trigger_pattern: str
    oi_confirmation: str
    confluence_count: int
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    reason: str
    chart_support: float
    chart_resistance: float

@dataclass
class MultiTimeframeData:
    df_1h: pd.DataFrame
    df_15m: pd.DataFrame
    df_5m: pd.DataFrame
    spot_price: float
    atr: float
    trend_1h: str
    trend_1h_confidence: int

# ==================== EXPIRY CALCULATOR ====================
class ExpiryCalculator:
    @staticmethod
    def get_monthly_expiry(symbol_name: str) -> str:
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
        
        return expiry.strftime('%d%b%y').upper()

# ==================== REDIS OI MANAGER ====================
class RedisOIManager:
    @staticmethod
    def save_oi(symbol: str, expiry: str, oi_data: OIData):
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
    
    @staticmethod
    def get_comparison_oi(symbol: str, expiry: str, current_time: datetime) -> Optional[OIData]:
        two_hours_ago = current_time - timedelta(hours=2)
        comparison_time = two_hours_ago.replace(minute=(two_hours_ago.minute // 15) * 15, second=0, microsecond=0)
        
        key = f"oi:{symbol}:{expiry}:{comparison_time.strftime('%Y-%m-%d_%H:%M')}"
        data = redis_client.get(key)
        
        if data:
            parsed = json.loads(data)
            return OIData(
                pcr=parsed['pcr'], support_strike=parsed['support'], resistance_strike=parsed['resistance'],
                ce_oi_change_pct=parsed.get('ce_oi_change_pct', 0), pe_oi_change_pct=parsed.get('pe_oi_change_pct', 0),
                strikes_data=[StrikeData(s['strike'], s['ce_oi'], s['pe_oi'], 0, 0, 0, 0) for s in parsed['strikes']],
                timestamp=comparison_time
            )
        return None

# ==================== GOLDEN SETUP FINDER ====================
class GoldenSetupFinder:
    @staticmethod
    def find_support_resistance(df: pd.DataFrame, spot_price: float) -> Dict:
        """Chart वरून Support/Resistance zones शोधणे"""
        df_tail = df.tail(50)
        
        support_zones = []
        resistance_zones = []
        
        for i in range(2, len(df_tail)-2):
            # Swing Low
            if (df_tail.iloc[i]['low'] < df_tail.iloc[i-1]['low'] and 
                df_tail.iloc[i]['low'] < df_tail.iloc[i-2]['low'] and
                df_tail.iloc[i]['low'] < df_tail.iloc[i+1]['low'] and
                df_tail.iloc[i]['low'] < df_tail.iloc[i+2]['low']):
                support_zones.append(df_tail.iloc[i]['low'])
            
            # Swing High
            if (df_tail.iloc[i]['high'] > df_tail.iloc[i-1]['high'] and 
                df_tail.iloc[i]['high'] > df_tail.iloc[i-2]['high'] and
                df_tail.iloc[i]['high'] > df_tail.iloc[i+1]['high'] and
                df_tail.iloc[i]['high'] > df_tail.iloc[i+2]['high']):
                resistance_zones.append(df_tail.iloc[i]['high'])
        
        nearest_support = max([s for s in support_zones if s < spot_price], default=spot_price * 0.98)
        nearest_resistance = min([r for r in resistance_zones if r > spot_price], default=spot_price * 1.02)
        
        return {"support": nearest_support, "resistance": nearest_resistance}
    
    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> List[str]:
        """Last 3 candles मधून patterns शोधणे"""
        patterns = []
        recent = df.tail(3)
        
        if len(recent) < 2:
            return ["NONE"]
        
        for i in range(1, len(recent)):
            prev = recent.iloc[i-1]
            curr = recent.iloc[i]
            
            body = abs(curr['close'] - curr['open'])
            lower_shadow = (curr['open'] if curr['close'] > curr['open'] else curr['close']) - curr['low']
            upper_shadow = curr['high'] - max(curr['open'], curr['close'])
            
            if lower_shadow > body * 2 and upper_shadow < body * 0.5:
                patterns.append(f"HAMMER@{curr['close']:.0f}")
            if upper_shadow > body * 2 and lower_shadow < body * 0.5:
                patterns.append(f"SHOOTING_STAR@{curr['close']:.0f}")
            if (curr['close'] > curr['open'] and prev['close'] < prev['open'] and
                curr['open'] < prev['close'] and curr['close'] > prev['open']):
                patterns.append(f"BULLISH_ENGULF@{curr['close']:.0f}")
            if (curr['close'] < curr['open'] and prev['close'] > prev['open'] and
                curr['open'] > prev['close'] and curr['close'] < prev['open']):
                patterns.append(f"BEARISH_ENGULF@{curr['close']:.0f}")
        
        return patterns if patterns else ["NONE"]
    
    @staticmethod
    def check_oi_confluence(patterns: List[str], current_oi: OIData, prev_oi: Optional[OIData]) -> Tuple[str, int]:
        """OI flow ने pattern confirm करतोय का? (CORRECTED LOGIC)"""
        if not prev_oi or patterns[0] == "NONE":
            return "NO_CONFIRMATION", 0
        
        pattern = patterns[0].split('@')[0]
        
        # BULLISH PATTERNS
        if pattern in ["HAMMER", "BULLISH_ENGULF"]:
            if current_oi.ce_oi_change_pct < -5:  # Resistance weakening
                return "✅ STRONG: CE_UNWIND (Resistance breaking)", 20
            elif current_oi.pe_oi_change_pct > 10:  # Support strengthening
                return "✅ MODERATE: PE_BUILD (Support forming)", 15
            else:
                return "⚠️ WEAK: OI neutral", 0
        
        # BEARISH PATTERNS
        elif pattern in ["SHOOTING_STAR", "BEARISH_ENGULF"]:
            if current_oi.pe_oi_change_pct < -5:  # Support weakening
                return "✅ STRONG: PE_UNWIND (Support breaking)", 20
            elif current_oi.ce_oi_change_pct > 10:  # Resistance strengthening
                return "✅ MODERATE: CE_BUILD (Resistance forming)", 15
            else:
                return "⚠️ WEAK: OI neutral", 0
        
        return "NEUTRAL", 0
    
    @staticmethod
    def find_golden_setup(df_15m: pd.DataFrame, df_5m: pd.DataFrame, spot_price: float, 
                         atr: float, current_oi: OIData, prev_oi: Optional[OIData],
                         trend_1h: str) -> GoldenSetup:
        """🎯 Main Engine: 4-Signal Confluence"""
        
        # STEP 1: Chart Zones (15M)
        zones = GoldenSetupFinder.find_support_resistance(df_15m, spot_price)
        chart_support = zones['support']
        chart_resistance = zones['resistance']
        
        # Zone Confluence Check
        support_match = abs(chart_support - current_oi.support_strike) < 100
        resistance_match = abs(chart_resistance - current_oi.resistance_strike) < 100
        
        distance_support = abs(spot_price - chart_support)
        distance_resistance = abs(spot_price - chart_resistance)
        
        at_support = distance_support < atr * 1.5 and support_match
        at_resistance = distance_resistance < atr * 1.5 and resistance_match
        
        if not (at_support or at_resistance):
            return GoldenSetup(
                signal="WAIT", confidence=0, zone_price=0, trigger_pattern="NONE",
                oi_confirmation="Price not in confluence zone", confluence_count=0,
                entry_price=spot_price, stop_loss=0, target_1=0, target_2=0,
                reason="Not at High Probability Zone", chart_support=chart_support,
                chart_resistance=chart_resistance
            )
        
        # STEP 2: Trigger Pattern (15M)
        patterns = GoldenSetupFinder.detect_patterns(df_15m)
        
        if patterns[0] == "NONE":
            zone_name = "Support" if at_support else "Resistance"
            return GoldenSetup(
                signal="WAIT", confidence=0, zone_price=chart_support if at_support else chart_resistance,
                trigger_pattern="NONE", oi_confirmation="Waiting for trigger",
                confluence_count=2, entry_price=spot_price, stop_loss=0, target_1=0, target_2=0,
                reason=f"At {zone_name} Zone (Chart+OI) but no pattern", 
                chart_support=chart_support, chart_resistance=chart_resistance
            )
        
        pattern_name = patterns[0].split('@')[0]
        bullish_trigger = pattern_name in ["HAMMER", "BULLISH_ENGULF"]
        bearish_trigger = pattern_name in ["SHOOTING_STAR", "BEARISH_ENGULF"]
        
        # Pattern-Zone match
        if (at_support and not bullish_trigger) or (at_resistance and not bearish_trigger):
            return GoldenSetup(
                signal="WAIT", confidence=0, zone_price=chart_support if at_support else chart_resistance,
                trigger_pattern=patterns[0], oi_confirmation="Pattern mismatch",
                confluence_count=2, entry_price=spot_price, stop_loss=0, target_1=0, target_2=0,
                reason=f"Pattern-Zone mismatch: {pattern_name} at {'Support' if at_support else 'Resistance'}",
                chart_support=chart_support, chart_resistance=chart_resistance
            )
        
        # STEP 3: OI Confirmation
        oi_confirm, bonus = GoldenSetupFinder.check_oi_confluence(patterns, current_oi, prev_oi)
        
        if bonus < 10:
            return GoldenSetup(
                signal="WAIT", confidence=0, zone_price=chart_support if at_support else chart_resistance,
                trigger_pattern=patterns[0], oi_confirmation=oi_confirm,
                confluence_count=3, entry_price=spot_price, stop_loss=0, target_1=0, target_2=0,
                reason=f"Zone+Pattern OK but {oi_confirm}",
                chart_support=chart_support, chart_resistance=chart_resistance
            )
        
        # STEP 4: Trend Filter (1H must align)
        if bullish_trigger and trend_1h == "BEARISH":
            return GoldenSetup(
                signal="WAIT", confidence=0, zone_price=chart_support,
                trigger_pattern=patterns[0], oi_confirmation=oi_confirm,
                confluence_count=3, entry_price=spot_price, stop_loss=0, target_1=0, target_2=0,
                reason=f"Bullish setup but 1H trend is BEARISH (counter-trend)",
                chart_support=chart_support, chart_resistance=chart_resistance
            )
        
        if bearish_trigger and trend_1h == "BULLISH":
            return GoldenSetup(
                signal="WAIT", confidence=0, zone_price=chart_resistance,
                trigger_pattern=patterns[0], oi_confirmation=oi_confirm,
                confluence_count=3, entry_price=spot_price, stop_loss=0, target_1=0, target_2=0,
                reason=f"Bearish setup but 1H trend is BULLISH (counter-trend)",
                chart_support=chart_support, chart_resistance=chart_resistance
            )
        
        # 🎯 GOLDEN SETUP!
        signal = "HIGH_PROB_BUY" if bullish_trigger else "HIGH_PROB_SELL"
        zone_price = chart_support if at_support else chart_resistance
        
        # Entry/SL/Target from 5M chart
        recent_5m = df_5m.tail(10)
        entry = spot_price
        
        if bullish_trigger:
            sl = min(recent_5m['low']) - (atr * 0.3)
            risk = entry - sl
            target_1 = entry + (risk * 2)
            target_2 = entry + (risk * 3.5)
        else:
            sl = max(recent_5m['high']) + (atr * 0.3)
            risk = sl - entry
            target_1 = entry - (risk * 2)
            target_2 = entry - (risk * 3.5)
        
        confidence = 95 if bonus >= 20 else 85
        
        reason = (
            f"✅ 4-SIGNAL CONFLUENCE:\n"
            f"1. 1H Trend: {trend_1h}\n"
            f"2. Chart Zone: ₹{zone_price:.0f} (15M)\n"
            f"3. OI Zone: {current_oi.support_strike if at_support else current_oi.resistance_strike}\n"
            f"4. Trigger: {pattern_name} (15M)\n"
            f"5. OI Flow: {oi_confirm}\n"
            f"6. Entry/Exit: 5M precision"
        )
        
        return GoldenSetup(
            signal=signal, confidence=confidence, zone_price=zone_price,
            trigger_pattern=patterns[0], oi_confirmation=oi_confirm,
            confluence_count=4, entry_price=entry, stop_loss=sl,
            target_1=target_1, target_2=target_2, reason=reason,
            chart_support=chart_support, chart_resistance=chart_resistance
        )

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
    
    @staticmethod
    def get_trend(df: pd.DataFrame) -> Tuple[str, int]:
        """1H Trend identification"""
        if len(df) < 20:
            return "NEUTRAL", 50
        
        closes = df.tail(20)['close'].values
        sma_20 = closes.mean()
        current = closes[-1]
        deviation = ((current - sma_20) / sma_20) * 100
        
        highs = df.tail(10)['high'].values
        hh = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        
        if deviation > 1 and hh >= 6:
            return "BULLISH", min(95, 60 + int(deviation * 5))
        elif deviation < -1:
            return "BEARISH", min(95, 60 + int(abs(deviation) * 5))
        else:
            return "NEUTRAL", 50

# ==================== DATA COMPRESSOR ====================
class DataCompressor:
    @staticmethod
    def compress_candles(df: pd.DataFrame) -> str:
        """Token-efficient summary"""
        recent = df.tail(20)
        bullish = sum(1 for _, r in recent.iterrows() if r['close'] > r['open'])
        bearish = len(recent) - bullish
        
        highs = recent['high'].values
        lows = recent['low'].values
        hh = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        ll = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        
        current = df['close'].iloc[-1]
        sma20 = df['close'].tail(20).mean()
        
        return f"PRICE:{current:.1f} ({'ABOVE' if current>sma20 else 'BELOW'} SMA20={sma20:.1f})|STRUCT:{bullish}🟢{bearish}🔴|HH:{hh}LL:{ll}"
    
    @staticmethod
    def compress_oi(current: OIData, prev: Optional[OIData]) -> str:
        """Compressed OI insight"""
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
        
        result = f"PCR:{current.pcr:.2f}|"
        if ce_builds: result += f"CE_BUILD:{','.join(map(str, ce_builds[:2]))}|"
        if pe_builds: result += f"PE_BUILD:{','.join(map(str, pe_builds[:2]))}|"
        if ce_unwinds: result += f"CE_UNWIND:{','.join(map(str, ce_unwinds[:2]))}|"
        if pe_unwinds: result += f"PE_UNWIND:{','.join(map(str, pe_unwinds[:2]))}"
        
        return result.strip('|')

# ==================== CHART GENERATOR ====================
class ChartGenerator:
    @staticmethod
    def create_chart(symbol: str, df: pd.DataFrame, setup: GoldenSetup, spot: float, path: str):
        BG, GRID, TEXT = '#131722', '#1e222d', '#d1d4dc'
        GREEN, RED, YELLOW = '#26a69a', '#ef5350', '#ffd700'
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [3, 1]}, facecolor=BG)
        
        ax1.set_facecolor(BG)
        df_plot = df.tail(100).reset_index(drop=True)
        
        # Candles
        for idx, row in df_plot.iterrows():
            color = GREEN if row['close'] > row['open'] else RED
            ax1.add_patch(Rectangle((idx, min(row['open'], row['close'])), 0.6,
                                   abs(row['close'] - row['open']), facecolor=color, alpha=0.8))
            ax1.plot([idx+0.3, idx+0.3], [row['low'], row['high']], color=color, linewidth=1, alpha=0.6)
        
        # Zones
        ax1.axhline(setup.chart_support, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.axhline(setup.chart_resistance, color=RED, linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Entry/SL/Targets
        if setup.signal != "WAIT":
            ax1.scatter([len(df_plot)-1], [setup.entry_price], color=YELLOW, s=300, marker='D', zorder=5)
            ax1.axhline(setup.stop_loss, color=RED, linewidth=2, linestyle=':')
            ax1.axhline(setup.target_1, color=GREEN, linewidth=2, linestyle=':')
            ax1.axhline(setup.target_2, color=GREEN, linewidth=1.5, linestyle=':')
        
        # Info Box
        info = f"""{'🟢 BUY' if setup.signal=='HIGH_PROB_BUY' else '🔴 SELL' if setup.signal=='HIGH_PROB_SELL' else '⏸️ WAIT'}
Conf: {setup.confidence}%
Zone: ₹{setup.zone_price:.0f}
Pattern: {setup.trigger_pattern.split('@')[0]}
OI: {setup.oi_confirmation[:30]}

Entry: ₹{setup.entry_price:.1f}
SL: ₹{setup.stop_loss:.1f}
T1: ₹{setup.target_1:.1f}
T2: ₹{setup.target_2:.1f}"""
        
        ax1.text(0.01, 0.99, info, transform=ax1.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor=GRID, alpha=0.95), color=TEXT, family='monospace')
        
        ax1.set_title(f"{symbol} | 15M | Golden Setup Finder", color=TEXT, fontsize=13, fontweight='bold')
        ax1.grid(True, color=GRID, alpha=0.3)
        ax1.tick_params(colors=TEXT)
        
        # Volume
        ax2.set_facecolor(BG)
        colors = [GREEN if df_plot.iloc[i]['close']>df_plot.iloc[i]['open'] else RED for i in range(len(df_plot))]
        ax2.bar(range(len(df_plot)), df_plot['volume'], color=colors, alpha=0.6)
        ax2.tick_params(colors=TEXT)
        ax2.grid(True, color=GRID, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, facecolor=BG)
        plt.close()

# ==================== DATA FETCHER ====================
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
        try:
            response = requests.get("https://api.upstox.com/v2/option/chain",
                                  headers=self.headers, 
                                  params={"instrument_key": key, "expiry_date": expiry}, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                strikes = []
                for item in data.get('data', []):
                    call = item.get('call_options', {}).get('market_data', {})
                    put = item.get('put_options', {}).get('market_data', {})
                    strikes.append(StrikeData(
                        strike=int(item.get('strike_price', 0)),
                        ce_oi=call.get('oi', 0), pe_oi=put.get('oi', 0),
                        ce_volume=call.get('volume', 0), pe_volume=put.get('volume', 0),
                        ce_price=call.get('ltp', 0), pe_price=put.get('ltp', 0)
                    ))
                return strikes
            return []
        except:
            return []

# ==================== MAIN BOT ====================
class HybridBot:
    def __init__(self):
        self.data_fetcher = UpstoxDataFetcher(UPSTOX_ACCESS_TOKEN)
        self.telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.processed_signals = set()
    
    async def send_startup_message(self):
        message = f"""
🚀 **HYBRID BOT v24.0 - GOLDEN SETUP FINDER**

⏰ **Time:** {datetime.now(IST).strftime('%d-%b-%Y %H:%M:%S')}

📊 **Features:**
✅ Multi-Timeframe: 1H (Trend) + 15M (Analysis) + 5M (Entry)
✅ Golden Setup Finder (4-Signal Confluence)
✅ Token-Optimized AI Prompts
✅ Corrected OI Logic
✅ Professional Charts

📈 **Monitoring:**
- 2 Indices + 5 F&O Stocks

🎯 **Alert Criteria:**
- 4/4 Confluence Signals
- Confidence ≥ 85%

📡 **Scan Interval:** 15 minutes
🔄 **Status:** Active
"""
        await self.telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
        logger.info("✅ Startup message sent")
    
    async def send_alert(self, symbol: str, setup: GoldenSetup, chart_path: str, 
                        trend_1h: str, price_summary: str, oi_summary: str):
        try:
            # Send Chart
            with open(chart_path, 'rb') as photo:
                await self.telegram_bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo)
            
            # Send Details
            risk = abs(setup.entry_price - setup.stop_loss)
            reward1 = abs(setup.target_1 - setup.entry_price)
            rr = reward1 / risk if risk > 0 else 0
            
            message = f"""
🚨 **{symbol} {setup.signal}**

📊 **Golden Setup Detected!**
Confidence: {setup.confidence}%
Confluence: {setup.confluence_count}/4 signals

📈 **Multi-Timeframe Analysis:**
1H Trend: {trend_1h}
15M Zone: ₹{setup.zone_price:.0f}
15M Pattern: {setup.trigger_pattern.split('@')[0]}
5M Entry: ₹{setup.entry_price:.2f}

📊 **Price Action (15M):**
{price_summary}

📊 **OI Flow:**
{oi_summary}
{setup.oi_confirmation}

💰 **Trade Setup:**
Entry: ₹{setup.entry_price:.2f}
Stop Loss: ₹{setup.stop_loss:.2f}
Target 1: ₹{setup.target_1:.2f}
Target 2: ₹{setup.target_2:.2f}
Risk:Reward: 1:{rr:.1f}

📝 **Confluence:**
{setup.reason}

🕐 {datetime.now(IST).strftime('%d-%b %H:%M:%S')}
"""
            
            await self.telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
            logger.info(f"  ✅ Alert sent for {symbol}")
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    async def analyze_symbol(self, instrument_key: str, symbol_info: Dict):
        try:
            symbol_name = symbol_info['name']
            display_name = symbol_info['display_name']
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 {display_name} ({symbol_name})")
            logger.info(f"{'='*70}")
            
            # 1. Expiry
            expiry = ExpiryCalculator.get_monthly_expiry(symbol_name)
            logger.info(f"  📅 Expiry: {expiry}")
            
            # 2. Fetch 1-min data (10 days)
            df_1m = self.data_fetcher.get_historical(instrument_key, "1minute", days=10)
            if df_1m.empty:
                logger.warning(f"  ⚠️ No data")
                return
            
            # 3. Resample to timeframes
            df_1h = MultiTimeframeProcessor.resample(df_1m, '1H')
            df_15m = MultiTimeframeProcessor.resample(df_1m, '15T')
            df_5m = MultiTimeframeProcessor.resample(df_1m, '5T')
            
            logger.info(f"  📊 Data: 1H({len(df_1h)}) | 15M({len(df_15m)}) | 5M({len(df_5m)})")
            
            # 4. Get 1H Trend
            trend_1h, conf_1h = MultiTimeframeProcessor.get_trend(df_1h)
            logger.info(f"  📊 1H Trend: {trend_1h} ({conf_1h}%)")
            
            # 5. Spot Price & ATR
            spot_price = self.data_fetcher.get_ltp(instrument_key)
            if spot_price == 0:
                spot_price = df_15m['close'].iloc[-1]
            
            df_15m['tr'] = df_15m[['high', 'low', 'close']].apply(
                lambda x: max(x['high']-x['low'], abs(x['high']-x['close']), abs(x['low']-x['close'])), axis=1
            )
            atr = df_15m['tr'].rolling(14).mean().iloc[-1]
            logger.info(f"  💹 Spot: ₹{spot_price:.2f} | ATR: {atr:.2f}")
            
            # 6. Option Chain
            all_strikes = self.data_fetcher.get_option_chain(instrument_key, expiry)
            if not all_strikes:
                logger.warning(f"  ⚠️ No OI data")
                return
            
            # Get top 15 ATM strikes
            atm = round(spot_price / 100) * 100
            atm_range = range(atm - 700, atm + 800, 100)
            top_15 = sorted([s for s in all_strikes if s.strike in atm_range],
                          key=lambda x: (x.ce_oi + x.pe_oi), reverse=True)[:15]
            
            # 7. OI Analysis
            total_ce = sum(s.ce_oi for s in top_15)
            total_pe = sum(s.pe_oi for s in top_15)
            pcr = total_pe / total_ce if total_ce > 0 else 0
            
            max_ce_strike = max(top_15, key=lambda x: x.ce_oi).strike
            max_pe_strike = max(top_15, key=lambda x: x.pe_oi).strike
            
            # Get previous OI
            prev_oi = RedisOIManager.get_comparison_oi(symbol_name, expiry, datetime.now(IST))
            
            # Calculate changes
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
            
            logger.info(f"  📊 PCR: {pcr:.2f} | S: {max_pe_strike} | R: {max_ce_strike} | CE: {ce_change_pct:+.1f}% | PE: {pe_change_pct:+.1f}%")
            
            # 8. Save OI
            RedisOIManager.save_oi(symbol_name, expiry, current_oi)
            
            # 9. Golden Setup Finder
            setup = GoldenSetupFinder.find_golden_setup(
                df_15m=df_15m,
                df_5m=df_5m,
                spot_price=spot_price,
                atr=atr,
                current_oi=current_oi,
                prev_oi=prev_oi,
                trend_1h=trend_1h
            )
            
            logger.info(f"  🎯 Setup: {setup.signal} | Conf: {setup.confidence}% | Confluence: {setup.confluence_count}/4")
            logger.info(f"     {setup.reason.split(chr(10))[0]}")
            
            # 10. Check alert threshold
            if setup.signal != "WAIT" and setup.confidence >= 85 and setup.confluence_count == 4:
                signal_key = f"{symbol_name}_{setup.signal}_{datetime.now(IST).strftime('%Y%m%d_%H')}"
                
                if signal_key not in self.processed_signals:
                    logger.info(f"  🚨 GOLDEN SETUP ALERT!")
                    
                    # Generate chart
                    chart_path = f"/tmp/{symbol_name}_golden.png"
                    ChartGenerator.create_chart(display_name, df_15m, setup, spot_price, chart_path)
                    
                    # Compress data for message
                    price_summary = DataCompressor.compress_candles(df_15m)
                    oi_summary = DataCompressor.compress_oi(current_oi, prev_oi)
                    
                    # Send alert
                    await self.send_alert(display_name, setup, chart_path, trend_1h, price_summary, oi_summary)
                    self.processed_signals.add(signal_key)
                else:
                    logger.info(f"  ⏭️ Already alerted")
            else:
                logger.info(f"  ⏸️ No alert: {'Low confidence' if setup.confidence < 85 else 'Incomplete confluence'}")
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            traceback.print_exc()
    
    async def run_scanner(self):
        logger.info("\n" + "="*80)
        logger.info("🚀 HYBRID BOT v24.0 - GOLDEN SETUP FINDER")
        logger.info("="*80)
        
        await self.send_startup_message()
        
        while True:
            try:
                now = datetime.now(IST)
                current_time = now.time()
                
                # Market hours
                if current_time < time(9, 15) or current_time > time(15, 30):
                    logger.info(f"⏸️ Market closed. Waiting...")
                    await asyncio.sleep(300)
                    continue
                
                # Holidays
                if now.strftime('%Y-%m-%d') in NSE_HOLIDAYS_2025 or now.weekday() >= 5:
                    logger.info(f"📅 Holiday. Pausing...")
                    await asyncio.sleep(3600)
                    continue
                
                logger.info(f"\n🔄 Scan started: {now.strftime('%H:%M:%S')}")
                
                # Scan all symbols
                for instrument_key, symbol_info in ALL_SYMBOLS.items():
                    await self.analyze_symbol(instrument_key, symbol_info)
                    await asyncio.sleep(2)
                
                logger.info(f"\n✅ Scan complete. Next in 15 min...")
                await asyncio.sleep(900)
                
            except Exception as e:
                logger.error(f"Scanner error: {e}")
                traceback.print_exc()
                await asyncio.sleep(60)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    bot = HybridBot()
    asyncio.run(bot.run_scanner())
