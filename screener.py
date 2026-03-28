"""
長期投資信託スクリーニング 全26戦略（A-1〜I-3）
対象: 国内ETF（東証）/ 海外ETF（NYSE・NASDAQ）/ 投資信託
長期投資重視: コスト・リターン・リスク調整済パフォーマンス・AUM
参照: screening_thresholds.txt
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import os
import io
import json
import shutil
from datetime import datetime

# ── .env 読み込み ──────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and "=" in _line and not _line.startswith("#"):
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── OpenAI ────────────────────────────────
_OPENAI_CLIENT = None
try:
    import openai as _openai_mod
    _oai_key = os.environ.get("OPENAI_API_KEY", "")
    if _oai_key:
        _OPENAI_CLIENT = _openai_mod.OpenAI(api_key=_oai_key)
        print("[GPT] OpenAI クライアント初期化完了")
except Exception as _e:
    print(f"[GPT] OpenAI 初期化スキップ: {_e}")

_GPT_CACHE: dict = {}

# ── 翻訳 ──────────────────────────────────
from deep_translator import GoogleTranslator

# ── PDF 生成 ──────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Table, TableStyle,
                                 Spacer, PageBreak, HRFlowable, KeepTogether)
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# 日本語フォント設定
_FONT_CANDIDATES = [
    ("/mnt/c/Windows/Fonts/YuGothR.ttc",  "YuGothic",     "YuGothic"),
    ("/mnt/c/Windows/Fonts/YuGothB.ttc",  "YuGothicBold", "YuGothicBold"),
    ("/mnt/c/Windows/Fonts/BIZ-UDGothicR.ttc", "BIZGothic",     "BIZGothic"),
    ("/mnt/c/Windows/Fonts/BIZ-UDGothicB.ttc", "BIZGothicBold", "BIZGothicBold"),
]
_FONT_NORMAL = "HeiseiKakuGo-W5"
_FONT_BOLD   = "HeiseiKakuGo-W5"
_using_ttf   = False

_reg_normal, _reg_bold = None, None
_FONT_PATH = None
for _path, _reg, _ in _FONT_CANDIDATES:
    if os.path.exists(_path):
        if _FONT_PATH is None:
            _FONT_PATH = _path
        try:
            pdfmetrics.registerFont(TTFont(_reg, _path))
            if _reg_normal is None:
                _reg_normal = _reg
            elif _reg_bold is None:
                _reg_bold = _reg
        except Exception:
            pass

if _reg_normal:
    _FONT_NORMAL = _reg_normal
    _FONT_BOLD   = _reg_bold or _reg_normal
    _using_ttf   = True
    pdfmetrics.registerFontFamily(_reg_normal,
                                  normal=_reg_normal,
                                  bold=_FONT_BOLD,
                                  italic=_reg_normal,
                                  boldItalic=_FONT_BOLD)
else:
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))


# カラーパレット
C_NAVY   = colors.HexColor("#1a3a5c")
C_BLUE   = colors.HexColor("#2563a8")
C_LBLUE  = colors.HexColor("#dbeafe")
C_ORANGE = colors.HexColor("#f57c00")
C_GREEN  = colors.HexColor("#15803d")
C_RED    = colors.HexColor("#dc2626")
C_LGRAY  = colors.HexColor("#f5f5f5")
C_LGREEN = colors.HexColor("#dcfce7")
C_WHITE  = colors.white
C_BLACK  = colors.black
C_GOLD   = colors.HexColor("#b7791f")

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
TODAY        = datetime.today().strftime("%Y-%m-%d")
RUN_DATETIME = datetime.today().strftime("%Y-%m-%d_%H%M")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "results", RUN_DATETIME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 市場別パラメータ
# ─────────────────────────────────────────
MARKET_PARAMS = {
    "JP": {
        "aum_min":        10_000_000_000,   # 100億円
        "aum_large":      500_000_000_000,  # 5,000億円
        "ret1y_min":      0.10,             # 1年リターン 10%以上
        "ret3y_min":      0.08,             # 3年年率リターン 8%以上
        "expense_max":    0.010,            # 信託報酬 1.0%以下
        "expense_low":    0.002,            # 超低コスト 0.2%以下
        "div_yield_min":  0.025,            # 分配金利回り 2.5%以上
        "sharpe_min":     0.8,              # シャープレシオ 0.8以上
        "dd_max":        -0.30,             # 最大ドローダウン -30%以内
        "vol_max":        0.25,             # 年率ボラティリティ 25%以内
        "currency":       "円",
    },
    "US": {
        "aum_min":        100_000_000,      # 1億USD
        "aum_large":      5_000_000_000,    # 50億USD
        "ret1y_min":      0.10,
        "ret3y_min":      0.08,
        "expense_max":    0.010,
        "expense_low":    0.002,
        "div_yield_min":  0.020,            # 分配金利回り 2.0%以上
        "sharpe_min":     0.8,
        "dd_max":        -0.30,
        "vol_max":        0.25,
        "currency":       "USD",
    },
}

# ─────────────────────────────────────────
# 信託ユニバース（ETF代理）
# ─────────────────────────────────────────

# 国内ETF（東証上場）
TRUST_UNIVERSE_JP = [
    # ── 国内株式 ──
    "1306.T",   # NEXT FUNDS TOPIX連動型
    "1321.T",   # 日経225連動型投信
    "1330.T",   # 上場インデックスファンド日経225
    "1348.T",   # MAXIS トピックス上場投信
    "1478.T",   # iShares MSCIジャパン高配当利回り
    "1577.T",   # 野村日本株高配当70連動型
    "1615.T",   # NEXT FUNDS 東証銀行業株価指数連動型
    "1698.T",   # 上場インデックスファンド日本高配当
    "2516.T",   # 東証マザーズETF
    "2563.T",   # iShares S&P500 ETF（東証）
    "2558.T",   # MAXIS米国株式(S&P500)上場投信
    # ── 国内REIT ──
    "1343.T",   # NEXT FUNDS 東証REIT指数連動型
    "1345.T",   # 上場インデックスファンド国内REIT
    "2552.T",   # 上場インデックスファンドJリート隔月分配型
    # ── 国内債券 ──
    "1320.T",   # ダイワ上場投信-日経225
    "2510.T",   # NEXT FUNDS 国内債券・NOMURA-BPI総合連動型
    # ── コモディティ ──
    "1540.T",   # 純金上場信託（現物国内保管型）
    "1541.T",   # 純プラチナ上場信託
    "1699.T",   # NEXT FUNDS 原油指数連動型
]

# 海外ETF（米国上場）
TRUST_UNIVERSE_US = [
    # ── 全世界株式 ──
    "VT",       # Vanguard Total World Stock ETF
    "ACWI",     # iShares MSCI ACWI ETF
    # ── 米国株式 ──
    "SPY",      # SPDR S&P 500 ETF
    "VOO",      # Vanguard S&P 500 ETF
    "VTI",      # Vanguard Total Stock Market ETF
    "QQQ",      # Invesco QQQ (NASDAQ-100)
    "IVV",      # iShares Core S&P 500 ETF
    # ── 先進国株式 ──
    "VEA",      # Vanguard Developed Markets ETF
    "EFA",      # iShares MSCI EAFE ETF
    "EWJ",      # iShares MSCI Japan ETF
    "DXJ",      # WisdomTree Japan Hedged Equity
    # ── 新興国株式 ──
    "VWO",      # Vanguard Emerging Markets ETF
    "EEM",      # iShares MSCI Emerging Markets ETF
    "IEMG",     # iShares Core MSCI Emerging Markets
    # ── テーマ・セクター ──
    "QQQ",      # テクノロジー/NASDAQ-100
    "SOXX",     # iShares Semiconductor ETF
    "XLK",      # Technology Select Sector SPDR
    "ARKK",     # ARK Innovation ETF
    "BOTZ",     # Global X Robotics & AI ETF
    # ── 高配当 ──
    "VYM",      # Vanguard High Dividend Yield ETF
    "HDV",      # iShares Core High Dividend ETF
    "SCHD",     # Schwab US Dividend Equity ETF
    "DVY",      # iShares Select Dividend ETF
    # ── バランス・債券 ──
    "AGG",      # iShares Core U.S. Aggregate Bond ETF
    "BND",      # Vanguard Total Bond Market ETF
    "TLT",      # iShares 20+ Year Treasury Bond ETF
    "LQD",      # iShares iBoxx $ Investment Grade Corp Bond
    # ── REIT ──
    "VNQ",      # Vanguard Real Estate ETF
    "IYR",      # iShares U.S. Real Estate ETF
    # ── コモディティ ──
    "GLD",      # SPDR Gold Shares
    "IAU",      # iShares Gold Trust
    "PDBC",     # Invesco Optimum Yield Diversified Commodity
    # ── バランス型 ──
    "AOM",      # iShares Core Moderate Allocation ETF
    "AOA",      # iShares Core Aggressive Allocation ETF
]

TRUST_UNIVERSE = list(dict.fromkeys(TRUST_UNIVERSE_JP + TRUST_UNIVERSE_US))
MARKET_MAP     = {t: "JP" if t.endswith(".T") else "US" for t in TRUST_UNIVERSE}

_n_jp = sum(1 for t in TRUST_UNIVERSE if t.endswith(".T"))
_n_us = len(TRUST_UNIVERSE) - _n_jp
print(f"対象ファンド数: {len(TRUST_UNIVERSE)}  (国内: {_n_jp}本 / 海外: {_n_us}本)")

# ─────────────────────────────────────────
# カテゴリマッピング
# ─────────────────────────────────────────
CATEGORY_MAP = {
    # 国内ETF
    "1306.T": "国内株式（TOPIX）",
    "1321.T": "国内株式（日経225）",
    "1330.T": "国内株式（日経225）",
    "1348.T": "国内株式（TOPIX）",
    "1478.T": "国内株式（高配当）",
    "1577.T": "国内株式（高配当）",
    "1615.T": "国内株式（銀行セクター）",
    "1698.T": "国内株式（高配当）",
    "2516.T": "国内株式（グロース）",
    "2563.T": "外国株式（S&P500）",
    "2558.T": "外国株式（S&P500）",
    "1343.T": "国内REIT",
    "1345.T": "国内REIT",
    "2552.T": "国内REIT",
    "1320.T": "国内株式（日経225）",
    "2510.T": "国内債券",
    "1540.T": "コモディティ（金）",
    "1541.T": "コモディティ（プラチナ）",
    "1699.T": "コモディティ（原油）",
    # 海外ETF
    "VT":   "全世界株式",
    "ACWI": "全世界株式",
    "SPY":  "外国株式（S&P500）",
    "VOO":  "外国株式（S&P500）",
    "VTI":  "外国株式（全米）",
    "QQQ":  "外国株式（NASDAQ100）",
    "IVV":  "外国株式（S&P500）",
    "VEA":  "外国株式（先進国）",
    "EFA":  "外国株式（先進国）",
    "EWJ":  "外国株式（日本）",
    "DXJ":  "外国株式（日本・円ヘッジ）",
    "VWO":  "外国株式（新興国）",
    "EEM":  "外国株式（新興国）",
    "IEMG": "外国株式（新興国）",
    "SOXX": "テーマ型（半導体）",
    "XLK":  "テーマ型（テクノロジー）",
    "ARKK": "テーマ型（イノベーション）",
    "BOTZ": "テーマ型（AI・ロボット）",
    "VYM":  "高配当株式",
    "HDV":  "高配当株式",
    "SCHD": "高配当株式",
    "DVY":  "高配当株式",
    "AGG":  "外国債券（総合）",
    "BND":  "外国債券（総合）",
    "TLT":  "外国債券（長期国債）",
    "LQD":  "外国債券（社債）",
    "VNQ":  "海外REIT",
    "IYR":  "海外REIT",
    "GLD":  "コモディティ（金）",
    "IAU":  "コモディティ（金）",
    "PDBC": "コモディティ（総合）",
    "AOM":  "バランス型（安定）",
    "AOA":  "バランス型（積極）",
}

TECH_TICKERS   = {"QQQ", "SOXX", "XLK", "ARKK", "BOTZ", "2516.T"}
REIT_TICKERS   = {"VNQ", "IYR", "1343.T", "1345.T", "2552.T"}
BOND_TICKERS   = {"AGG", "BND", "TLT", "LQD", "2510.T"}
CMDTY_TICKERS  = {"GLD", "IAU", "PDBC", "1540.T", "1541.T", "1699.T"}
DIV_TICKERS    = {"VYM", "HDV", "SCHD", "DVY", "1478.T", "1577.T", "1698.T"}
WORLD_TICKERS  = {"VT", "ACWI"}
SP500_TICKERS  = {"SPY", "VOO", "IVV", "2563.T", "2558.T"}
JP_EQ_TICKERS  = {"1306.T", "1321.T", "1330.T", "1348.T", "1320.T"}

# ─────────────────────────────────────────
# データ取得
# ─────────────────────────────────────────
def fetch_fund_data(tickers: list, period: str = "5y") -> dict:
    print(f"\nファンドデータ取得中... ({len(tickers)}本)")
    data = {}
    for i in range(0, len(tickers), 10):
        batch = tickers[i:i+10]
        print(f"  {i+1}〜{min(i+10, len(tickers))}本目...")
        for ticker in batch:
            try:
                df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                if df is not None and len(df) >= 60:
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                    data[ticker] = df
            except Exception:
                pass
        time.sleep(0.3)
    print(f"取得成功: {len(data)}本")
    return data

def fetch_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

# ─────────────────────────────────────────
# テクニカル指標計算
# ─────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    close  = df["Close"]
    volume = df.get("Volume", pd.Series(0, index=df.index))

    df["SMA20"]  = ta.sma(close, 20)
    df["SMA50"]  = ta.sma(close, 50)
    df["SMA120"] = ta.sma(close, 120)
    df["SMA200"] = ta.sma(close, 200)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["MACD"]      = macd.get("MACD_12_26_9",  pd.Series(dtype=float))
        df["MACD_sig"]  = macd.get("MACDs_12_26_9", pd.Series(dtype=float))
        df["MACD_hist"] = macd.get("MACDh_12_26_9", pd.Series(dtype=float))

    df["RSI14"] = ta.rsi(close, 14)

    bb = ta.bbands(close, length=20, std=2)
    if bb is not None and not bb.empty:
        df["BB_upper"] = bb.get("BBU_20_2.0", pd.Series(dtype=float))
        df["BB_mid"]   = bb.get("BBM_20_2.0", pd.Series(dtype=float))
        df["BB_lower"] = bb.get("BBL_20_2.0", pd.Series(dtype=float))

    adx_in = ta.adx(df["High"], df["Low"], close, length=14)
    if adx_in is not None and not adx_in.empty:
        df["ADX"]    = adx_in.get("ADX_14", pd.Series(dtype=float))
        df["DI_pos"] = adx_in.get("DMP_14", pd.Series(dtype=float))
        df["DI_neg"] = adx_in.get("DMN_14", pd.Series(dtype=float))

    df["VOL_MA20"] = volume.rolling(20).mean()
    df["HIGH_52W"] = close.rolling(252).max().shift(1)
    df["LOW_52W"]  = close.rolling(252).min()

    # リターン計算
    df["RET_20D"]   = close.pct_change(20)  * 100
    df["RET_60D"]   = close.pct_change(60)  * 100
    df["RET_120D"]  = close.pct_change(120) * 100
    df["RET_252D"]  = close.pct_change(252) * 100   # 1年リターン
    df["RET_756D"]  = close.pct_change(756) * 100   # 3年リターン（約）

    # SMA200乖離率
    df["DEV_SMA200"] = (close - df["SMA200"]) / df["SMA200"] * 100

    # ローリングボラティリティ（年率）
    df["VOL_1Y"] = close.pct_change().rolling(252).std() * np.sqrt(252)

    # 最大ドローダウン（1年・3年）
    roll_max_1y = close.rolling(252).max()
    df["DD_1Y"] = (close - roll_max_1y) / roll_max_1y

    roll_max_3y = close.rolling(756).max()
    df["DD_3Y"] = (close - roll_max_3y) / roll_max_3y

    return df

def get_latest(df):
    return df.iloc[-1]

# ─────────────────────────────────────────
# ファンド固有指標の計算
# ─────────────────────────────────────────
def calc_fund_metrics(df: pd.DataFrame, info: dict) -> dict:
    """投資信託/ETF固有指標をdictで返す"""
    r = get_latest(df)
    close = df["Close"]

    # リターン
    ret1y  = float(r.RET_252D)  if not pd.isna(r.RET_252D)  else None
    ret3y  = float(r.RET_756D)  if not pd.isna(r.RET_756D)  else None
    ret3m  = float(r.RET_60D)   if not pd.isna(r.RET_60D)   else None
    ret1m  = float(r.RET_20D)   if not pd.isna(r.RET_20D)   else None

    # 3年年率換算
    ret3y_ann = None
    if ret3y is not None:
        try:
            ret3y_ann = ((1 + ret3y / 100) ** (1/3) - 1) * 100
        except Exception:
            pass

    # ボラティリティ（年率）
    vol1y = float(r.VOL_1Y) if not pd.isna(r.VOL_1Y) else None

    # シャープレシオ（リスクフリーレート 0.5% 想定）
    sharpe = None
    if ret3y_ann is not None and vol1y is not None and vol1y > 0:
        sharpe = (ret3y_ann - 0.5) / (vol1y * 100)

    # 最大ドローダウン
    dd1y = float(r.DD_1Y) if not pd.isna(r.DD_1Y) else None
    dd3y = float(r.DD_3Y) if not pd.isna(r.DD_3Y) else None

    # AUM
    aum = info.get("totalAssets") or 0

    # 信託報酬
    expense = info.get("annualReportExpenseRatio") or info.get("expenseRatio") or None

    # 分配金利回り
    div_yield = info.get("yield") or info.get("dividendYield") or 0

    # ベータ
    beta = info.get("beta3Year") or info.get("beta") or None

    # カテゴリ
    category = (CATEGORY_MAP.get(
        # tickerはinfo内にないのでinfo["symbol"]で取得
        info.get("symbol", ""),
        info.get("category") or info.get("fundFamily") or "N/A"
    ))

    return {
        "ret1y":      ret1y,
        "ret3y_ann":  ret3y_ann,
        "ret3m":      ret3m,
        "ret1m":      ret1m,
        "vol1y":      vol1y,
        "sharpe":     sharpe,
        "dd1y":       dd1y,
        "dd3y":       dd3y,
        "aum":        aum,
        "expense":    expense,
        "div_yield":  div_yield,
        "beta":       beta,
        "category":   category,
    }

# ─────────────────────────────────────────
# 共通ベースフィルター（市場別）
# ─────────────────────────────────────────
def passes_base_filter(df: pd.DataFrame, info: dict, market: str, fm: dict) -> bool:
    p = MARKET_PARAMS[market]
    r = get_latest(df)

    # AUM フィルター（情報がある場合のみ）
    if fm["aum"] and fm["aum"] > 0 and fm["aum"] < p["aum_min"]:
        return False

    # 極端な高コスト除外
    if fm["expense"] and fm["expense"] > 0.05:  # 5%超は除外
        return False

    # データ不足除外
    if len(df) < 60:
        return False

    return True

# ─────────────────────────────────────────
# スクリーニング関数（全26戦略）
# ─────────────────────────────────────────

def screen_A1(df, info, market, fm) -> bool:
    """A-1: 1年高リターンモメンタム"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if fm["ret1y"] is None or fm["ret1y"] < p["ret1y_min"] * 100: return False
        if fm["ret3m"] is None or fm["ret3m"] < 3: return False
        if fm["expense"] and fm["expense"] > p["expense_max"]: return False
        if fm["dd1y"] is not None and fm["dd1y"] < -0.40: return False
        if fm["aum"] and fm["aum"] > 0 and fm["aum"] < p["aum_min"]: return False
        return True
    except Exception: return False

def screen_A2(df, info, market, fm) -> bool:
    """A-2: 長期安定成長トレンド"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if fm["ret3y_ann"] is None or fm["ret3y_ann"] < p["ret3y_min"] * 100: return False
        if fm["ret1y"] is None or fm["ret1y"] < 0: return False
        if fm["vol1y"] is not None and fm["vol1y"] > p["vol_max"]: return False
        if fm["aum"] and fm["aum"] > 0 and fm["aum"] < p["aum_min"]: return False
        return True
    except Exception: return False

def screen_A3(df, info, market, fm) -> bool:
    """A-3: 相対モメンタム（上昇トレンド継続）"""
    try:
        r = get_latest(df)
        if fm["ret1y"] is None or fm["ret1y"] < 5: return False
        if fm["ret3m"] is None or fm["ret3m"] < 1: return False
        if fm["ret1m"] is None or fm["ret1m"] < 0: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 75): return False
        return True
    except Exception: return False

def screen_B1(df, info, market, fm) -> bool:
    """B-1: 低コスト × 高リターン"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if fm["expense"] is None: return False
        if fm["expense"] > p["expense_low"]: return False
        if fm["ret1y"] is None or fm["ret1y"] < p["ret1y_min"] * 100: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if fm["aum"] and fm["aum"] > 0 and fm["aum"] < p["aum_min"] * 10: return False
        return True
    except Exception: return False

def screen_B2(df, info, market, fm) -> bool:
    """B-2: シャープレシオ優位"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if fm["sharpe"] is None or fm["sharpe"] < p["sharpe_min"]: return False
        if fm["dd3y"] is not None and fm["dd3y"] < -0.30: return False
        if fm["ret1y"] is None or fm["ret1y"] < 5: return False
        if pd.isna(r.RSI14) or r.RSI14 < 45: return False
        if pd.isna(r.SMA120) or r.Close <= r.SMA120: return False
        return True
    except Exception: return False

def screen_B3(df, info, market, fm) -> bool:
    """B-3: β低・安定リターン（保守的バランス型）"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if fm["beta"] is None or fm["beta"] > 0.7: return False
        if fm["vol1y"] is not None and fm["vol1y"] > 0.15: return False
        if fm["ret1y"] is None or fm["ret1y"] < 3: return False
        if fm["ret3y_ann"] is None or fm["ret3y_ann"] < 5: return False
        return True
    except Exception: return False

def screen_C1(df, info, market, fm) -> bool:
    """C-1: 最大ドローダウン小 × リターン安定"""
    try:
        r = get_latest(df)
        if fm["dd1y"] is None or fm["dd1y"] < -0.20: return False
        if fm["ret1y"] is None or fm["ret1y"] < 5: return False
        if fm["vol1y"] is not None and fm["vol1y"] > 0.18: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        return True
    except Exception: return False

def screen_C2(df, info, market, fm) -> bool:
    """C-2: 分散効果優位（低β・安定リターン）"""
    try:
        r = get_latest(df)
        if fm["beta"] is None or fm["beta"] > 0.6: return False
        if fm["ret1y"] is None or fm["ret1y"] < 3: return False
        if pd.isna(r.RSI14) or not (40 <= r.RSI14 <= 65): return False
        if pd.isna(r.SMA50) or r.Close <= r.SMA50: return False
        return True
    except Exception: return False

def screen_D1(df, info, market, fm) -> bool:
    """D-1: AUM急成長（規模大・高リターン）"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if fm["aum"] is None or fm["aum"] < p["aum_min"] * 10: return False
        if fm["ret1y"] is None or fm["ret1y"] < p["ret1y_min"] * 100: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        return True
    except Exception: return False

def screen_D2(df, info, market, fm) -> bool:
    """D-2: 長期資金安定（超大型・安定リターン）"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if fm["aum"] is None or fm["aum"] < p["aum_large"]: return False
        if fm["ret1y"] is None or fm["ret1y"] < 5: return False
        if fm["ret3y_ann"] is None or fm["ret3y_ann"] < p["ret3y_min"] * 100: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 70): return False
        return True
    except Exception: return False

def screen_E1(df, info, market, fm) -> bool:
    """E-1: 強気ダイバージェンス + ファンダ良好"""
    try:
        r = get_latest(df)
        if fm["ret3y_ann"] is None or fm["ret3y_ann"] < 5: return False
        if fm["ret1y"] is None or fm["ret1y"] < -10: return False
        # RSI回復確認
        if pd.isna(r.RSI14) or r.RSI14 > 55: return False
        # MACDダイバージェンス簡易確認
        recent = df.iloc[-20:]
        if "MACD_hist" not in recent.columns: return False
        macd_vals = recent["MACD_hist"].dropna()
        close_vals = recent["Close"].dropna()
        if len(macd_vals) < 5 or len(close_vals) < 5: return False
        # 価格下落 + MACDヒスト上昇（強気ダイバージェンス）
        if close_vals.iloc[-1] < close_vals.iloc[0] and macd_vals.iloc[-1] > macd_vals.iloc[0]:
            return True
        return False
    except Exception: return False

def screen_E2(df, info, market, fm) -> bool:
    """E-2: 押し目買い（200日SMA付近サポート）"""
    try:
        r = get_latest(df)
        if pd.isna(r.SMA200): return False
        dev = (r.Close - r.SMA200) / r.SMA200
        if not (-0.08 <= dev <= 0.08): return False
        if pd.isna(r.RSI14) or not (35 <= r.RSI14 <= 55): return False
        if fm["ret3y_ann"] is None or fm["ret3y_ann"] < 8: return False
        if fm["dd3y"] is not None and fm["dd3y"] < -0.35: return False
        return True
    except Exception: return False

def screen_F1(df, info, market, fm) -> bool:
    """F-1: AI・テクノロジーテーマ リーダー"""
    try:
        # ticker取得
        sym = info.get("symbol", "")
        is_tech = (sym in TECH_TICKERS or
                   "テクノロジー" in (fm.get("category") or "") or
                   "半導体" in (fm.get("category") or "") or
                   "AI" in (fm.get("category") or "") or
                   "イノベーション" in (fm.get("category") or ""))
        if not is_tech: return False
        r = get_latest(df)
        if fm["ret1y"] is None or fm["ret1y"] < 20: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 78): return False
        return True
    except Exception: return False

def screen_F2(df, info, market, fm) -> bool:
    """F-2: 高配当・インカム型"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        sym = info.get("symbol", "")
        is_div = (sym in DIV_TICKERS or
                  "高配当" in (fm.get("category") or "") or
                  "配当" in (fm.get("category") or ""))
        if not is_div and (fm["div_yield"] is None or fm["div_yield"] < p["div_yield_min"]):
            return False
        if fm["div_yield"] and fm["div_yield"] < p["div_yield_min"]: return False
        if fm["ret1y"] is None or fm["ret1y"] < 0: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RSI14) or not (40 <= r.RSI14 <= 65): return False
        return True
    except Exception: return False

def screen_F3(df, info, market, fm) -> bool:
    """F-3: ESG・サステナブル投資"""
    try:
        r = get_latest(df)
        esg_score = info.get("esgScores") or info.get("sustainabilityScore")
        # ESGスコアがない場合はカテゴリ名で代用
        cat = fm.get("category") or ""
        is_esg = bool(esg_score) or "ESG" in cat or "サステナ" in cat
        if not is_esg: return False
        if fm["ret1y"] is None or fm["ret1y"] < 8: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 70): return False
        return True
    except Exception: return False

def screen_G1(df, info, market, fm) -> bool:
    """G-1: 信託総合最優秀（全軸高水準）"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if fm["expense"] is None or fm["expense"] > 0.005: return False
        if fm["ret1y"] is None or fm["ret1y"] < 12: return False
        if fm["ret3y_ann"] is None or fm["ret3y_ann"] < p["ret3y_min"] * 100: return False
        if fm["sharpe"] is not None and fm["sharpe"] < p["sharpe_min"]: return False
        if fm["aum"] and fm["aum"] > 0 and fm["aum"] < p["aum_min"] * 10: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 75): return False
        return True
    except Exception: return False

def screen_G2(df, info, market, fm) -> bool:
    """G-2: 長期積立ベスト（iDeCo・NISA最適化）"""
    try:
        p = MARKET_PARAMS[market]
        r = get_latest(df)
        if fm["expense"] is None or fm["expense"] > 0.003: return False
        if fm["ret1y"] is None or fm["ret1y"] < 8: return False
        if fm["ret3y_ann"] is None or fm["ret3y_ann"] < p["ret3y_min"] * 100: return False
        if fm["aum"] and fm["aum"] > 0 and fm["aum"] < p["aum_min"] * 5: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        return True
    except Exception: return False

def screen_H1(df, info, market, fm) -> bool:
    """H-1: 日本株集中型リーダー"""
    try:
        sym = info.get("symbol", "")
        is_jp = (sym in JP_EQ_TICKERS or
                 "国内株式（TOPIX）" in (fm.get("category") or "") or
                 "国内株式（日経" in (fm.get("category") or ""))
        if not is_jp: return False
        r = get_latest(df)
        if fm["ret1y"] is None or fm["ret1y"] < 15: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RSI14) or not (50 <= r.RSI14 <= 75): return False
        return True
    except Exception: return False

def screen_H2(df, info, market, fm) -> bool:
    """H-2: 全世界分散型リーダー"""
    try:
        sym = info.get("symbol", "")
        is_world = (sym in WORLD_TICKERS or
                    "全世界株式" in (fm.get("category") or "") or
                    sym in SP500_TICKERS or
                    "S&P500" in (fm.get("category") or ""))
        if not is_world: return False
        r = get_latest(df)
        if fm["ret1y"] is None or fm["ret1y"] < 12: return False
        if fm["ret3y_ann"] is None or fm["ret3y_ann"] < 10: return False
        if fm["expense"] is not None and fm["expense"] > 0.002: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RSI14) or not (48 <= r.RSI14 <= 72): return False
        return True
    except Exception: return False

def screen_H3(df, info, market, fm) -> bool:
    """H-3: 新興国成長型"""
    try:
        sym = info.get("symbol", "")
        is_em = (sym in {"VWO", "EEM", "IEMG"} or
                 "新興国" in (fm.get("category") or ""))
        if not is_em: return False
        r = get_latest(df)
        if fm["ret1y"] is None or fm["ret1y"] < 10: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 70): return False
        return True
    except Exception: return False

def screen_I1(df, info, market, fm) -> bool:
    """I-1: REIT・不動産型"""
    try:
        sym = info.get("symbol", "")
        is_reit = (sym in REIT_TICKERS or
                   "REIT" in (fm.get("category") or "") or
                   "不動産" in (fm.get("category") or ""))
        if not is_reit: return False
        r = get_latest(df)
        if fm["div_yield"] is not None and fm["div_yield"] < 0.025: return False
        if fm["ret1y"] is None or fm["ret1y"] < 3: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RSI14) or not (40 <= r.RSI14 <= 65): return False
        return True
    except Exception: return False

def screen_I2(df, info, market, fm) -> bool:
    """I-2: コモディティ・実物資産型"""
    try:
        sym = info.get("symbol", "")
        is_cmdty = (sym in CMDTY_TICKERS or
                    "コモディティ" in (fm.get("category") or "") or
                    "金" in (fm.get("category") or ""))
        if not is_cmdty: return False
        r = get_latest(df)
        if fm["ret1y"] is None or fm["ret1y"] < 5: return False
        if pd.isna(r.SMA200) or r.Close <= r.SMA200: return False
        if pd.isna(r.RSI14) or not (45 <= r.RSI14 <= 75): return False
        return True
    except Exception: return False

def screen_I3(df, info, market, fm) -> bool:
    """I-3: 債券・固定収益型（防御的配分）"""
    try:
        sym = info.get("symbol", "")
        is_bond = (sym in BOND_TICKERS or
                   "債券" in (fm.get("category") or ""))
        if not is_bond: return False
        r = get_latest(df)
        if fm["ret1y"] is None or fm["ret1y"] < 0: return False
        if fm["vol1y"] is not None and fm["vol1y"] > 0.10: return False
        if pd.isna(r.SMA120) or r.Close <= r.SMA120: return False
        return True
    except Exception: return False

# ─────────────────────────────────────────
# 戦略一覧
# ─────────────────────────────────────────
STRATEGIES = {
    "A-1_高リターンモメンタム":   screen_A1,
    "A-2_長期安定成長":           screen_A2,
    "A-3_相対モメンタム上昇":     screen_A3,
    "B-1_低コスト高リターン":     screen_B1,
    "B-2_シャープレシオ優位":     screen_B2,
    "B-3_低β安定リターン":       screen_B3,
    "C-1_低DD安定型":             screen_C1,
    "C-2_分散効果優位":           screen_C2,
    "D-1_大型高リターン":         screen_D1,
    "D-2_超大型安定資金":         screen_D2,
    "E-1_強気ダイバージェンス":   screen_E1,
    "E-2_押し目サポート":         screen_E2,
    "F-1_AIテクノロジー":         screen_F1,
    "F-2_高配当インカム":         screen_F2,
    "F-3_ESGサステナブル":        screen_F3,
    "G-1_信託総合最優秀":         screen_G1,
    "G-2_長期積立ベスト":         screen_G2,
    "H-1_日本株リーダー":         screen_H1,
    "H-2_全世界分散リーダー":     screen_H2,
    "H-3_新興国成長型":           screen_H3,
    "I-1_REIT不動産型":           screen_I1,
    "I-2_コモディティ実物資産":   screen_I2,
    "I-3_債券固定収益型":         screen_I3,
}

# ─────────────────────────────────────────
# 長期投資重視スコアリング
# ─────────────────────────────────────────
GROWTH_STRATEGIES = {
    "A-1", "A-2", "A-3",    # モメンタム
    "F-1",                   # テーマ
    "G-1", "G-2",            # 総合
    "H-1", "H-2",            # 地域リーダー
    "D-1", "D-2",            # AUM規模
}

STABLE_STRATEGIES = {
    "B-2", "B-3",            # 安定系
    "C-1", "C-2",            # ドローダウン耐性
}

def calc_trust_score(r: dict, fm: dict) -> float:
    """長期投資信託評価スコア（高いほど推奨度高）"""
    score = 0.0
    for s in r["マッチ戦略"].split(" | "):
        key = s[:3]
        if key in GROWTH_STRATEGIES:
            score += 2.0
        elif key in STABLE_STRATEGIES:
            score += 1.5
        else:
            score += 1.0

    # リターン加点
    ret1y = fm.get("ret1y") or 0
    if   ret1y >= 30: score += 5
    elif ret1y >= 20: score += 3
    elif ret1y >= 12: score += 2
    elif ret1y >= 8:  score += 1

    # コスト加点（低コストほど高スコア）
    exp = fm.get("expense")
    if exp is not None:
        if   exp <= 0.001: score += 4
        elif exp <= 0.002: score += 3
        elif exp <= 0.005: score += 1

    # AUM加点
    aum = fm.get("aum") or 0
    if   aum >= 1_000_000_000_000:  score += 3   # 1兆円 or 100億USD
    elif aum >= 500_000_000_000:    score += 2   # 5,000億
    elif aum >= 100_000_000_000:    score += 1   # 1,000億

    # シャープレシオ加点
    sharpe = fm.get("sharpe") or 0
    if   sharpe >= 1.5: score += 3
    elif sharpe >= 1.0: score += 2
    elif sharpe >= 0.8: score += 1

    # RSIトレンドゾーン加点
    rsi = r.get("RSI14") or 0
    if 50 <= rsi <= 72: score += 1

    return round(score, 1)

def is_high_risk(df: pd.DataFrame, fm: dict) -> bool:
    """ハイリスクファンド判定（True = 除外対象）"""
    try:
        latest = df.iloc[-1]
        rsi = float(latest.RSI14) if not pd.isna(latest.RSI14) else 0
        if rsi > 82: return True

        if fm["dd1y"] is not None and fm["dd1y"] < -0.45: return True

        if fm["vol1y"] is not None and fm["vol1y"] > 0.40: return True

        if fm["expense"] is not None and fm["expense"] > 0.025: return True
    except Exception:
        pass
    return False

# ─────────────────────────────────────────
# 推奨理由・リスク・シナリオ生成
# ─────────────────────────────────────────
def generate_buy_reasons(r: dict, df: pd.DataFrame, info: dict, market: str, fm: dict):
    """(推奨理由リスト, リスク要因リスト, シナリオ文字列) を返す"""
    reasons, risks = [], []
    p = MARKET_PARAMS[market]

    ret1y   = fm.get("ret1y") or 0
    ret3y   = fm.get("ret3y_ann") or 0
    sharpe  = fm.get("sharpe")
    expense = fm.get("expense")
    aum     = fm.get("aum") or 0
    dd1y    = fm.get("dd1y") or 0
    div_yield = fm.get("div_yield") or 0
    rsi     = r.get("RSI14") or 0

    # ── 推奨理由 ──
    if ret1y >= 20:
        reasons.append(f"1年リターン {ret1y:.1f}%（高リターン・モメンタム継続中）")
    elif ret1y >= 10:
        reasons.append(f"1年リターン {ret1y:.1f}%（安定した高リターン）")

    if ret3y >= 12:
        reasons.append(f"3年年率リターン {ret3y:.1f}%（長期成長力が実証済み）")
    elif ret3y >= 8:
        reasons.append(f"3年年率リターン {ret3y:.1f}%（長期安定パフォーマンス）")

    if expense is not None:
        if expense <= 0.001:
            reasons.append(f"信託報酬 {expense*100:.3f}%（業界最低水準コスト・長期保有に最適）")
        elif expense <= 0.003:
            reasons.append(f"信託報酬 {expense*100:.3f}%（超低コストで長期リターンを最大化）")

    if sharpe is not None and sharpe >= 1.0:
        reasons.append(f"シャープレシオ {sharpe:.2f}（優れたリスク調整済リターン）")

    if aum >= 1_000_000_000_000:
        currency = p["currency"]
        if market == "JP":
            reasons.append(f"純資産 {aum/1e12:.1f}兆円（超大型・高い流動性と安定性）")
        else:
            reasons.append(f"AUM {aum/1e9:.0f}億USD（超大型・高い流動性と安定性）")

    if dd1y >= -0.10:
        reasons.append(f"最大ドローダウン {dd1y*100:.1f}%（価格安定性が高い）")

    if 52 <= rsi <= 68:
        reasons.append(f"RSI {rsi:.0f}（上昇トレンドの理想ゾーン・過熱感なし）")

    try:
        latest = df.iloc[-1]
        if not pd.isna(latest.SMA200) and latest.Close > latest.SMA200:
            dev = (latest.Close - latest.SMA200) / latest.SMA200 * 100
            reasons.append(f"200日SMAを{dev:.1f}%上回る（長期上昇トレンド確認）")
    except Exception:
        pass

    matched = r["マッチ戦略"].split(" | ")
    _key = {
        "G-1": "全軸高水準クリア（コスト・リターン・リスク・AUM 総合最優秀）",
        "G-2": "長期積立 NISA/iDeCo 最適化条件クリア",
        "H-2": "全世界分散リーダー（低コスト・高リターン・大型）",
        "B-1": "低コスト×高リターン（コスト効率最高クラス）",
        "A-1": "1年高リターンモメンタム継続中",
        "F-1": "AI・テクノロジーテーマ セクターリーダー",
        "F-2": "高配当インカム型（安定分配金収入）",
        "I-2": "コモディティ・インフレヘッジ（金など実物資産）",
    }
    for s in matched:
        if s[:3] in _key:
            reasons.append(_key[s[:3]])
            break

    # ── リスク要因 ──
    if rsi > 72:
        risks.append(f"RSI {rsi:.0f}（やや過熱気味・短期調整リスクあり）")
    if dd1y < -0.25:
        risks.append(f"直近ドローダウン {dd1y*100:.1f}%（価格変動が大きい局面あり）")
    if fm.get("vol1y") and fm["vol1y"] > 0.20:
        risks.append(f"年率ボラティリティ {fm['vol1y']*100:.0f}%（価格変動が大きい）")
    if expense and expense > 0.005:
        risks.append(f"信託報酬 {expense*100:.2f}%（長期保有でコスト負担に注意）")
    if not risks:
        risks.append("特段の高リスク要因なし（ベースフィルター・ハイリスク除外済）")

    # ── シナリオ ──
    try:
        cl = float(df.iloc[-1].Close)
        h52 = float(df["Close"].rolling(252).max().iloc[-1])
        tgt = max(h52 * 1.08, cl * 1.15)
        stop = cl * 0.90
        currency = "円" if market == "JP" else "USD"
        scenario = (f"目標: {tgt:.2f}{currency} ({(tgt/cl-1)*100:+.0f}%) /"
                    f" 撤退ライン: {stop:.2f}{currency} (-10%)")
    except Exception:
        scenario = "目標・撤退ラインはご自身でご判断ください"

    return reasons[:5], risks[:3], scenario

# ─────────────────────────────────────────
# PDF ヘルパー
# ─────────────────────────────────────────
_trans_cache: dict = {}

def _translate(text: str, max_len: int = 400) -> str:
    if not text or not text.strip():
        return ""
    text = text[:max_len]
    if text in _trans_cache:
        return _trans_cache[text]
    try:
        result = GoogleTranslator(source="auto", target="ja").translate(text)
        _trans_cache[text] = result or text
    except Exception:
        _trans_cache[text] = text
    return _trans_cache[text]

_CIRCLED = ["①","②","③","④","⑤","⑥","⑦","⑧"]

def _safe(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

_cs_hdr   = ParagraphStyle("_csh",  fontName=_FONT_BOLD,   fontSize=8.5,
                             leading=12, textColor=C_WHITE,  wordWrap="CJK")
_cs_body  = ParagraphStyle("_csb",  fontName=_FONT_NORMAL, fontSize=8.5,
                             leading=12, textColor=C_BLACK,  wordWrap="CJK")
_cs_lblue = ParagraphStyle("_cslb", fontName=_FONT_BOLD,   fontSize=8.5,
                             leading=12, textColor=C_NAVY,   wordWrap="CJK")

def _style(name, **kw):
    base = dict(fontName=_FONT_NORMAL, fontSize=10, leading=16,
                textColor=C_BLACK, spaceAfter=4, wordWrap="CJK")
    base.update(kw)
    return ParagraphStyle(name, **base)

def _p(text, s):
    return Paragraph(_safe(text), s)

def _tbl(data, col_widths, extra_styles=None, hdr_bg=C_NAVY, subhdr_rows=None):
    processed = []
    for ri, row in enumerate(data):
        new_row = []
        for cell in row:
            if isinstance(cell, Paragraph):
                new_row.append(cell)
            else:
                st = _cs_hdr if ri == 0 else _cs_body
                new_row.append(Paragraph(_safe(str(cell)), st))
        processed.append(new_row)

    t = Table(processed, colWidths=col_widths, repeatRows=1)
    base = [
        ("BACKGROUND",    (0, 0), (-1, 0), hdr_bg),
        ("ROWBACKGROUNDS",(0, 1), (-1,-1), [C_WHITE, C_LGRAY]),
        ("GRID",          (0, 0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",        (0, 0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1,-1), 4),
        ("BOTTOMPADDING", (0, 0), (-1,-1), 4),
        ("LEFTPADDING",   (0, 0), (-1,-1), 5),
        ("RIGHTPADDING",  (0, 0), (-1,-1), 5),
    ]
    if subhdr_rows:
        for row_i, bg in subhdr_rows:
            base += [("BACKGROUND", (0, row_i), (-1, row_i), bg)]
            for ci in range(len(processed[row_i])):
                if not isinstance(data[row_i][ci], Paragraph):
                    processed[row_i][ci] = Paragraph(
                        _safe(str(data[row_i][ci])), _cs_lblue)
    if extra_styles:
        base += extra_styles
    t.setStyle(TableStyle(base))
    return t


# ─────────────────────────────────────────
# CSV保存
# ─────────────────────────────────────────
def save_csv(results: list):
    if not results:
        return

    rows = [{
        "市場":              r["市場"],
        "ティッカー":        r["ティッカー"],
        "ファンド名":        r.get("ファンド名", ""),
        "カテゴリ":          r.get("カテゴリ", ""),
        "評価スコア":        r["評価スコア"],
        "1年リターン(%)":   r.get("1年リターン(%)", ""),
        "3年年率リターン(%)": r.get("3年年率リターン(%)", ""),
        "信託報酬(%)":       r.get("信託報酬(%)", ""),
        "シャープレシオ":    r.get("シャープレシオ", ""),
        "最大DD(1年)(%)":   r.get("最大DD(1年)(%)", ""),
        "AUM":               r.get("AUM", ""),
        "RSI14":             r.get("RSI14", ""),
        "分配金利回り(%)":   r.get("分配金利回り(%)", ""),
        "ハイリスク":        r.get("ハイリスク", False),
        "マッチ戦略数":      r["マッチ戦略数"],
        "マッチ戦略":        r["マッチ戦略"],
    } for r in results]

    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "trust_screening_all.csv"),
        index=False, encoding="utf-8-sig")

    # 戦略別サマリー
    strat_rows = []
    for sname in STRATEGIES:
        hits = [r for r in results if sname in r["マッチ戦略"]]
        jp_s = [r["ティッカー"] for r in hits if r["市場"] == "JP"]
        us_s = [r["ティッカー"] for r in hits if r["市場"] == "US"]
        strat_rows.append({
            "戦略":         sname,
            "国内ヒット数": len(jp_s),
            "海外ヒット数": len(us_s),
            "合計":         len(hits),
            "国内銘柄":     "、".join(jp_s),
            "海外銘柄":     "、".join(us_s),
        })
    pd.DataFrame(strat_rows).to_csv(
        os.path.join(OUTPUT_DIR, "trust_screening_summary.csv"),
        index=False, encoding="utf-8-sig")

    print(f"[保存] CSV: {OUTPUT_DIR}")

# ─────────────────────────────────────────
# メイン実行
# ─────────────────────────────────────────
def run_all_screens():
    """スクリーニング実行。(results, fund_data_raw) を返す"""
    print("\nスクリーニング実行中...")
    fund_data_raw = fetch_fund_data(TRUST_UNIVERSE, period="5y")

    results = []
    for ticker, raw_df in fund_data_raw.items():
        try:
            market = MARKET_MAP.get(ticker, "US")
            df     = calc_indicators(raw_df)
            info   = fetch_info(ticker)
            info["symbol"] = ticker

            fm = calc_fund_metrics(df, info)
            fm["category"] = CATEGORY_MAP.get(ticker, fm.get("category", "N/A"))

            if not passes_base_filter(df, info, market, fm):
                continue

            hit = []
            for name, fn in STRATEGIES.items():
                try:
                    if fn(df, info, market, fm):
                        hit.append(name)
                except Exception:
                    pass

            if hit:
                r    = df.iloc[-1]
                fund_name = (info.get("longName") or info.get("shortName") or ticker)
                rec = {
                    "市場":               market,
                    "ティッカー":         ticker,
                    "ファンド名":         fund_name,
                    "カテゴリ":           fm["category"],
                    "現在値":             round(float(r.Close), 2),
                    "RSI14":              round(float(r.RSI14), 1) if not pd.isna(r.RSI14) else None,
                    "1年リターン(%)":    round(fm["ret1y"], 1) if fm["ret1y"] is not None else None,
                    "3年年率リターン(%)": round(fm["ret3y_ann"], 1) if fm["ret3y_ann"] is not None else None,
                    "信託報酬(%)":        round(fm["expense"] * 100, 3) if fm["expense"] else None,
                    "シャープレシオ":     round(fm["sharpe"], 2) if fm["sharpe"] is not None else None,
                    "最大DD(1年)(%)":    round(fm["dd1y"] * 100, 1) if fm["dd1y"] is not None else None,
                    "AUM":                fm["aum"],
                    "分配金利回り(%)":   round(fm["div_yield"] * 100, 2) if fm["div_yield"] else None,
                    "マッチ戦略数":       len(hit),
                    "マッチ戦略":         " | ".join(hit),
                    "ハイリスク":         is_high_risk(df, fm),
                    "_df":               df,
                    "_info":             info,
                    "_fm":               fm,
                }
                rec["評価スコア"] = calc_trust_score(rec, fm)
                results.append(rec)
        except Exception as e:
            print(f"  [スキップ] {ticker}: {e}")

    results_sorted = sorted(results,
                            key=lambda x: (-x["評価スコア"], x["市場"], x["ティッカー"]))
    return results_sorted, fund_data_raw

# ─────────────────────────────────────────
# PDF 出力
# ─────────────────────────────────────────
def generate_pdf(results: list, fund_data_raw: dict):
    pdf_path = os.path.join(OUTPUT_DIR, f"trust_screening_report_{RUN_DATETIME}.pdf")

    # 推奨候補: ハイリスク除外 & 2戦略以上
    candidates = [r for r in results
                  if not r["ハイリスク"] and r["マッチ戦略数"] >= 2]
    top         = [r for r in results if r["評価スコア"] >= 5]
    jp_hits     = [r for r in results if r["市場"] == "JP"]
    us_hits     = [r for r in results if r["市場"] == "US"]

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=18*mm,  bottomMargin=18*mm,
        title="長期投資信託スクリーニング 総合レポート",
        author="Trust-Screener",
    )

    s_title = _style("s_title", fontName=_FONT_BOLD, fontSize=20, leading=26,
                     textColor=C_NAVY, spaceAfter=4)
    s_sub   = _style("s_sub",   fontName=_FONT_BOLD, fontSize=11, leading=16,
                     textColor=C_BLUE, spaceAfter=3)
    s_body  = _style("s_body",  fontSize=9, leading=14, spaceAfter=3)
    s_part  = _style("s_part",  fontName=_FONT_BOLD, fontSize=13, leading=18,
                     textColor=C_WHITE, spaceAfter=0)
    s_h2    = _style("s_h2",    fontName=_FONT_BOLD, fontSize=11, leading=16,
                     textColor=C_NAVY, spaceAfter=3, spaceBefore=6)
    s_note  = _style("s_note",  fontSize=7.5, leading=11, textColor=colors.HexColor("#374151"),
                     spaceAfter=2)

    PAGE_W = A4[0]
    W = PAGE_W - 36*mm

    elems = []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 表紙
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elems += [
        Spacer(1, 12*mm),
        _p("長期投資信託スクリーニング　総合レポート", s_title),
        _p("長期投資重視スクリーニング　全23戦略", s_sub),
        _p(f"実行日: {TODAY}　　対象: 国内ETF（東証）/ 海外ETF（NYSE・NASDAQ）", s_body),
        HRFlowable(width="100%", thickness=2.5, color=C_NAVY, spaceAfter=8),
        Spacer(1, 5*mm),
    ]

    sum_data = [
        ["項目", "国内", "海外", "合計"],
        ["スキャンファンド数", f"{_n_jp}本", f"{_n_us}本", f"{_n_jp+_n_us}本"],
        ["ヒットファンド数", f"{len(jp_hits)}本", f"{len(us_hits)}本", f"{len(results)}本"],
        ["評価スコア5以上", "－", "－", f"{len(top)}本"],
        ["推奨ファンド数（ハイリスク除外・2戦略以上）", "－", "－", f"{len(candidates)}本"],
    ]
    elems.append(_tbl(sum_data, [100*mm, 25*mm, 25*mm, 30*mm]))
    elems.append(Spacer(1, 6*mm))

    elems += [
        _p("【本レポートの構成】", _style("s_toc_h", fontName=_FONT_BOLD,
           fontSize=10, leading=15, spaceAfter=3, textColor=C_NAVY)),
        _p(f"　PART 1　今買うべき推奨ファンド　（{len(candidates)}本 / 各ファンドチャート・指標・投資シナリオ付き）", s_body),
        _p("　PART 2　全体スクリーニング分析　（戦略別ヒット数・全ファンド一覧・カテゴリ別分析）", s_body),
        Spacer(1, 4*mm),
    ]
    elems.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PART 1: 推奨ファンド
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    part1_hdr = Table(
        [[Paragraph("PART 1　今買うべき推奨ファンド", s_part)]],
        colWidths=[W])
    part1_hdr.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ]))
    elems += [
        part1_hdr,
        Spacer(1, 5*mm),
        _p("長期投資重視 × ハイリスク除外 × コスト効率最優先", s_sub),
        Spacer(1, 3*mm),
        _p("【選定基準】", _style("s_crit_h", fontName=_FONT_BOLD, fontSize=9.5,
           textColor=C_NAVY, spaceAfter=2, leading=14)),
        _p("・ 23戦略スクリーニングで2戦略以上にヒット", s_body),
        _p("・ 評価スコア上位（リターン・コスト・シャープレシオ・AUM 総合評価）", s_body),
        _p("・ ハイリスク除外（RSI82超の過熱 / ドローダウン-45%超 / ボラ40%超 / 信託報酬2.5%超）", s_body),
        _p("・ 推奨順位は評価スコア降順", s_body),
        Spacer(1, 4*mm),
    ]

    # 推奨ファンド概要テーブル
    ov_data = [["順位", "市場", "ティッカー", "ファンド名", "スコア",
                "1年RET%", "信託報酬%", "シャープ", "マッチ数"]]
    for rank, r in enumerate(candidates, 1):
        name_s = (r.get("ファンド名") or r["ティッカー"])[:22]
        ret1y_s  = f"{r['1年リターン(%)']:.1f}" if r.get("1年リターン(%)") is not None else "N/A"
        exp_s    = f"{r['信託報酬(%)']:.3f}" if r.get("信託報酬(%)") is not None else "N/A"
        sharpe_s = f"{r['シャープレシオ']:.2f}" if r.get("シャープレシオ") is not None else "N/A"
        ov_data.append([
            str(rank), r["市場"], r["ティッカー"], name_s,
            str(r["評価スコア"]), ret1y_s, exp_s, sharpe_s,
            str(r["マッチ戦略数"]),
        ])
    elems.append(_tbl(ov_data, [12*mm, 12*mm, 20*mm, 60*mm, 14*mm,
                                 18*mm, 18*mm, 16*mm, 14*mm]))
    elems.append(Spacer(1, 6*mm))

    # 個別ファンド詳細
    for rank, r in enumerate(candidates[:10], 1):
        ticker    = r["ティッカー"]
        fund_name = r.get("ファンド名") or ticker
        fm        = r.get("_fm") or {}
        df_raw    = fund_data_raw.get(ticker)
        df_ind    = r.get("_df")
        info      = r.get("_info") or {}

        reasons, risks, scenario = generate_buy_reasons(r, df_ind, info, r["市場"], fm)

        block = []
        block.append(PageBreak() if rank > 1 else Spacer(1, 2*mm))

        # ファンド名ヘッダー
        hdr_tbl = Table(
            [[Paragraph(f"[{rank}位] {ticker}  {fund_name[:35]}", s_part)]],
            colWidths=[W])
        hdr_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C_BLUE),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        block.append(hdr_tbl)
        block.append(Spacer(1, 3*mm))

        # 基本指標テーブル
        aum_str = "N/A"
        if fm.get("aum") and fm["aum"] > 0:
            if r["市場"] == "JP":
                aum_str = f"{fm['aum']/1e8:.0f}億円"
            else:
                aum_str = f"{fm['aum']/1e9:.1f}億USD"

        metrics_data = [
            ["ティッカー", "カテゴリ", "現在値", "1年RET", "3年年率RET",
             "信託報酬", "シャープ", "最大DD(1年)", "AUM", "RSI14"],
            [
                ticker,
                r.get("カテゴリ", "N/A")[:18],
                f"{r['現在値']:.2f}",
                f"{r['1年リターン(%)']:.1f}%" if r.get("1年リターン(%)") is not None else "N/A",
                f"{r['3年年率リターン(%)']:.1f}%" if r.get("3年年率リターン(%)") is not None else "N/A",
                f"{r['信託報酬(%)']:.3f}%" if r.get("信託報酬(%)") is not None else "N/A",
                f"{r['シャープレシオ']:.2f}" if r.get("シャープレシオ") is not None else "N/A",
                f"{r['最大DD(1年)(%)']:.1f}%" if r.get("最大DD(1年)(%)") is not None else "N/A",
                aum_str,
                str(r["RSI14"] or "N/A"),
            ],
        ]
        metrics_tbl = _tbl(metrics_data,
                           [18*mm, 30*mm, 17*mm, 17*mm, 22*mm,
                            17*mm, 16*mm, 22*mm, 22*mm, 13*mm])
        block.append(metrics_tbl)
        block.append(Spacer(1, 3*mm))

        # マッチ戦略
        strat_text = "マッチ戦略: " + r["マッチ戦略"].replace(" | ", "  /  ")
        block.append(_p(strat_text, _style(f"sstrat_{rank}", fontSize=8.5,
                        textColor=C_NAVY, leading=12, spaceAfter=3)))

        # 推奨理由・リスク
        rr_data = [["推奨理由", "リスク要因"]]
        reason_str = "\n".join(f"・{x}" for x in reasons)
        risk_str   = "\n".join(f"・{x}" for x in risks)
        rr_data.append([
            Paragraph(_safe(reason_str), _cs_body),
            Paragraph(_safe(risk_str),   _cs_body),
        ])
        block.append(_tbl(rr_data, [W*0.55, W*0.45], hdr_bg=C_GREEN))
        block.append(Spacer(1, 2*mm))

        # シナリオ
        block.append(_p(f"■ 投資シナリオ: {scenario}", _style(f"sscen_{rank}",
                        fontName=_FONT_BOLD, fontSize=8.5, textColor=C_NAVY,
                        leading=13, spaceAfter=3)))

        block.append(Spacer(1, 4*mm))

        elems.extend(block)

    elems.append(PageBreak())

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PART 2: 全体スクリーニング分析
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    part2_hdr = Table(
        [[Paragraph("PART 2　全体スクリーニング分析", s_part)]],
        colWidths=[W])
    part2_hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ]))
    elems += [part2_hdr, Spacer(1, 5*mm)]

    # 戦略別ヒット数
    elems.append(_p("■ 戦略別ヒット数", s_h2))
    strat_data = [["戦略名", "国内", "海外", "合計", "代表ファンド（上位3）"]]
    for sname in STRATEGIES:
        hits    = [r for r in results if sname in r["マッチ戦略"]]
        jp_hits2 = [r["ティッカー"] for r in hits if r["市場"] == "JP"]
        us_hits2 = [r["ティッカー"] for r in hits if r["市場"] == "US"]
        top3    = ", ".join([r["ティッカー"] for r in hits[:3]])
        strat_data.append([sname, str(len(jp_hits2)), str(len(us_hits2)),
                           str(len(hits)), top3 or "－"])
    elems.append(_tbl(strat_data, [60*mm, 14*mm, 14*mm, 14*mm, 72*mm]))
    elems.append(Spacer(1, 6*mm))

    # 全ファンド一覧
    elems.append(_p("■ ヒット全ファンド一覧", s_h2))
    all_data = [["市場", "ティッカー", "ファンド名", "カテゴリ", "スコア",
                 "1年RET%", "信託報酬%", "シャープ", "AUM", "戦略数"]]
    for r in results:
        aum_s = "N/A"
        if fm2 := r.get("_fm"):
            if fm2.get("aum") and fm2["aum"] > 0:
                aum_s = (f"{fm2['aum']/1e8:.0f}億円" if r["市場"] == "JP"
                         else f"{fm2['aum']/1e9:.1f}BUSD")
        all_data.append([
            r["市場"],
            r["ティッカー"],
            (r.get("ファンド名") or r["ティッカー"])[:20],
            r.get("カテゴリ", "N/A")[:15],
            str(r["評価スコア"]),
            f"{r['1年リターン(%)']:.1f}" if r.get("1年リターン(%)") is not None else "N/A",
            f"{r['信託報酬(%)']:.3f}" if r.get("信託報酬(%)") is not None else "N/A",
            f"{r['シャープレシオ']:.2f}" if r.get("シャープレシオ") is not None else "N/A",
            aum_s,
            str(r["マッチ戦略数"]),
        ])
    elems.append(_tbl(all_data, [12*mm, 18*mm, 44*mm, 30*mm, 14*mm,
                                  18*mm, 18*mm, 16*mm, 24*mm, 12*mm]))
    elems.append(Spacer(1, 6*mm))

    # フッター注記
    elems += [
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#9ca3af")),
        Spacer(1, 2*mm),
        _p("※ 本レポートは情報提供を目的としたものであり、投資勧誘を目的とするものではありません。"
           "投資に関する最終決定はご自身の判断と責任のもとで行ってください。"
           "データはyfinanceを通じて取得しており、遅延・欠損が含まれる場合があります。", s_note),
        _p(f"生成日時: {TODAY}　Trust-Screener v1.0", s_note),
    ]

    doc.build(elems)
    print(f"\n[OK] PDF出力完了: {pdf_path}")
    return pdf_path


# ─────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────
if __name__ == "__main__":
    results, fund_data_raw = run_all_screens()

    print(f"\n--- スクリーニング結果 ---")
    print(f"ヒットファンド: {len(results)}本")
    candidates = [r for r in results if not r["ハイリスク"] and r["マッチ戦略数"] >= 2]
    print(f"推奨ファンド:  {len(candidates)}本（ハイリスク除外・2戦略以上）")

    save_csv(results)

    if results:
        pdf_path = generate_pdf(results, fund_data_raw)
        print(f"保存先: {pdf_path}")
    else:
        print("ヒットなし。スクリーニング基準を確認してください。")
