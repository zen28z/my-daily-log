"""
日米業種リードラグ投資戦略
部分空間正則化付きPCAを用いた翌日ロング・ショートシグナル生成
"""

import numpy as np
import pandas as pd
import yfinance as yf
from numpy.linalg import eigh
from datetime import datetime, timedelta
import warnings
import json
import os

warnings.filterwarnings("ignore")

# ============================================================
# 設定
# ============================================================

US_TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "XLC"]
JP_TICKERS = [f"{i}.T" for i in range(1617, 1634)]

ALL_TICKERS = US_TICKERS + JP_TICKERS
N_US = len(US_TICKERS)
N_JP = len(JP_TICKERS)
N = N_US + N_JP

WINDOW = 60
LAMBDA = 0.9
K = 3
Q = 0.30

US_CYCLICAL   = {"XLB", "XLE", "XLF", "XLRE"}
US_DEFENSIVE  = {"XLK", "XLP", "XLU", "XLV"}
JP_CYCLICAL   = {"1618.T", "1625.T", "1629.T", "1631.T"}
JP_DEFENSIVE  = {"1617.T", "1621.T", "1627.T", "1630.T"}

# 日本ETFのセクター名マッピング（ダッシュボード表示用）
JP_ETF_NAMES = {
    "1617.T": "食品", "1618.T": "エネルギー資源", "1619.T": "建設・資材",
    "1620.T": "素材・化学", "1621.T": "医薬品", "1622.T": "自動車・輸送機",
    "1623.T": "鉄鋼・非鉄", "1624.T": "機械", "1625.T": "電機・精密",
    "1626.T": "情報通信・サービスその他", "1627.T": "電力・ガス",
    "1628.T": "運輸・物流", "1629.T": "商社・卸売", "1630.T": "小売",
    "1631.T": "銀行", "1632.T": "金融（除く銀行）", "1633.T": "不動産"
}

# ============================================================
# 計算ロジック（Step 0 ～ 6 は元のまま）
# ============================================================

def fetch_data(tickers: list, days: int = 120) -> pd.DataFrame:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days * 2)
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]
    raw = raw.reindex(columns=tickers)
    returns = raw.pct_change().dropna(how="all")
    returns = returns.dropna(how="any")
    return returns

def standardize(returns_window: pd.DataFrame):
    window = returns_window.iloc[:-1]
    today  = returns_window.iloc[-1]
    mu    = window.mean()
    sigma = window.std(ddof=1)
    sigma = sigma.replace(0, np.nan)
    z_today = (today - mu) / sigma
    return z_today, mu, sigma

def compute_correlation_matrix(returns_window: pd.DataFrame) -> np.ndarray:
    window = returns_window.iloc[:-1]
    mu    = window.mean()
    sigma = window.std(ddof=1).replace(0, np.nan)
    Z = ((window - mu) / sigma).fillna(0.0)
    L = len(Z)
    C = Z.values.T @ Z.values / (L - 1)
    d = np.sqrt(np.diag(C))
    d[d == 0] = 1.0
    C = C / np.outer(d, d)
    return C

def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    basis = []
    for v in vectors.T:
        w = v.copy().astype(float)
        for b in basis:
            w -= np.dot(w, b) * b
        norm = np.linalg.norm(w)
        if norm > 1e-10:
            basis.append(w / norm)
    return np.column_stack(basis)

def build_prior_subspace(tickers: list) -> np.ndarray:
    N = len(tickers)
    v1 = np.ones(N) / np.sqrt(N)
    v2_raw = np.array([1.0 if t in US_TICKERS else -1.0 for t in tickers])
    v3_raw = np.zeros(N)
    for i, t in enumerate(tickers):
        if t in US_CYCLICAL or t in JP_CYCLICAL:
            v3_raw[i] = 1.0
        elif t in US_DEFENSIVE or t in JP_DEFENSIVE:
            v3_raw[i] = -1.0
    raw_mat = np.column_stack([v1, v2_raw, v3_raw])
    V0 = gram_schmidt(raw_mat)
    return V0

def build_C0(V0: np.ndarray, C_full: np.ndarray) -> np.ndarray:
    D0 = np.diag(np.diag(V0.T @ C_full @ V0))
    C0_raw = V0 @ D0 @ V0.T
    d = np.sqrt(np.diag(C0_raw))
    d[d == 0] = 1.0
    C0 = C0_raw / np.outer(d, d)
    np.fill_diagonal(C0, 1.0)
    return C0

def subspace_regularized_pca(Ct: np.ndarray, C0: np.ndarray, lam: float = LAMBDA, k: int = K) -> np.ndarray:
    C_reg = (1 - lam) * Ct + lam * C0
    eigenvalues, eigenvectors = eigh(C_reg)
    idx = np.argsort(eigenvalues)[::-1]
    V_k = eigenvectors[:, idx[:k]]
    return V_k

def compute_signal(V_k: np.ndarray, z_u_today: np.ndarray) -> np.ndarray:
    V_U = V_k[:N_US, :]
    V_J = V_k[N_US:, :]
    f = V_U.T @ z_u_today
    signal_jp = V_J @ f
    return signal_jp

def select_portfolio(signal_jp: np.ndarray, jp_tickers: list, q: float = Q) -> tuple[list, list]:
    n = len(signal_jp)
    n_select = max(1, round(n * q))
    ranked = sorted(range(n), key=lambda i: signal_jp[i], reverse=True)
    long_idx  = ranked[:n_select]
    short_idx = ranked[-n_select:]
    long_tickers  = [jp_tickers[i] for i in long_idx]
    short_tickers = [jp_tickers[i] for i in short_idx]
    return long_tickers, short_tickers

# ============================================================
# メイン処理 ＆ JSON保存
# ============================================================

def run_strategy():
    returns = fetch_data(ALL_TICKERS, days=120)
    if len(returns) < WINDOW + 5:
        print(f"[ERROR] データが不足しています。")
        return

    returns_window = returns.iloc[-(WINDOW + 1):]
    z_today, mu, sigma = standardize(returns_window)
    z_u_today = z_today[US_TICKERS].fillna(0.0).values
    Ct = compute_correlation_matrix(returns_window)
    V0 = build_prior_subspace(ALL_TICKERS)

    all_window = returns.copy()
    mu_full    = all_window.mean()
    sigma_full = all_window.std(ddof=1).replace(0, np.nan)
    Z_full     = ((all_window - mu_full) / sigma_full).fillna(0.0)
    L_full     = len(Z_full)
    C_full_raw = Z_full.values.T @ Z_full.values / (L_full - 1)
    d_cf       = np.sqrt(np.diag(C_full_raw))
    d_cf[d_cf == 0] = 1.0
    C_full     = C_full_raw / np.outer(d_cf, d_cf)
    np.fill_diagonal(C_full, 1.0)

    C0 = build_C0(V0, C_full)
    V_k = subspace_regularized_pca(Ct, C0, lam=LAMBDA, k=K)
    signal_jp = compute_signal(V_k, z_u_today)
    
    long_tickers, short_tickers = select_portfolio(signal_jp, JP_TICKERS, q=Q)

    # ==== ここからダッシュボード用JSON出力処理 ====
    # 日本時間の取得
    jst_now = datetime.utcnow() + timedelta(hours=9)
    date_str = jst_now.strftime("%Y-%m-%d")
    time_label = jst_now.strftime("%m月%d日 朝%H時%M分")

    # ダッシュボード用に銘柄と名前を結合
    long_data = [{"code": t, "name": JP_ETF_NAMES.get(t, "不明")} for t in long_tickers]
    short_data = [{"code": t, "name": JP_ETF_NAMES.get(t, "不明")} for t in short_tickers]

    file_path = 'data/logs.json'
    
    # dataフォルダが存在しない場合は作成
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 既存のログを読み込み
    logs = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                pass

    # 新しい結果を追加
    logs.append({
        "date": date_str,
        "time": time_label,
        "long": long_data,
        "short": short_data
    })

    # 最新の50件（約10日分）を保存
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(logs[-50:], f, ensure_ascii=False, indent=2)
    
    print(f"{time_label} のシグナルを logs.json に保存しました。")

if __name__ == "__main__":
    run_strategy()
