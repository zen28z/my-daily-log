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
import json  # 追加：JSON保存用
import os    # 追加：フォルダ作成用

warnings.filterwarnings("ignore")

# ============================================================
# 設定
# ============================================================

# 米国ETFティッカー（11銘柄）
US_TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "XLC"]
# 日本ETFティッカー（17銘柄）
JP_TICKERS = [f"{i}.T" for i in range(1617, 1634)]

ALL_TICKERS = US_TICKERS + JP_TICKERS
N_US = len(US_TICKERS)   # 11
N_JP = len(JP_TICKERS)   # 17
N = N_US + N_JP          # 28

# ハイパーパラメータ
WINDOW = 60        # 推定ウィンドウ（営業日）
LAMBDA = 0.9       # 正則化強度
K = 3              # 上位固有ベクトル数（共通ファクター数）
Q = 0.30           # ロング・ショートの分位点（上下30%）

# シクリカル/ディフェンシブラベル（論文 Section 4.1）
US_CYCLICAL   = {"XLB", "XLE", "XLF", "XLRE"}
US_DEFENSIVE  = {"XLK", "XLP", "XLU", "XLV"}
JP_CYCLICAL   = {"1618.T", "1625.T", "1629.T", "1631.T"}
JP_DEFENSIVE  = {"1617.T", "1621.T", "1627.T", "1630.T"}

# 画面表示用：日本ETFのセクター名（追加）
JP_ETF_NAMES = {
    "1617.T": "食品", "1618.T": "エネルギー資源", "1619.T": "建設・資材",
    "1620.T": "素材・化学", "1621.T": "医薬品", "1622.T": "自動車・輸送機",
    "1623.T": "鉄鋼・非鉄", "1624.T": "機械", "1625.T": "電機・精密",
    "1626.T": "情報通信・サービスその他", "1627.T": "電力・ガス",
    "1628.T": "運輸・物流", "1629.T": "商社・卸売", "1630.T": "小売",
    "1631.T": "銀行", "1632.T": "金融（除く銀行）", "1633.T": "不動産"
}

# ============================================================
# Step 0: データ取得
# ============================================================

def fetch_data(tickers: list, days: int = 120) -> pd.DataFrame:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days * 2)

    print(f"データ取得中... ({start_date.date()} ～ {end_date.date()})")
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]
    raw = raw.reindex(columns=tickers)
    returns = raw.pct_change().dropna(how="all")
    returns = returns.dropna(how="any")

    print(f"  有効営業日数: {len(returns)} 日")
    return returns

# ============================================================
# Step 1: 前処理 — 標準化リターン（Zスコア）
# ============================================================

def standardize(returns_window: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
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

# ============================================================
# Step 2: 事前部分空間 V0 の構築
# ============================================================

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

# ============================================================
# Step 3: 事前エクスポージャー行列 C0 の構築
# ============================================================

def build_C0(V0: np.ndarray, C_full: np.ndarray) -> np.ndarray:
    D0 = np.diag(np.diag(V0.T @ C_full @ V0))
    C0_raw = V0 @ D0 @ V0.T
    d = np.sqrt(np.diag(C0_raw))
    d[d == 0] = 1.0
    C0 = C0_raw / np.outer(d, d)
    np.fill_diagonal(C0, 1.0)
    return C0

# ============================================================
# Step 4: 部分空間正則化PCA
# ============================================================

def subspace_regularized_pca(Ct: np.ndarray, C0: np.ndarray, lam: float = LAMBDA, k: int = K) -> np.ndarray:
    C_reg = (1 - lam) * Ct + lam * C0
    eigenvalues, eigenvectors = eigh(C_reg)
    idx = np.argsort(eigenvalues)[::-1]
    V_k = eigenvectors[:, idx[:k]]
    return V_k

# ============================================================
# Step 5: シグナル生成
# ============================================================

def compute_signal(V_k: np.ndarray, z_u_today: np.ndarray) -> np.ndarray:
    V_U = V_k[:N_US, :]
    V_J = V_k[N_US:, :]
    f = V_U.T @ z_u_today
    signal_jp = V_J @ f
    return signal_jp

# ============================================================
# Step 6: ロング・ショート銘柄の決定
# ============================================================

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
# メイン処理
# ============================================================

def run_strategy():
    print("=" * 60)
    print("日米業種リードラグ投資戦略（部分空間正則化PCA）")
    print("=" * 60)

    # ---- データ取得 ----
    returns = fetch_data(ALL_TICKERS, days=120)

    if len(returns) < WINDOW + 5:
        print(f"[ERROR] データが不足しています（{len(returns)}日分）。")
        return

    # 直近WINDOW+1日分を使用
    returns_window = returns.iloc[-(WINDOW + 1):]

    # ---- Step 1: 当日標準化リターン ----
    z_today, mu, sigma = standardize(returns_window)
    z_u_today = z_today[US_TICKERS].fillna(0.0).values

    # ---- Step 1: ウィンドウ内相関行列 Ct ----
    Ct = compute_correlation_matrix(returns_window)

    # ---- Step 2: 事前部分空間 V0 ----
    V0 = build_prior_subspace(ALL_TICKERS)

    # ---- Step 3: 長期相関行列（全データ使用）と C0 ----
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

    # ---- Step 4: 部分空間正則化PCA ----
    V_k = subspace_regularized_pca(Ct, C0, lam=LAMBDA, k=K)

    # ---- Step 5: シグナル生成 ----
    signal_jp = compute_signal(V_k, z_u_today)
    signal_series = pd.Series(signal_jp, index=JP_TICKERS)

    # ---- Step 6: ロング・ショート決定 ----
    long_tickers, short_tickers = select_portfolio(signal_jp, JP_TICKERS, q=Q)

    # ---- ターミナル出力（GitHub Actionsのログ確認用） ----
    today_str = returns_window.index[-1].strftime("%Y-%m-%d")
    print(f"\n使用した米国データの最終日（当日）: {today_str}")
    print(f"ウィンドウ: 過去{WINDOW}営業日 / λ={LAMBDA} / K={K} / q={Q}")

    print("\n" + "=" * 60)
    print(f"📈 明日【ロング（買い）】すべき日本ETF （上位{int(Q*100)}%）")
    print("=" * 60)
    for t in long_tickers:
        print(f"  {t}  スコア: {signal_series[t]:+.4f}")

    print("\n" + "=" * 60)
    print(f"📉 明日【ショート（空売り）】すべき日本ETF （下位{int(Q*100)}%）")
    print("=" * 60)
    for t in short_tickers:
        print(f"  {t}  スコア: {signal_series[t]:+.4f}")

    print("\n" + "=" * 60)
    print("全銘柄の予測スコア（降順）")
    print("=" * 60)
    sorted_signal = signal_series.sort_values(ascending=False)
    for ticker, score in sorted_signal.items():
        print(f"  {ticker}  {score:+.4f}")

    # ============================================================
    # 【追加】 GitHub Pages ダッシュボード用の JSON 出力処理
    # ============================================================
    
    # 日本時間 (JST) で時刻を取得
    jst_now = datetime.utcnow() + timedelta(hours=9)
    date_str = jst_now.strftime("%Y-%m-%d")
    time_label = jst_now.strftime("%m月%d日 朝%H時%M分")

    # セクター名を紐付けながら辞書型に変換
    long_data = [{"code": t, "name": JP_ETF_NAMES.get(t, "不明")} for t in long_tickers]
    short_data = [{"code": t, "name": JP_ETF_NAMES.get(t, "不明")} for t in short_tickers]

    # 保存先のパス設定（dataフォルダ）
    file_path = 'data/logs.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 既存のデータを読み込み
    logs = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                pass

    # 新しい結果をリストの最後に追加
    logs.append({
        "date": date_str,
        "time": time_label,
        "long": long_data,
        "short": short_data
    })

    # JSONファイルに書き出し（最新の50件のみ保持して容量肥大化を防ぐ）
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(logs[-50:], f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print(f"[SUCCESS] {time_label} のシグナルを logs.json に保存しました！")
    print("=" * 60)

    return long_tickers, short_tickers, signal_series


if __name__ == "__main__":
    run_strategy()
