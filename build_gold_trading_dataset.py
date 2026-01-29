port yfinance as yf
import pandas as pd
import numpy as np

SYMBOL = "GC=F"
START_DATE = "2025-01-01"
OUTPUT_FILE = "data_trading_gold_real.csv"

df = yf.download(SYMBOL, start=START_DATE, interval="1d")
df = df.dropna()

df["return"] = df["Close"].pct_change()
df["momentum_5d"] = df["return"].rolling(5).mean()
df["volatility_10d"] = df["return"].rolling(10).std()
df["volume_norm"] = df["Volume"] / df["Volume"].rolling(20).mean()

out = pd.DataFrame({
    "date": df.index.astype(str),
    "mood": df["momentum_5d"],
    "feature_activity": df["volume_norm"],
    "feature_sleep": 1 / (1 + df["volatility_10d"]),
    "feature_stress": df["volatility_10d"],
    "target": df["return"].shift(-1)
})

out = out.dropna()
out.to_csv(OUTPUT_FILE, index=False)

print("Dataset created:", OUTPUT_FILE)

out_path = "runs/data_trading_gold_real.csv"
os.makedirs("runs", exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Saved dataset to {out_path}")
