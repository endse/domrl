import pandas as pd

try:
    df = pd.read_csv("logs/evaluation_results.csv")
    print(df.groupby("scenario")[["reward", "churned", "steps"]].mean())
except Exception as e:
    print(e)
