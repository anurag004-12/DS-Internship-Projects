"""Look at the most extreme player in the Cookie Cats file.
Run from the backend folder:  python scripts/inspect_cookie_cats.py "path/to/cookie_cats.txt"
"""
import sys

import pandas as pd

if len(sys.argv) != 2:
    sys.exit('Usage: python scripts/inspect_cookie_cats.py "path/to/cookie_cats.txt"')

df = pd.read_csv(sys.argv[1])
biggest = df.sum_gamerounds.idxmax()

print("Player with the most game rounds:")
print(df.loc[biggest].to_string(), "\n")

print("Average, median and maximum rounds per group:")
print(df.groupby("version").sum_gamerounds.agg(["mean", "median", "max"]).round(3), "\n")

print("Average rounds per group WITHOUT that one player:")
print(df.drop(biggest).groupby("version").sum_gamerounds.mean().round(3))