import os
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from utils import animate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Directory to input CSV file(s).")
    parser.add_argument("--output", "-o", type=str, default="out/animations", help="Directory to save animations.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    input_path = Path(args.input)
    os.makedirs(args.output, exist_ok=True)
    for csv_path in tqdm(sorted(input_path.glob("*.csv"))):
        df = pd.read_csv(csv_path, index_col="timestamp")
        save_path = os.path.join(args.output, f"{csv_path.stem}.gif")
        animate(df, save_path=save_path)