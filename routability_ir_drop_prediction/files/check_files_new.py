from pathlib import Path
import sys
import pandas as pd

#!/usr/bin/env python3

loc = "/home/ec2-user/CircuitNet/IR_drop_training_set/IR_drop"
new_csv_name = "train_N14_checked.csv"
csv_name = "train_N14.csv"

def check_file_exists(loc=loc, file_path=None):
    p = Path(file_path)
    if not p.is_absolute():
        p = Path(loc) / file_path
    return p.exists()

def check_csv(loc = loc, csv_name=csv_name, new_csv_name=new_csv_name):
    df = pd.read_csv(csv_name, header=None)
    for index, row in df.iterrows():
        if not check_file_exists(loc, row[0]) and not check_file_exists(loc, row[1]):
            print(f"Removing row {index} : {row[0]}, {row[1]}")
            df = df.drop(index)
    df.to_csv(new_csv_name, index=False)

if __name__ == "__main__":
    check_csv()