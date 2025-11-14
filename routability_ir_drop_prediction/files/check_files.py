from pathlib import Path
import sys

#!/usr/bin/env python3

loc = "/home/ec2-user/CircuitNet/IR_drop_training_set/IR_drop"



def main():
    exists = []
    notexists = []
    listfile = Path("test_N14.csv")
    if not listfile.exists():
        print(f"Error: '{listfile}' not found", file=sys.stderr)
        sys.exit(1)

    with listfile.open("r", encoding="utf-8") as fh:
        for raw in fh:
            name = raw.strip()
            name = name.split(",")[0]
            if not name or name.startswith("#"):
                continue
            p = Path(name)
            if not p.is_absolute():
                p = Path(loc) / name
            if not p.exists():
                notexists.append(str(p))
            else:
                exists.append(str(p))
    print(f"Found {len(exists)} files, missing {len(notexists)} files.")
    return exists, notexists
if __name__ == "__main__":
    exists, notexists = main()
    print(f"example: found: {exists[:3]}")
    print(f"example: missing: {notexists[:3]}")