#!/usr/bin/env python3
"""
List all CSV objects stored in OSS, and write a full inventory to a local file.

Default output: output/oss_inventory.csv
Each row: key,satellite,bag,filename,year,week,size,last_modified

Example:
  python oss/list_oss_csvs.py
  python oss/list_oss_csvs.py --prefix E/ --stdout
"""
import argparse
import csv
import os
import re
import sys

try:
    import oss2
except Exception as exc:
    print("ERROR: oss2 is not installed:", exc)
    print("Please run: python -m pip install oss2")
    sys.exit(1)

try:
    import config
except Exception:
    config = None


def get_config_value(name, default=None):
    if config is not None and hasattr(config, name):
        return getattr(config, name)
    return os.environ.get(name, default)


ACCESS_KEY_ID = get_config_value("OSS_ACCESS_KEY_ID")
ACCESS_KEY_SECRET = get_config_value("OSS_ACCESS_KEY_SECRET")
ENDPOINT = get_config_value("OSS_ENDPOINT", "oss-cn-beijing.aliyuncs.com")
BUCKET_NAME = get_config_value("OSS_BUCKET_NAME")

OUTPUT_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "output",
    "oss_inventory.csv",
)

YEAR_WEEK_RE = re.compile(r"^(?P<year>\\d{4})_(?P<week>\\d{2})\\.csv$", re.IGNORECASE)


def iter_objects(bucket, prefix=""):
    marker = ""
    while True:
        res = bucket.list_objects(prefix=prefix, marker=marker, max_keys=1000)
        for obj in res.object_list:
            yield obj
        if not res.is_truncated:
            break
        marker = res.next_marker


def parse_key(key):
    parts = key.strip("/").split("/")
    satellite = parts[0] if len(parts) >= 3 else ""
    bag = parts[1] if len(parts) >= 3 else ""
    filename = parts[-1] if parts else ""
    year = ""
    week = ""
    m = YEAR_WEEK_RE.match(filename)
    if m:
        year = m.group("year")
        week = m.group("week")
    return satellite, bag, filename, year, week


def main():
    parser = argparse.ArgumentParser(description="List all OSS CSV objects.")
    parser.add_argument("--prefix", default="", help="Optional OSS prefix filter (e.g. E/)")
    parser.add_argument("--out", default=OUTPUT_DEFAULT, help="Output CSV path")
    parser.add_argument("--stdout", action="store_true", help="Print each row to stdout")
    args = parser.parse_args()

    if not ACCESS_KEY_ID or "YOUR_ACCESS_KEY" in ACCESS_KEY_ID:
        print("ERROR: OSS_ACCESS_KEY_ID is not configured")
        sys.exit(2)
    if not ACCESS_KEY_SECRET or "YOUR_ACCESS_KEY" in ACCESS_KEY_SECRET:
        print("ERROR: OSS_ACCESS_KEY_SECRET is not configured")
        sys.exit(2)
    if not BUCKET_NAME:
        print("ERROR: OSS_BUCKET_NAME is not configured")
        sys.exit(2)

    auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
    bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total = 0
    summary = {}

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "key",
            "satellite",
            "bag",
            "filename",
            "year",
            "week",
            "size",
            "last_modified",
        ])

        for obj in iter_objects(bucket, prefix=args.prefix):
            if not obj.key.lower().endswith(".csv"):
                continue
            satellite, bag, filename, year, week = parse_key(obj.key)
            row = [
                obj.key,
                satellite,
                bag,
                filename,
                year,
                week,
                obj.size,
                obj.last_modified,
            ]
            writer.writerow(row)
            total += 1

            if satellite and bag:
                summary.setdefault(satellite, {}).setdefault(bag, 0)
                summary[satellite][bag] += 1

            if args.stdout:
                print(",".join(str(x) for x in row))

    print(f"Total CSV objects: {total}")
    print(f"Inventory written to: {args.out}")
    if summary:
        print("\nSummary by satellite/bag:")
        for sat in sorted(summary.keys()):
            print(f"- {sat}")
            for bag in sorted(summary[sat].keys()):
                print(f"  - {bag}: {summary[sat][bag]}")


if __name__ == "__main__":
    main()
