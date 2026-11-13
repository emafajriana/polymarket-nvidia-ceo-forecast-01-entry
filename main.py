#!/usr/bin/env python3
"""Entry point for the polymarket-nvidia-ceo-forecast-01 forecaster.

Reads every case file (*.in) in the input directory, forecasts each market's
next 240 minutes as five quantiles per minute via the quantshape/volmodel
modules, and writes one <id>.out file per case to the output directory.

The grader runs this as `python3 -P -s /bot/main.py`, where the script's own
directory is not on sys.path, so it is inserted explicitly before importing
the sibling modules. Input/output directories come from MAKEFUN_INPUT_DIR /
MAKEFUN_OUTPUT_DIR when set, else /input and /output.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantshape import forecast_case

# Final calibrated parameters (filled in after local practice calibration;
# an empty dict means quantshape.DEFAULT_PARAMS apply unchanged).
PARAMS = {}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    for name in names:
        case_id = name[: -len(".in")]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        forecasts = forecast_case(case, PARAMS if PARAMS else None)
        payload = {
            "forecasts": [
                {"id": m["id"], "q": forecasts[m["id"]]} for m in case["markets"]
            ]
        }
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, separators=(",", ":")))
            fh.write("\n")


if __name__ == "__main__":
    main()
