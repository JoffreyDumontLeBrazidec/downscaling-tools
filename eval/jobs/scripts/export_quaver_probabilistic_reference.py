"""Export quaver probabilistic score curves to CSV for local comparison.

Run this through the quaver binary, not plain Python, because the quaver module
sets up its private Python/runtime dependencies:

    module load quaver
    export TMPDIR=/path/to/scratch/tmp
    quaver eval/jobs/export_quaver_probabilistic_reference.py --out-csv /path/ref.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _csv_list(value: str, *, cast=str) -> list:
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def _export(args: argparse.Namespace) -> int:
    from quaver.quaver.DBAccess import DBAccess

    scores = _csv_list(args.scores)
    parameters = _csv_list(args.parameters)
    domains = _csv_list(args.domains)
    steps = _csv_list(args.steps, cast=int)

    db = DBAccess(args.database)
    agent = db.openDB(args.database)
    rows: list[dict] = []
    for score in scores:
        with agent.cursor() as cursor:
            cursor.execute(
                f"""
                select step, parameter, domain_name, avg(value)::float, count(*)
                from y_{args.year}.v_{score}
                where date between %s and %s
                  and step = any(%s)
                  and parameter = any(%s)
                  and domain_name = any(%s)
                  and vstream = %s
                  and stream = %s
                  and type = %s
                  and expver = %s
                  and grid = %s
                group by step, parameter, domain_name
                order by parameter, domain_name, step
                """,
                (
                    args.first_date * 100,
                    args.last_date * 100,
                    steps,
                    parameters,
                    domains,
                    args.vstream,
                    args.stream,
                    args.type,
                    args.expver,
                    args.grid,
                ),
            )
            for step, parameter, domain_name, value, count in cursor.fetchall():
                rows.append(
                    {
                        "step": int(step),
                        "weather_state": parameter,
                        "domain": domain_name,
                        "metric": score,
                        "value": value,
                        "n_dates": count,
                        "label": args.label,
                    }
                )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "weather_state", "domain", "metric", "value", "n_dates", "label"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(args.out_csv)
    print("rows", len(rows))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--database", default="rd_3")
    parser.add_argument("--year", default=2023, type=int)
    parser.add_argument("--first-date", default=20230816, type=int)
    parser.add_argument("--last-date", default=20230830, type=int)
    parser.add_argument("--steps", default="24,48,72,96,120")
    parser.add_argument("--parameters", default="2t,2d,10ff")
    parser.add_argument("--domains", default="n.hem,tropics,s.hem,europe")
    parser.add_argument("--scores", default="fcrps,spread,crps")
    parser.add_argument("--vstream", default="prepml_0001_ob")
    parser.add_argument("--stream", default="enfo")
    parser.add_argument("--type", default="pf")
    parser.add_argument("--expver", default="0001")
    parser.add_argument("--grid", default="O320")
    parser.add_argument("--label", default="quaver_enfo_o320_ob")
    raise SystemExit(_export(parser.parse_args()))


main()
