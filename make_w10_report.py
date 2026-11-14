"""W10: bangun tabel hasil validation dari experiments/experiment_log.csv.

Output (validation saja, tidak menyentuh test):
    reports/w10/hasil_validation.csv   E1, E2 (3), E3 (6), dan E4 terpilih
    reports/w10/hasil_validation.md    tabel yang sama + grid E4 (alpha x beta)
Semua angka dibaca dari log; konfigurasi terpilih diambil dari experiments/selection.json
(hasil aturan seleksi pre-registered di configs/cp3_search_space.json).

Run: python make_w10_report.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "experiments" / "experiment_log.csv"
SELECTION = ROOT / "experiments" / "selection.json"
OUT = ROOT / "reports" / "w10"

METRICS = ["recall@10", "ndcg@10", "catalog_coverage@10", "long_tail_exposure@10", "long_tail_recall@10"]
HEAD = ["Run", "Model", "Parameter", "Recall@10", "NDCG@10", "Coverage@10", "LT exposure@10", "LT recall@10", "Terpilih"]


def fmt_params(row):
    p = json.loads(row["params"])
    return ", ".join(f"{k}={v}" for k, v in p.items()) or "-"


def md_table(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    for _, r in df.iterrows():
        cells = [f"{v:.5f}" if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    log = pd.read_csv(LOG)
    assert (log["split"] == "validation").all(), "log berisi split selain validation"
    selected = {exp: info["run_id"] for exp, info in json.loads(SELECTION.read_text(encoding="utf-8")).items()}

    keep = log[log.experiment.isin(["E1", "E2", "E3"]) | log.run_id.eq(selected["E4"])].copy()
    keep["Terpilih"] = keep.run_id.map(lambda r: "ya" if r in selected.values() else "")
    keep["Parameter"] = keep.apply(fmt_params, axis=1)
    table = keep[["run_id", "model", "Parameter", *METRICS, "Terpilih"]]
    table.columns = HEAD
    table = table.reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "hasil_validation.csv", index=False)

    e4 = log[log.experiment == "E4"].copy()
    e4[["alpha", "beta"]] = e4.params.apply(lambda s: pd.Series({k: json.loads(s)[k] for k in ("alpha", "beta")}))
    grids = []
    for col, name in [("ndcg@10", "NDCG@10"), ("long_tail_exposure@10", "Long-tail exposure@10")]:
        pv = e4.pivot(index="alpha", columns="beta", values=col)
        pv.columns = [f"beta={b}" for b in pv.columns]
        pv.insert(0, "alpha", [f"{a}" for a in pv.index])
        grids.append(f"### Grid E4: {name} (baris = alpha, kolom = beta)\n\n{md_table(pv.reset_index(drop=True))}\n")

    md = [
        "# Hasil Validation W10",
        "",
        f"Sumber: `experiments/experiment_log.csv` ({len(log)} run, seed 42, split validation, "
        f"{int(log['users_evaluated'].iloc[0])} evaluable users, git_dirty semua "
        f"{'False' if not log.git_dirty.astype(str).str.lower().eq('true').any() else 'TIDAK'}). "
        "Test split tidak dipakai.",
        "",
        "Kolom *Terpilih* menandai konfigurasi yang dipilih aturan seleksi pre-registered "
        "(E2/E3: NDCG@10 tertinggi; E4: long-tail exposure tertinggi dengan NDCG@10 >= 0,95 x NDCG@10 MF terpilih).",
        "",
        "## Tabel utama",
        "",
        md_table(table),
        "",
        *grids,
        "",
    ]
    (OUT / "hasil_validation.md").write_text("\n".join(md), encoding="utf-8")
    print(f"ditulis: {OUT / 'hasil_validation.csv'}, {OUT / 'hasil_validation.md'}")


if __name__ == "__main__":
    main()
