"""
Error analysis for Part 2 (Q6).

Loads the model's DEV predictions and the ground truth, executes both against the flight
database, and classifies every example where the generated query is wrong (record exact
match == False). Categories are heuristic and based on *actual* observed differences between
generated and gold SQL / records; nothing here is fabricated.

Inputs (defaults match train_t5.py's dev output naming for --finetune --experiment_name experiment):
    --pred_sql     results/t5_ft_experiment_dev.sql
    --gold_sql     data/dev.sql
    --nl           data/dev.nl
    --gold_records records/ground_truth_dev.pkl
    --out          results/error_analysis_dev.md   (markdown report of real errors + counts)

Run (after you have generated dev predictions):
    python3 error_analysis.py
"""
import os
import re
import argparse
import pickle
from collections import Counter

from utils import load_queries_and_records, compute_record_F1


# ---- SQL feature extraction for categorizing errors ---------------------------------------

def _norm(sql):
    return re.sub(r'\s+', ' ', sql.strip().lower())


def _tables(sql):
    # crude: table names appear after FROM / JOIN, aliased like "flight flight_1"
    s = _norm(sql)
    toks = re.findall(r'from\s+(.*?)(?:\s+where|\s*$|\s*\))', s)
    tabs = set()
    for chunk in toks:
        for part in chunk.split(','):
            name = part.strip().split(' ')[0]
            if name:
                tabs.add(name)
    return tabs


def _columns(sql):
    # table.column references
    return set(re.findall(r'[a-z_]+_\d+\.[a-z_]+', _norm(sql)))


def _string_literals(sql):
    return set(re.findall(r"'([^']*)'", sql))


def _numbers(sql):
    return set(re.findall(r'\b\d+\b', _norm(sql)))


def _agg_structure(sql):
    s = _norm(sql)
    feats = set()
    for kw in ['count(', 'min(', 'max(', 'sum(', 'avg(', 'distinct',
               'group by', 'order by', 'limit', 'having', ' between ',
               ' not ', ' or ', ' <', ' >', '<=', '>=']:
        if kw in s:
            feats.add(kw.strip())
    return feats


def categorize(gold_sql, pred_sql):
    """
    Return an ordered list of the error categories that apply to this (gold, pred) pair.
    Ordered by priority so the dominant failure is listed first.
    """
    cats = []

    if _norm(pred_sql) == '':
        return ['empty_prediction']

    # 1. Wrong tables/columns (schema linking failure)
    if _tables(gold_sql) != _tables(pred_sql) or _columns(gold_sql) != _columns(pred_sql):
        cats.append('wrong_tables_or_columns')

    # 2. Wrong WHERE values (city/airline/date literals or numbers differ)
    if (_string_literals(gold_sql) != _string_literals(pred_sql) or
            _numbers(gold_sql) != _numbers(pred_sql)):
        cats.append('wrong_where_values')

    # 3. Wrong aggregation / query structure (COUNT/MIN/MAX/ORDER BY/DISTINCT ... differ)
    if _agg_structure(gold_sql) != _agg_structure(pred_sql):
        cats.append('wrong_aggregation_or_structure')

    if not cats:
        # Different SQL text but none of the structural buckets caught it.
        cats.append('other_structural_difference')
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_sql', default='results/t5_ft_experiment_dev.sql')
    ap.add_argument('--gold_sql', default='data/dev.sql')
    ap.add_argument('--nl', default='data/dev.nl')
    ap.add_argument('--gold_records', default='records/ground_truth_dev.pkl')
    ap.add_argument('--pred_records', default=None,
                    help='Optional precomputed predicted-record pkl; if omitted, queries are executed.')
    ap.add_argument('--out', default='results/error_analysis_dev.md')
    args = ap.parse_args()

    for p in [args.pred_sql, args.gold_sql, args.nl, args.gold_records]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing required input: {p}\n"
                f"Generate dev predictions first by running train_t5.py (which writes "
                f"{args.pred_sql}), then re-run this script."
            )

    with open(args.nl) as f:
        nls = [l.strip() for l in f]

    gold_qs, gold_recs, _ = load_queries_and_records(args.gold_sql, args.gold_records)
    pred_qs, pred_recs, pred_errs = load_queries_and_records(args.pred_sql, args.pred_records)

    n = min(len(gold_qs), len(pred_qs), len(nls))
    cat_counter = Counter()
    cat_examples = {}
    n_record_match = 0
    n_sql_match = 0

    for i in range(n):
        gq, pq = gold_qs[i], pred_qs[i]
        g_rec = set(gold_recs[i]) if i < len(gold_recs) else set()
        p_rec = set(pred_recs[i]) if i < len(pred_recs) else set()

        if gq.strip() == pq.strip():
            n_sql_match += 1
        record_match = (g_rec == p_rec)
        if record_match:
            n_record_match += 1
            continue  # only analyze errors

        for cat in categorize(gq, pq):
            cat_counter[cat] += 1
            # keep the first representative example per category
            if cat not in cat_examples:
                cat_examples[cat] = {
                    'nl': nls[i],
                    'gold': gq,
                    'pred': pq,
                    'gold_n_records': len(g_rec),
                    'pred_n_records': len(p_rec),
                    'exec_error': pred_errs[i] if i < len(pred_errs) and pred_errs[i] else '',
                }

    total_errors = n - n_record_match
    f1 = compute_record_F1(gold_recs[:n], pred_recs[:n])

    # ---- Write markdown report -------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    lines = []
    lines.append("# Dev-set Error Analysis (Q6)\n")
    lines.append(f"- Examples analyzed: **{n}**")
    lines.append(f"- Record F1: **{f1*100:.2f}**")
    lines.append(f"- Record Exact Match: **{n_record_match}/{n} = {100*n_record_match/n:.2f}%**")
    lines.append(f"- SQL Exact Match: **{n_sql_match}/{n} = {100*n_sql_match/n:.2f}%**")
    lines.append(f"- Total erroneous examples (record mismatch): **{total_errors}**\n")

    lines.append("## Error category counts\n")
    lines.append("| Category | Count | % of errors |")
    lines.append("|---|---:|---:|")
    for cat, cnt in cat_counter.most_common():
        pct = 100 * cnt / total_errors if total_errors else 0
        lines.append(f"| {cat} | {cnt} | {pct:.1f}% |")
    lines.append("\n*(An error can match multiple categories, so counts may sum to more than "
                 "the number of erroneous examples.)*\n")

    lines.append("## Representative examples (real, from dev predictions)\n")
    for cat, _ in cat_counter.most_common():
        ex = cat_examples.get(cat)
        if not ex:
            continue
        lines.append(f"### {cat}  (count = {cat_counter[cat]})\n")
        lines.append(f"- **NL input:** {ex['nl']}")
        lines.append(f"- **Ground-truth SQL:** `{ex['gold']}`")
        lines.append(f"- **Generated SQL:** `{ex['pred']}`")
        lines.append(f"- **# gold records:** {ex['gold_n_records']}  |  **# pred records:** {ex['pred_n_records']}")
        if ex['exec_error']:
            lines.append(f"- **Execution error:** {ex['exec_error']}")
        lines.append("")

    report = "\n".join(lines)
    with open(args.out, 'w') as f:
        f.write(report)

    print(report)
    print(f"\n[Saved error-analysis report to {args.out}]")


if __name__ == '__main__':
    main()
