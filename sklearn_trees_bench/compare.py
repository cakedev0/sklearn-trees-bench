import pandas as pd
import numpy as np
import json


def read_file(date_str, branch):
    with open(f'results/{date_str}_{branch}.jsonl') as f:
        return [json.loads(l) for l in f]


def process_result(r):
    row = {
        "model": r["model"],
        "n_repeats": r["n_repeats"],
        **r["data_params"],
        **r["model_params"],
    }
    times = [t['train_time'] for t in r['results']]
    row['t_low'] = np.quantile(times, 1/7)
    row['t_high'] = np.quantile(times, 6/7)
    row['t_med'] = np.median(times).round(2)
    return row


def compare_branches(date_str, branch_a, branch_b):
    results = read_file(date_str, branch_a)
    df_a = pd.DataFrame([process_result(r) for r in results])

    results = read_file(date_str, branch_b)
    df_b = pd.DataFrame([process_result(r) for r in results])

    df = pd.merge(
        df_a,
        df_b,
        on=[c for c in df_a.columns if not c.startswith('t_')],
        suffixes=(f'_a', f'_b')
    )

    gain = 1 - np.clip(df['t_low_a'] / df['t_high_b'], 1, None)
    loss = np.clip(df['t_low_b'] / df['t_high_a'], 1, None) - 1

    df['delta'] = (gain + loss).round(2)
    return (
        df.sort_values(by='delta')
        .drop(columns=[
            c for c in df.columns if c.startswith(('t_low', 't_high'))
        ])
    )
