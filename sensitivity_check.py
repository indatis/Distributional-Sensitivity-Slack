#!/usr/bin/env python3
"""Targeted Monte Carlo stability.
Runs only (M,N)=(100,100), theta=0.7,
three representative weight regimes, two independent seeds. Prefixes of each
R=1000 sequence supply R=250 and R=500 results without duplicate simulation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from simulation import run_simulation, p_rule

DISTS = ['gaussian', 'power_law_1_5', 'lognormal_baci']
ROWS = []
for seed in [42, 123]:
    print(f'Seed {seed}: generating 1000 paired replications (3 regimes) ...', flush=True)
    data = run_simulation(n_sources=100, n_targets=100,
                          p_edge=p_rule(100), theta=0.7,
                          replications=1000, seed=seed,
                          dist_names=DISTS)
    for R in [250, 500, 1000]:
        x = data[data['rep'] <= R]
        gauss = x.loc[x.dist == 'gaussian'].set_index('rep')
        for dist in DISTS:
            y = x.loc[x.dist == dist].set_index('rep')
            vals = y['min_slack_weighted'].to_numpy(float)
            frac = y['frac_slack_weighted_lt_one'].to_numpy(float)
            sd = vals.std(ddof=1)
            dmin = y['min_slack_weighted'] - gauss['min_slack_weighted']
            dpiv = y['frac_slack_weighted_lt_one'] - gauss['frac_slack_weighted_lt_one']
            ROWS.append(dict(seed=seed, R=R, distribution=dist,
                mean_min_slack=vals.mean(),
                mc_95ci_low=vals.mean()-1.96*sd/np.sqrt(len(vals)),
                mc_95ci_high=vals.mean()+1.96*sd/np.sqrt(len(vals)),
                pivotal_sources_pct=100*frac.mean(),
                paired_mean_diff_min_slack=dmin.mean(),
                share_lower_min_slack=(dmin < 0).mean(),
                share_higher_pivotal=(dpiv > 0).mean()))

result = pd.DataFrame(ROWS)
out = Path(__file__).resolve().parent / 'R3C5_seed_replication_sensitivity.csv'
result.to_csv(out, index=False)
with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                       'display.width', 180, 'display.float_format', lambda v: f'{v:.4f}'):
    print(result[['seed','R','distribution','mean_min_slack',
                  'pivotal_sources_pct','paired_mean_diff_min_slack']].to_string(index=False))
print(f'\nSaved: {out}')
