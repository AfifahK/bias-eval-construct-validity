#!/bin/bash
# Regenerate a single figure from committed results.
# Usage: bash regenerate_figure.sh
#
# Requires: pip install -r requirements.txt
# Reads from: results/tables/table_bias_rates.csv
# Writes to:  results/figures/fig1_bias_rates_by_method.png

set -e
cd "$(dirname "$0")/src/analysis"
python3 -c "
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'

br = pd.read_csv('../../results/tables/table_bias_rates.csv').sort_values('bias_rate')
fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(br)))
bars = ax.barh(br['method'], br['bias_rate'], color=colors, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Bias Detection Rate', fontsize=11)
ax.set_title(f'Bias Rates Across {len(br)} Evaluation Methods\n(n=293 non-refusal explanations)', fontsize=12)
ax.set_xlim(0, 1.1)
for bar, rate, n, flagged in zip(bars, br['bias_rate'], br['n'], br['flagged']):
    ax.text(rate + 0.01, bar.get_y() + bar.get_height()/2,
            f'{rate:.1%} ({flagged}/{n})', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('../../results/figures/fig1_bias_rates_by_method.png')
plt.close()
print('Regenerated: results/figures/fig1_bias_rates_by_method.png')
"
