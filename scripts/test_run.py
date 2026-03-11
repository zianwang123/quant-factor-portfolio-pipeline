"""Debug script to isolate segfault."""
import matplotlib
matplotlib.use("Agg")
import sys
import gc
import faulthandler
faulthandler.enable()

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.loader import DataPanel
from src.factors.registry import build_all_factors
from src.factors.validation import QuintileSorter

config = load_config(str(PROJECT_ROOT / "config" / "pipeline.yaml"), project_root=str(PROJECT_ROOT))

print("Loading data...", flush=True)
panel = DataPanel(config)
returns = panel.get_returns()
is_sp500 = panel.get_sp500_membership()
print(f"Returns shape: {returns.shape}", flush=True)

print("Building factors...", flush=True)
factors = build_all_factors(panel, config, include_extended=True, exclude=["Beta"])
print(f"Built {len(factors)} factors", flush=True)

# Free raw data
panel._raw = None
gc.collect()
print("Freed raw data", flush=True)

print("Running quintile sorts...", flush=True)
sorter = QuintileSorter(n_bins=5)
sort_results = sorter.sort_all_factors(factors, returns, is_sp500)
print(f"Completed {len(sort_results)} sorts", flush=True)

print("Saving QSpreads...", flush=True)
import pandas as pd
qspreads = pd.DataFrame({name: res["qspread"] for name, res in sort_results.items()})
tables_path = config.tables_path()
qspreads.to_csv(tables_path / "s1_factor_qspreads.csv")
print(f"QSpreads saved: {qspreads.shape}", flush=True)

print("Testing matplotlib...", flush=True)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 2, 3])
fig_path = config.figures_path("stage_1")
plt.savefig(fig_path / "test_plot.png")
plt.close()
print("Matplotlib OK", flush=True)

print("All done!", flush=True)
