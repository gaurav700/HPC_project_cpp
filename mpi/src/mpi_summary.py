python3 - <<'PY'
import pandas as pd
df = pd.read_csv("results/mpi_repeats.csv")
agg = df.groupby(['n','processes']).time_s.agg(['mean','std','count']).reset_index()
agg.to_csv("results/day2_mpi_summary.csv", index=False)
print(agg)
print("\n✅ Saved results/mpi_summary.csv")
PY
