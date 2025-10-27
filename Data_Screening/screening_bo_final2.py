import os
import glob
import re
import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.preprocessing import MinMaxScaler
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.distributions import Normal

# Pfade
DFT_PATH  = "/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/DFT_Data_clean_64grid_kond.csv"
VEXT_DIR  = "/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/Vext_allcsv/"
LOG_PATH  = "/Users/danielbock/MASTERTHESIS/MASTA/Results/BO_log_allVersion_frfrtrue.csv"

# BO-Settings
TRAIN_ITERS = 250
PATIENCE    = 10
N_INIT      = 1   # Start mit kleinstem Label
RNG         = 42

# ---------------- GP ----------------
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp(xt_train, yt_train, training_iterations):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model      = ExactGPModel(xt_train, yt_train, likelihood)
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    mll       = ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(training_iterations):
        optimizer.zero_grad()
        output = model(xt_train)
        loss   = -mll(output, yt_train)
        loss.backward()
        optimizer.step()
    return model.eval(), likelihood.eval()

def af_log_expIm(mean, var, best_f, xi):
    var  = torch.nan_to_num(var, nan=0.0, posinf=1e3, neginf=0.0)
    std  = torch.sqrt(var + 1e-9)
    z    = (mean - best_f - xi) / std
    z    = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    cdf    = normal.cdf(z)
    pdf    = torch.exp(normal.log_prob(z))
    ei     = std * (z * cdf + pdf)
    ei     = torch.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    log_ei = torch.log(ei + 1e-12)
    # NaNs in der AF „schlecht“ setzen, damit sie nicht ausgewählt werden
    log_ei = torch.nan_to_num(log_ei, nan=-1e9)
    return log_ei

# ------------- Utils ---------------
BIN_REGEX = re.compile(r"^bin_(\d+)$")

def get_bin_columns(df):
    """Nur echte bin_* Spalten, sortiert nach Index."""
    pairs = []
    for c in df.columns:
        m = BIN_REGEX.match(c)
        if m:
            pairs.append((int(m.group(1)), c))
    pairs.sort(key=lambda t: t[0])
    return [c for _, c in pairs]

def merge_one_to_one_for_TP(dft_data, expV_data, T, P):
    """
    Filtere DFT auf (T,P), VEXT auf T, säubere Keys, dedupliziere je Schlüssel
    und merge one-to-one nur mit bin_* aus VEXT.
    """
    # 1) Vorfiltern
    dft_tp = dft_data[
        np.isclose(dft_data["temperature_kelvin"], T) &
        np.isclose(dft_data["pressure_bar"], P)
    ].copy()
    vext_t = expV_data[
        np.isclose(expV_data["temperature_kelvin"], T)
    ].copy()

    if dft_tp.empty or vext_t.empty:
        return pd.DataFrame(), []

    # 2) Keys säubern
    dft_tp["structure_name"] = dft_tp["structure_name"].astype(str).str.strip()
    vext_t["structure_name"] = vext_t["structure_name"].astype(str).str.strip()

    # 3) pro Schlüssel eindeutig machen
    dft_tp = dft_tp.sort_values("structure_name").drop_duplicates(
        subset=["structure_name", "temperature_kelvin", "pressure_bar"],
        keep="first"
    )
    vext_t = vext_t.sort_values("structure_name").drop_duplicates(
        subset=["structure_name", "temperature_kelvin"],
        keep="first"
    )

    # 4) nur bin_* aus VEXT
    bin_cols = get_bin_columns(vext_t)
    vext_trim = vext_t[["structure_name", "temperature_kelvin"] + bin_cols].copy()

    # 5) Merge (one-to-one validieren)
    data = pd.merge(
        dft_tp, vext_trim,
        how="inner",
        on=["structure_name", "temperature_kelvin"],
        validate="one_to_one"
    )
    return data, bin_cols

# ------------- Main ----------------
def main():
    dft_data = pd.read_csv(DFT_PATH)
    temperatures = sorted(dft_data["temperature_kelvin"].dropna().unique())
    pressures    = sorted(dft_data["pressure_bar"].dropna().unique())

    log_cols = ["vext_file", "temperature", "pressure", "best_value", "true_best_value", "iterations", "n_data"]
    if not os.path.exists(LOG_PATH):
        pd.DataFrame(columns=log_cols).to_csv(LOG_PATH, index=False)

    for vext_file in glob.glob(os.path.join(VEXT_DIR, "*.csv")):
        expV_data = pd.read_csv(vext_file)

        for T in temperatures:
            for P in pressures:
                # Merge nur mit DFT(T,P) × VEXT(T)
                data, bin_cols = merge_one_to_one_for_TP(dft_data, expV_data, T, P)
                if data.empty or len(bin_cols) == 0:
                    continue

                # Erwartete Anzahl aus DFT(T,P): unique Strukturen
                expected = data[["structure_name"]].drop_duplicates().shape[0]  # i.d.R. 245
                # Falls du hart 245 prüfen willst, ersetze expected = 245

                # Label-Filter
                data = data[data["beladung_mol_per_kg"] > 0]

                # Normierung & Label
                data = data.copy()
                data["beladung_pro_vol"] = data["beladung_atoms"] / data["volume_kubAng"]
                feat = data[bin_cols].multiply(data["grid.dv"], axis=0).div(data["volume_kubAng"], axis=0)
                data[bin_cols] = feat

                n_data = len(data)
                if n_data != expected:
                    # Nur Warnung, BO läuft wie gewünscht weiter
                    print(f"[WARN] {os.path.basename(vext_file)} | T={T}, P={P}: n_data={n_data}, expected≈{expected}")

                if n_data == 0:
                    continue

                label = "beladung_pro_vol"
                true_best = data[label].max()

                candidates = data.copy()
                # Start mit kleinstem Label
                initial_indices = candidates.nsmallest(N_INIT, label).index
                selected  = candidates.loc[initial_indices].copy()
                candidates = candidates.drop(initial_indices).copy()
                best_hist = [selected[label].max()]

                # BO-Loop
                for _ in range(100):
                    if len(best_hist) >= PATIENCE and len(np.unique(best_hist[-PATIENCE:])) == 1:
                        break
                    if candidates.empty:
                        break

                    fx = MinMaxScaler()
                    fy = MinMaxScaler()

                    train_x = torch.tensor(fx.fit_transform(selected[bin_cols].values), dtype=torch.float32)
                    train_y = torch.tensor(fy.fit_transform(selected[[label]].values), dtype=torch.float32).flatten()
                    test_x  = torch.tensor(fx.transform(candidates[bin_cols].values), dtype=torch.float32)

                    model, likelihood = train_gp(train_x, train_y, TRAIN_ITERS)
                    with torch.no_grad():
                        pred = model(test_x)
                        mean, var = pred.mean, pred.variance

                    best_f = train_y.max()
                    log_ei = af_log_expIm(mean, var, best_f, xi=0.01 * best_f)
                    idx_local = torch.argmax(log_ei).item()
                    if not (0 <= idx_local < len(candidates)):
                        break

                    best_hist.append(selected[label].max())
                    next_row = candidates.iloc[[idx_local]].copy()
                    selected = pd.concat([selected, next_row])
                    candidates = candidates.drop(candidates.index[idx_local])

                # Logging wie bei dir
                pd.DataFrame([{
                    "vext_file": os.path.basename(vext_file),
                    "temperature": T,
                    "pressure": P,
                    "best_value": best_hist[-1],
                    "true_best_value": true_best,
                    "iterations": len(best_hist),
                    "n_data": n_data
                }]).to_csv(LOG_PATH, mode="a", header=False, index=False)

if __name__ == "__main__":
    main()
