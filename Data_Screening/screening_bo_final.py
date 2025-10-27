import pandas as pd
import torch
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
from torch.distributions import Normal

# --- Settings ---
DFT_PATH = "/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/DFT_Data_clean_64grid_kond.csv"
VEXT_FILE = "/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/Vext_allcsv/Vext_allTEMP_64grid_20b.csv"  # <<< HIER ändern
LOG_PATH = "/Users/danielbock/MASTERTHESIS/MASTA/Results/BO_log_SINGLE_20b.csv"

# BO Parameter
temperatures = [300., 350., 400., 450., 500.]
pressures = [1., 8.07142857, 15.14285714, 22.21428571, 29.28571429,
             36.35714286, 43.42857143, 50.5, 57.57142857, 64.64285714,
             71.71428571, 78.78571429, 85.85714286, 92.92857143, 100.]

# --- GP Model ---
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)

def train_gp(x, y, iters=200):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x, y, likelihood)
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(iters):
        optimizer.zero_grad()
        loss = -mll(model(x), y)
        loss.backward()
        optimizer.step()
    return model.eval(), likelihood.eval()

def af_log_ei(mean, var, best):
    std = torch.sqrt(torch.clamp(var, min=1e-9))
    z = (mean - best) / std
    normal = Normal(0, 1)
    ei = std * (z * normal.cdf(z) + normal.log_prob(z).exp())
    return torch.log(torch.clamp(ei, min=1e-12))

# --- Load data ---
dft_data = pd.read_csv(DFT_PATH)
expV_data = pd.read_csv(VEXT_FILE)
expV_data = expV_data[expV_data["temperature_kelvin"].isin(temperatures)]

# Log vorbereiten
cols = ["vext_file", "temperature", "pressure", "best_value", "true_best_value", "iterations", "n_data"]
if not os.path.exists(LOG_PATH):
    pd.DataFrame(columns=cols).to_csv(LOG_PATH, index=False)

# --- BO SINGLE ---
for T in temperatures:
    for P in pressures:
        data = pd.merge(dft_data, expV_data, on=["structure_name", "temperature_kelvin"], how="inner")
        data = data[np.isclose(data["temperature_kelvin"], T) & np.isclose(data["pressure_bar"], P)]
        if data.empty:
            continue

        feature_cols = [c for c in data.columns if "bin_" in c]
        if not feature_cols:
            continue
        print(data[feature_cols])
        
        data["beladung_pro_vol"] = data["beladung_atoms"] / data["volume_kubAng"]
        data[feature_cols] = data[feature_cols].multiply(data["grid.dv"], axis=0).div(data["volume_kubAng"], axis=0)

        candidates = data.copy()
        selected = candidates.nsmallest(1, "beladung_pro_vol")

        candidates = candidates.drop(selected.index)
        best_hist = [selected["beladung_pro_vol"].max()]
        true_best = data["beladung_pro_vol"].max()

        for _ in range(50):
            fx = MinMaxScaler()
            fy = MinMaxScaler()
            train_x = torch.tensor(fx.fit_transform(selected[feature_cols]), dtype=torch.float32)
            train_y = torch.tensor(fy.fit_transform(selected[["beladung_pro_vol"]]).flatten(), dtype=torch.float32)
            test_x = torch.tensor(fx.transform(candidates[feature_cols]), dtype=torch.float32)

            model, likelihood = train_gp(train_x, train_y)
            with torch.no_grad():
                pred = model(test_x)
                ei = af_log_ei(pred.mean, pred.variance, train_y.max())
            idx = torch.argmax(ei).item()

            selected = pd.concat([selected, candidates.iloc[[idx]]])
            candidates = candidates.drop(candidates.index[idx])
            best_hist.append(selected["beladung_pro_vol"].max())

        pd.DataFrame([{
            "vext_file": os.path.basename(VEXT_FILE),
            "temperature": T,
            "pressure": P,
            "best_value": best_hist[-1],
            "true_best_value": true_best,
            "iterations": len(best_hist),
            "n_data": len(data),
        }]).to_csv(LOG_PATH, mode="a", header=False, index=False)

print("✅ Fertig – Ergebnisse gespeichert in:", LOG_PATH)
