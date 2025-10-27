import pandas as pd
import torch
import numpy as np
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
from torch.distributions import Normal
import re
from scipy.stats import norm

# --- Utility functions ---
def is_bin_column(col) -> bool:
    if isinstance(col, (int, np.integer)):
        return True
    s = str(col)
    if s.isdigit():
        return True
    if re.fullmatch(r"bin_\d+", s):
        return True
    if re.fullmatch(r"Bin_\d+", s):
        return True
    return False


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp(xt_train, yt_train, training_iterations=100):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(xt_train, yt_train, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(xt_train)
        loss = -mll(output, yt_train)
        loss.backward()
        optimizer.step()

    return model.eval(), likelihood.eval()


def af_log_expIm(mean, var, best_f, xi=0.01):
    std = torch.sqrt(var)
    std_safe = torch.clamp(std, min=1e-9)
    z = (mean - best_f - xi) / std_safe
    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    cdf = normal.cdf(z)
    pdf = torch.exp(normal.log_prob(z))
    ei = std * (z * cdf + pdf)
    ei_safe = torch.clamp(ei, min=1e-9)

    print("=== DEBUG EI ===")
    print("mean NaN:", torch.isnan(mean).sum().item())
    print("var NaN:", torch.isnan(var).sum().item())
    print("var min:", var.min().item(), "max:", var.max().item())
    print("best_f:", best_f.item())

    log_ei = torch.log(ei_safe)
    return log_ei


# --- Settings ---
DFT_PATH = "/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/dft_fckin_clean_kond_64grid.csv"
#DFT_PATH = "DataArchiv/DFT_Data_clean_64grid_kond.csv"
VEXT_DIR = "/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/Vext_allcsv/"
LOG_PATH = "/Users/danielbock/MASTERTHESIS/MASTA/Results/BO_log_allVersion_frfrFINAL.csv"

# --- Load DFT data once ---
dft_data = pd.read_csv(DFT_PATH)

#temperatures = [300., 350., 400., 450., 500.]
#pressures = [  1.        ,   8.07142857,  15.14285714,  22.21428571,
        29.28571429,  36.35714286,  43.42857143,  50.5       ,
        57.57142857,  64.64285714,  71.71428571,  78.78571429,
        85.85714286,  92.92857143, 100.        ]
            
temperatures = sorted(dft_data["temperature_kelvin"].dropna().unique())
pressures = sorted(dft_data["pressure_bar"].dropna().unique())



# --- Prepare log file ---
log_cols = ["vext_file", "temperature", "pressure", "best_value", "true_best_value", "iterations", "n_data"]
if not os.path.exists(LOG_PATH):
    pd.DataFrame(columns=log_cols).to_csv(LOG_PATH, index=False)

# --- Loop over all Vext files, temperatures, and pressures ---
for vext_file in glob.glob(os.path.join(VEXT_DIR, "*.csv")):
    vext_name = os.path.basename(vext_file)
    expV_data = pd.read_csv(vext_file)
    expV_data = expV_data[expV_data["temperature_kelvin"].isin(temperatures)].copy()
    print(f"\n=== Processing {vext_name} ===")

    for T in temperatures:
        for P in pressures:
            #print(f"\n--- T={T} K, P={P} bar ---")

            # --- Merge and filter ---
            data = pd.merge(dft_data, expV_data, "inner", on=["structure_name", "temperature_kelvin"])
            
            feature_columns = [col for col in data.columns if is_bin_column(col)]
            data = data[data.beladung_mol_per_kg > 0]
            data = data[np.isclose(data.temperature_kelvin, T) & np.isclose(data.pressure_bar, P)]
            print(data)
            if data.empty:
                print(f"No data for T={T}, P={P}, skipping.")
                continue
            
            if len(feature_columns) == 0:
                print(f"No feature columns found in data for T={T}, P={P}, skipping.")
                continue

            data["beladung_pro_vol"] = data["beladung_atoms"] / data["volume_kubAng"]
            data[feature_columns] = (
                data[feature_columns]
                .multiply(data["grid.dv"], axis=0)
                .div(data["volume_kubAng"], axis=0)
            )
            label = "beladung_pro_vol"
            true_best = data[label].max()

            candidates = data.copy()
            patience = 10
            n_initial = 1
            initial_indices = candidates.nsmallest(n_initial, label).index

            selected = candidates.loc[initial_indices]
            candidates = candidates.drop(initial_indices)
            best = [selected[label].max()]
            #print(candidates)
            #print(selected)
            #print("Feature dtypes in selected:")
            #print(selected[feature_columns].dtypes.value_counts())

            # --- BO loop ---
            for i in range(100):
                if len(best) >= patience and len(np.unique(best[-patience:])) == 1:
                    print(f"Early stopping at iteration {i}.")
                    break

                feature_transformer = MinMaxScaler()
                label_transformer = MinMaxScaler()

                train_x = torch.tensor(feature_transformer.fit_transform(selected[feature_columns].values), dtype=torch.float32)
                train_y = torch.tensor(label_transformer.fit_transform(selected[[label]].values), dtype=torch.float32).flatten()
                test_x = torch.tensor(feature_transformer.transform(candidates[feature_columns].values), dtype=torch.float32)

                model, likelihood = train_gp(train_x, train_y, 250)
                with torch.no_grad():
                    prediction = model(test_x)
                    mean, var = prediction.mean, prediction.variance

                best_f = train_y.max()
                log_ei = af_log_expIm(mean, var, best_f, 0.01 * best_f)

                index = torch.argmax(log_ei).item()
                best.append(selected[label].max())

                selected = pd.concat([selected, candidates.iloc[[index]]])
                candidates = candidates.drop(candidates.index[index])

                print(f"Iter {i:02d} | Current best: {selected[label].max():.3e}")

            # --- Log results ---
            log_entry = pd.DataFrame([{
                "vext_file": vext_name,
                "temperature": T,
                "pressure": P,
                "best_value": best[-1],
                "true_best_value": true_best,
                "iterations": len(best),
                "n_data": len(data)
            }])
            log_entry.to_csv(LOG_PATH, mode="a", header=False, index=False)

            print(f"Finished {vext_name} | T={T}, P={P}, Best={best[-1]:.3e}")

print("\n=== Screening completed. Log saved at ===")
print(LOG_PATH)
