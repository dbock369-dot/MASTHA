import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.metrics import r2_score
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
import altair as alt
import matplotlib.pyplot as plt
import re
from scipy.stats import norm
from torch.distributions import Normal

def af_log_expIm(mean, var, best_f, xi=0.01):
    """Logarithmic Expected Improvement acquisition function."""

    std = torch.sqrt(var)
    std_safe = torch.clamp(std, min=1e-9)  # Avoid division by zero
    z = (mean - best_f - xi) / std_safe
    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    cdf = normal.cdf(z)
    pdf = torch.exp(normal.log_prob(z))

    ei = std * (z * cdf + pdf)

    ei_safe = torch.clamp(ei, min=1e-9)  # Avoid log(0)
    log_ei = torch.log(ei_safe)
    return log_ei

def is_bin_column(col) -> bool:
    # numerische Spaltennamen erlauben
    if isinstance(col, (int, np.integer)):
        return True

    s = str(col)

    # rein numerischer Spaltenname: '0', '1', ...
    if s.isdigit():
        return True

    # bin_X oder bin_X_high / bin_X_low
    if re.fullmatch(r"bin_\d+(_high|_low)?", s):
        return True
    if re.fullmatch(r"bin_\d+", s): # 'bin_0', 'bin_1', ...
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

    losses = []

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(xt_train)
        loss = -mll(output, yt_train)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return model.eval(), likelihood.eval(), losses

# dft Daten - beladungen, grid etc
dft_data1 = pd.read_csv('/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/DFT_Data_clean_06_10.csv')
dft_data2 = pd.read_csv("/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/dft_fckin_clean_kond_64grid.csv")
dft_data_all =  pd.concat([dft_data1, dft_data2], ignore_index=True)

# Feature Daten - bins zu Vext, Vext+chem_res etc
expV_data = pd.read_csv("/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/Vext_chem_res_allTEMP_pressure_20b_exp.csv")
#expV_data = pd.read_csv("/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/Vext_allcsv/Vext_allTEMP_64grid_20b.csv")

# Chem_res_bulk explizit - für additional feature 
chem_res = pd.read_csv("/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/bulk_potentials.csv")

# Kombi aus obigem 
data = pd.merge(dft_data_all, expV_data, 'inner', on=["structure_name", "temperature_kelvin", "pressure_bar"])
data = pd.merge(data, chem_res, 'inner', on=["structure_name", "temperature_kelvin", "pressure_bar"])
#feature_columns = [col for col in data.columns if is_bin_column(col)]
data = data[data.beladung_mol_per_kg > 0]

# wc(p, T) -> high
data_high = data[(data.temperature_kelvin == 300) & (data.pressure_bar == 1)]
data_high = data_high.drop_duplicates(subset=["structure_name", "temperature_kelvin", "pressure_bar"])
feature_columns_high = [col for col in data_high.columns if is_bin_column(col)]

# wc(p, T) -> low
data_low = data[(data.temperature_kelvin == 325) & (data.pressure_bar == 1)]
data_low = data_low.drop_duplicates(subset=["structure_name", "temperature_kelvin", "pressure_bar"])
feature_columns_low = [col for col in data_low.columns if is_bin_column(col)]

#duplicates = data[data.duplicated(subset=["structure_name", "temperature_kelvin", "pressure_bar"], keep=False)]
#print(duplicates[["structure_name", "temperature_kelvin", "pressure_bar"]])

print(data_high.shape == data_low.shape)
print(f"Data_HIGH: {data_high.shape}")
print(f"Data_LOW: {data_low.shape}")

add_features = True

data_high["beladung_pro_vol"] = (
    data_high["beladung_atoms"]
    #.div(data_high["density_bulk"], axis=0)
    .div(data_high["volume_kubAng"], axis=0)
)

data_low["beladung_pro_vol"] = (
    data_low["beladung_atoms"]
    #.div(data_low["density_bulk"], axis=0)
    .div(data_low["volume_kubAng"], axis=0)
)

data_high[feature_columns_high] = (
    data_high[feature_columns_high]
    .multiply(data_high["grid.dv"], axis=0)
    .div(data_high["volume_kubAng"], axis=0)
)

data_low[feature_columns_low] = (
    data_low[feature_columns_low]
    .multiply(data_low["grid.dv"], axis=0)
    .div(data_low["volume_kubAng"], axis=0)
)

additional_features = ["delta_p", "delta_T"]
additional_features12 = ["chem_potential_bulk_high", "chem_potential_bulk_low", "pressure_bar_high", "pressure_bar_low", "temperature_kelvin_high", "temperature_kelvin_low"]

merged = pd.merge(
    #data_high[["structure_name", "beladung_pro_vol"]],
    #data_low[["structure_name", "beladung_pro_vol"]],
    data_high,#[cols],
    data_low,#[cols],
    on="structure_name",
    suffixes=("_high", "_low")
)

# wc aus den beiden Zuständen high/low
merged["working_capacity"] = (merged["beladung_pro_vol_high"] - merged["beladung_pro_vol_low"]).abs()
merged["delta_T"] = (merged["temperature_kelvin_high"] - merged["temperature_kelvin_low"]).abs()
merged["delta_p"] = (merged["pressure_bar_high"] - merged["pressure_bar_low"]).abs()
feature_columns = [col for col in merged.columns if is_bin_column(col)]

if add_features:
    feature_columns += additional_features

merged["working_capacity"] = pd.to_numeric(merged["working_capacity"], errors="coerce")
print(merged["working_capacity"])

normalize_feature = True
normalize_labels = True

kf = KFold(n_splits=10, shuffle=True, random_state=42)

label = "working_capacity"
#X = data[feature_columns].values 
X = merged[feature_columns].values
#y = data[label].values 
y = merged[label].values 

ids = data.index.values

split_info = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    x_train = torch.tensor(X[train_idx], dtype=torch.float64)
    y_train = torch.tensor(y[train_idx], dtype=torch.float64)
    x_test = torch.tensor(X[test_idx], dtype=torch.float64)
    y_test = torch.tensor(y[test_idx], dtype=torch.float64)

    train_ids = ids[train_idx]
    test_ids = ids[test_idx]

    #test_df = data.iloc[test_idx].copy()
    test_df = merged.iloc[test_idx].copy()
    test_df["fold"] = fold

    if normalize_feature:
        feature_transformer = MinMaxScaler()
        feature_transformer.fit(x_train)
        xt_train = torch.tensor(feature_transformer.transform(x_train), dtype=torch.float64)
        xt_test = torch.tensor(feature_transformer.transform(x_test), dtype=torch.float64) #*2
    else:
        xt_train = x_train
        xt_test = x_test

    # Label-Normalisierung
    if normalize_labels:
        label_transformer = MinMaxScaler()  # oder StandardScaler()
        label_transformer.fit(y_train.unsqueeze(1))
        yt_train = torch.tensor(label_transformer.transform(y_train.unsqueeze(1)).flatten(), dtype=torch.float64)
        yt_test = torch.tensor(label_transformer.transform(y_test.unsqueeze(1)).flatten(), dtype=torch.float64)
    else:
        yt_train = y_train
        yt_test = y_test

    # Training
    model, likelihood, losses = train_gp(xt_train, yt_train, training_iterations=200)

    # Prediction
    with torch.no_grad():
        prediction = model(xt_test)
        inverse_transformed_prediction = label_transformer.inverse_transform(
            prediction.mean.unsqueeze(1)
        ).squeeze()
        inverse_transformed_prediction = np.where(
            inverse_transformed_prediction > 0, inverse_transformed_prediction, 0
        )

    # Ergebnisse
    test_df[f"{label}_pred"] = inverse_transformed_prediction
    test_df["abs_rel_deviation"] = np.abs(
        (test_df[label] - test_df[f"{label}_pred"]) / test_df[label] * 100
    )

    split_info.append(test_df)

results = pd.concat(split_info, ignore_index=True)
#results
# results.to_csv("GP_results_beladung_pro_vol_400K_0.1bar_customFeatures.csv", index=False)

candidates = merged.copy() # zunächst gefilteret, später alle Daten

patience = 10

n_initial = 1 # Anzahl der initialen Trainingspunkte
initial_indices = candidates.nsmallest(n_initial, label).index # hier geht auch random

print(f"Initial training points:")
for idx in initial_indices:
    print(f"  Index {idx}, Structure {candidates.loc[idx, 'structure_name']}, {label}: {candidates.loc[idx, label]:.4f}")

# Transfer from candidates to selection
selected = candidates.loc[initial_indices]
candidates = candidates.drop(initial_indices)
best = [selected[label].max()]

for i in range(100):
    if len(best) >= patience:
        if len(np.unique(best[-patience:])) == 1:
            print(f"Early stopping at iteration {i} due to no improvement in the last {patience} iterations.")
            break
    
    feature_transoformer = MinMaxScaler()
    label_transformer = MinMaxScaler()

    train_x = torch.tensor(feature_transoformer.fit_transform(selected[feature_columns].values))
    train_y = torch.tensor(label_transformer.fit_transform(selected[[label]].values)).flatten()

    test_x = torch.tensor(feature_transoformer.transform(candidates[feature_columns].values))

    model, likelihood, _ = train_gp(train_x, train_y, 250)
    with torch.no_grad():
        prediction = model(test_x)
        mean, var = prediction.mean, prediction.variance
    
    best_f = train_y.max()

    log_ei = af_log_expIm(mean, var, best_f, 0.01 * best_f)

    # Select the candidate with the highest acquisition value
    index = torch.argmax(log_ei).item()
    best.append(selected[label].max())
    print(f"Iteration: {i}, Current Best: {selected[label].max():.2e}")
    selected = pd.concat([selected, candidates.iloc[[index]]])
    canidates = candidates.drop(candidates.index[index])

print(f"Best Value after {len(best)} iterations: {best[-1]}")

mean_np = mean.detach().cpu().numpy().flatten()
var_np = var.detach().cpu().numpy().flatten()
std_np = np.sqrt(var_np)
ei_np = torch.exp(log_ei).detach().cpu().numpy().flatten()

# --- Probability of Improvement ---
best_f = train_y.max().item()
z = (mean_np - best_f) / std_np
pi_np = norm.cdf(z)

