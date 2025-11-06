import pandas as pd
import torch
import numpy as np
import re
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
from pathlib import Path


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

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

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(xt_train)
        loss = -mll(output, yt_train)
        loss.backward()
        optimizer.step()

    return model.eval(), likelihood.eval()


# ------------------------------------------------------------
# Pfad-Setup (plattformunabhängig)
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Results" / "DataArchiv"
RESULTS_DIR = BASE_DIR / "Results"

#for folder in [DATA_DIR, RESULTS_DIR]:
#    folder.mkdir(parents=True, exist_ok=True)

VEXT_FOLDER = DATA_DIR / "Vext_allcsv"
DFT_FILE = DATA_DIR / "DFT_Data_clean_06_10.csv"
SAVE_RESULTS = RESULTS_DIR / "GP_results_vext_chem.csv"
SAVE_METRICS = RESULTS_DIR / "GP_metrics_summary_vext_chem.csv"

normalize_feature = True
normalize_labels = True


# ------------------------------------------------------------
# Hauptprogramm
# ------------------------------------------------------------

if not DFT_FILE.exists():
    print(f"DFT-Datei nicht gefunden unter: {DFT_FILE}")
    print("Leere CSV wird angelegt.")
    pd.DataFrame(columns=["structure_name", "temperature_kelvin"]).to_csv(DFT_FILE, index=False)

dft_data = pd.read_csv(DFT_FILE)

if not VEXT_FOLDER.exists():
    print(f"Vext-Ordner {VEXT_FOLDER} existiert nicht, wird erstellt.")
    VEXT_FOLDER.mkdir(parents=True, exist_ok=True)

vext_files = [f for f in VEXT_FOLDER.glob("*.csv")]
if not vext_files:
    print(f"Keine Vext-Dateien gefunden unter {VEXT_FOLDER}.")
else:
    print(f"Gefundene Vext-Dateien: {[f.name for f in vext_files]}")

all_results = []
metrics_summary = []

for vext_file in vext_files:
    print(f"\n=== Starte Lauf mit {vext_file.name} ===")
    expV_data = pd.read_csv(vext_file)
    data = pd.merge(dft_data, expV_data, 'inner', on=["structure_name", "temperature_kelvin"])

    feature_columns = [col for col in data.columns if is_bin_column(col)]
    data = data[data.beladung_mol_per_kg > 0]

    for (T, P), subset in data.groupby(["temperature_kelvin", "pressure_bar"]):
        print(f"Zustand: {T} K | {P} bar")

        df = subset.copy()
        df["beladung_pro_vol"] = df["beladung_atoms"] / df["volume_kubAng"]

        df[feature_columns] = (
            df[feature_columns]
            .multiply(df["grid.dv"], axis=0)
            .div(df["volume_kubAng"], axis=0)
        )

        label = "beladung_pro_vol"
        X = df[feature_columns].values
        y = df[label].values

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        split_info = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
            x_train = torch.tensor(X[train_idx], dtype=torch.float64)
            y_train = torch.tensor(y[train_idx], dtype=torch.float64)
            x_test = torch.tensor(X[test_idx], dtype=torch.float64)
            y_test = torch.tensor(y[test_idx], dtype=torch.float64)

            test_df = df.iloc[test_idx].copy()
            test_df["fold"] = fold
            test_df["temperature_kelvin"] = T
            test_df["pressure_bar"] = P
            test_df["vext_version"] = vext_file.name

            # Feature-Normalisierung
            if normalize_feature:
                feature_transformer = MinMaxScaler()
                feature_transformer.fit(x_train)
                xt_train = torch.tensor(feature_transformer.transform(x_train), dtype=torch.float64)
                xt_test = torch.tensor(feature_transformer.transform(x_test), dtype=torch.float64)
            else:
                xt_train, xt_test = x_train, x_test

            # Label-Normalisierung
            if normalize_labels:
                label_transformer = MinMaxScaler()
                label_transformer.fit(y_train.unsqueeze(1))
                yt_train = torch.tensor(label_transformer.transform(y_train.unsqueeze(1)).flatten(), dtype=torch.float64)
                yt_test = torch.tensor(label_transformer.transform(y_test.unsqueeze(1)).flatten(), dtype=torch.float64)
            else:
                yt_train, yt_test = y_train, y_test

            # GP-Training
            model, likelihood = train_gp(xt_train, yt_train, training_iterations=200)

            # Vorhersage
            with torch.no_grad():
                prediction = model(xt_test)
                y_pred = label_transformer.inverse_transform(prediction.mean.unsqueeze(1)).squeeze()
                y_pred = np.where(y_pred > 0, y_pred, 0)

            # Ergebnisse speichern
            test_df[f"{label}_pred"] = y_pred
            test_df["abs_rel_deviation"] = np.abs((test_df[label] - y_pred) / test_df[label] * 100)

            # Metriken berechnen
            r2 = r2_score(test_df[label], y_pred)
            mae = mean_absolute_error(test_df[label], y_pred)
            median_dev = np.median(test_df["abs_rel_deviation"])

            metrics_summary.append({
                "vext_version": vext_file.name,
                "temperature_kelvin": T,
                "pressure_bar": P,
                "fold": fold,
                "r2_score": r2,
                "mae": mae,
                "median_abs_rel_dev": median_dev,
                "n_test_samples": len(test_df)
            })

            split_info.append(test_df)

        result = pd.concat(split_info, ignore_index=True)
        all_results.append(result)

# Ergebnisse abspeichern
final_results = pd.concat(all_results, ignore_index=True)
final_results.to_csv(SAVE_RESULTS, index=False)
metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv(SAVE_METRICS, index=False)

print(f"Detailergebnisse gespeichert unter: {SAVE_RESULTS}")
print(f"Metriken pro Fold und Zustand gespeichert unter: {SAVE_METRICS}")
