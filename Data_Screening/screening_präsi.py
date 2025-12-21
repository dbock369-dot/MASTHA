import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
import re
from sklearn.cluster import KMeans
from torch.distributions import Normal
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
TEMP = [298, 300, 350, 400, 450, 500]
PRESSURES = [0.1, 1.0, 100.0]   # low / mid / high pressure regime
LABEL = "beladung_pro_vol"

N_SPLITS = 10
N_BOOT = 2000
BOOT_SEED = 42

# BO stability settings
N_BO_RUNS = 20          # e.g. 10–30 for stability
BO_MAX_ITER = 100
BO_PATIENCE = 10

INIT_STRATEGY = "kmeans"    # "kmeans" or "random"
ACQ = "ucb"                 # "ei" or "ucb"

# EI params
XI0, XI_MIN = 0.05, 0.005
# UCB params 
BETA0, BETA_MIN = 3.0, 1.0

# output files
POOLED_LOG_FILE = Path("/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/gp_pooled_r2_log_nodens_ucb_BOLT.csv")
#BO_LOG_FILE     = Path("/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/bo_results5.csv")
RESULTS_FILE = Path("/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/bo_random_results_nodens_ucb_BOLT.csv")
HISTORY_FILE = Path("/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/bo_random_history_nodens_ucb_BOLT.csv")
# ============================================================
# HELPERS
# ============================================================
def is_bin_column(col) -> bool:
    if isinstance(col, (int, np.integer)):
        return True
    s = str(col)
    if s.isdigit():
        return True
    if re.fullmatch(r"bin_\d+", s):
        return True
    return False

def run_one_bo_or_random(
        *,
        method,                   # "bo" or "random"
        run_id,
        run_seed,
        data,
        feature_columns,
        label,
        max_iter,
        patience,
        init_strategy,
        acq,                      # "ei" or "ucb" (only used for method="bo")
        XI0, XI_MIN,
        BETA0, BETA_MIN,
        feature_scaler,
        label_scaler,
        initial_indices
    ):
    """Run one BO or Random trajectory starting from the same initial set."""
    
    candidates = data.copy()

    # fixed initial set
    selected = candidates.loc[initial_indices].copy()
    candidates = candidates.drop(initial_indices).copy()

    # --- FIX: numeric state, logged consistently everywhere ---
    t_state = float(data["temperature_kelvin"].iloc[0])
    p_state = float(data["pressure_bar"].iloc[0])

    best = [float(selected[label].max())]

    history = [{
        "run_id": run_id,
        "seed": run_seed,
        "iter": 0,
        "best_so_far": best[0],
        "temperature_kelvin": t_state,
        "pressure_bar": p_state,
        "method": method
    }]

    acq_param_last = np.nan
    rng = np.random.default_rng(run_seed)

    for i in range(max_iter):

        # early stop if plateau
        if len(best) >= patience and len(np.unique(np.round(best[-patience:], 12))) == 1:
            break

        if len(candidates) == 0:
            break

        # --------------------------------------------------
        # PICK NEXT CANDIDATE
        # --------------------------------------------------
        if method == "random":
            pick = rng.integers(0, len(candidates))

        elif method == "bo":
            # build train/test (global scaling)
            train_x = torch.tensor(
                feature_scaler.transform(selected[feature_columns].values),
                dtype=torch.float32
            )
            train_y = torch.tensor(
                label_scaler.transform(selected[[label]].values),
                dtype=torch.float32
            ).flatten()

            test_x = torch.tensor(
                feature_scaler.transform(candidates[feature_columns].values),
                dtype=torch.float32
            )

            model, likelihood, _ = train_gp(train_x, train_y, 250)

            with torch.no_grad():
                pred = model(test_x)
                mean, var = pred.mean, pred.variance

            best_f = train_y.max()

            if acq == "ei":
                xi = max(XI_MIN, XI0 * (0.95 ** i))
                score = log_expected_improvement(mean, var, best_f, xi)
                acq_param_last = float(xi)
            else:
                beta = max(BETA_MIN, BETA0 * (0.97 ** i))
                score = ucb(mean, var, beta)
                acq_param_last = float(beta)

            pick = int(torch.argmax(score).item())

        else:
            raise ValueError("method must be 'bo' or 'random'")

        # --------------------------------------------------
        # UPDATE SETS
        # --------------------------------------------------
        selected = pd.concat([selected, candidates.iloc[[pick]]], ignore_index=False)
        candidates = candidates.drop(candidates.index[pick])

        best.append(float(selected[label].max()))

        history.append({
            "run_id": run_id,
            "seed": run_seed,
            "iter": len(best) - 1,
            "best_so_far": best[-1],
            "temperature_kelvin": t_state,
            "pressure_bar": p_state,
            "method": method
        })

    return best, history, acq_param_last

def pooled_r2_with_bootstrap_ci(results, label, pred_col=None, n_boot=2000, seed=42, ci=95):
    if pred_col is None:
        pred_col = f"{label}_pred"

    df = results[[label, pred_col]].dropna()
    y_true = df[label].to_numpy()
    y_pred = df[pred_col].to_numpy()

    pooled_r2 = r2_score(y_true, y_pred)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = r2_score(y_true[idx], y_pred[idx])

    boot_std = boot.std(ddof=1)
    alpha = (100 - ci) / 2
    ci_low, ci_high = np.percentile(boot, [alpha, 100 - alpha])

    return pooled_r2, boot_std, (ci_low, ci_high), boot

def initial_indices_kmeans(df, feature_columns, n_initial, random_state=0):
    X = df[feature_columns].values
    if len(df) <= n_initial:
        return df.index

    km = KMeans(n_clusters=n_initial, n_init=1, random_state=random_state)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    picked = []
    for k in range(n_initial):
        members = np.where(labels == k)[0]
        if len(members) == 0:
            continue
        d = np.linalg.norm(X[members] - centers[k], axis=1)
        picked.append(df.index[members[np.argmin(d)]])

    if len(picked) < n_initial:
        rest = df.index.difference(picked)
        extra = np.random.default_rng(random_state).choice(
            rest, size=(n_initial - len(picked)), replace=False
        )
        picked.extend(list(extra))

    return pd.Index(picked)

# ============================================================
# GP MODEL
# ============================================================
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp(xt_train, yt_train, training_iterations=200):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(xt_train, yt_train, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    for _ in range(training_iterations):
        optimizer.zero_grad()
        output = model(xt_train)
        loss = -mll(output, yt_train)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return model.eval(), likelihood.eval(), losses

# ============================================================
# BO ACQUISITION
# ============================================================
def log_expected_improvement(mean, var, best_f, xi=0.0):
    std = torch.sqrt(var)
    std_safe = torch.clamp(std, min=1e-9)
    z = (mean - best_f - xi) / std_safe

    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    cdf = normal.cdf(z)
    pdf = torch.exp(normal.log_prob(z))

    ei = std * (z * cdf + pdf)
    ei_safe = torch.clamp(ei, min=1e-10)
    return torch.log(ei_safe)

def ucb(mean, var, beta=2.0):
    return mean + beta * torch.sqrt(var)

# ============================================================
# LOAD + PREPARE DATA (your existing pipeline)
# ============================================================
dft_data1 = pd.read_csv('/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/dft_data_temp_pressure_swingswingswing.csv')
dft_data2 = pd.read_csv("/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/dft_data_temp_pressure_präsi_20bin.csv")
dft_data1["density_bulk"] = (
    dft_data1["density_bulk"]
    .astype(str)                            # sicherstellen, dass alles string ist
    .str.strip()                             # Leerzeichen weg
    .str.replace('[', '', regex=False)       # "[" entfernen
    .str.replace(']', '', regex=False)       # "]" entfernen
)
dft_data2["density_bulk"] = (
    dft_data2["density_bulk"]
    .astype(str)                            # sicherstellen, dass alles string ist
    .str.strip()                             # Leerzeichen weg
    .str.replace('[', '', regex=False)       # "[" entfernen
    .str.replace(']', '', regex=False)       # "]" entfernen
)
dft_data1["density_bulk"] = pd.to_numeric(dft_data1["density_bulk"], errors="coerce")
dft_data2["density_bulk"] = pd.to_numeric(dft_data2["density_bulk"], errors="coerce")
dft_data = pd.concat([dft_data1, dft_data2], ignore_index=True).drop_duplicates()
#expV_data = pd.read_csv("/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/Vext_allTEMP_hist_no_pressure_no_chem_20b_swing.csv")
expV_data = pd.read_csv("/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/Boltzmann_allTEMP_hist_logbins_20b_FINALE.csv")

data_all = pd.merge(dft_data, expV_data, "inner", on=["structure_name", "temperature_kelvin"])
feature_columns = [col for col in data_all.columns if is_bin_column(col)]

data_all = data_all[data_all.beladung_mol_per_kg > 0].copy()

# create labels/features as you did
data_all["beladung_pro_vol"] = (
    data_all["beladung_atoms"]
    #.div(data_all["density_bulk"], axis=0)
    .div(data_all["volume_kubAng"], axis=0)
)

data_all[feature_columns] = (
    data_all[feature_columns]
    .multiply(data_all["grid.dv"], axis=0)
    .div(data_all["volume_kubAng"], axis=0)
)

# ============================================================
# MAIN LOOP: pressures (states)
# ============================================================
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

for pressure in PRESSURES:
    print("\n" + "="*80)
    print(f"STATE: T={TEMP} K, p={pressure} bar")

    # Filter state
    data = data_all[(data_all.temperature_kelvin == TEMP) & (data_all.pressure_bar == pressure)].copy()

    # You said: always 245 structures – still guard
    if len(data) < 10:
        print(f"Skipping state (too few samples): n={len(data)}")
        continue

    # -------------------------
    # 1) 10-fold CV → pooled R2 + CI
    # -------------------------
    X = data[feature_columns].values
    y = data[LABEL].values

    split_info = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        x_train = torch.tensor(X[train_idx], dtype=torch.float64)
        y_train = torch.tensor(y[train_idx], dtype=torch.float64)
        x_test  = torch.tensor(X[test_idx], dtype=torch.float64)
        y_test  = torch.tensor(y[test_idx], dtype=torch.float64)

        test_df = data.iloc[test_idx].copy()
        test_df["fold"] = fold

        # feature scaling (per fold, as in your code)
        feature_transformer = MinMaxScaler()
        feature_transformer.fit(x_train)
        xt_train = torch.tensor(feature_transformer.transform(x_train), dtype=torch.float64)
        xt_test  = torch.tensor(feature_transformer.transform(x_test), dtype=torch.float64)

        # label scaling (per fold, as in your code)
        label_transformer = MinMaxScaler()
        label_transformer.fit(y_train.unsqueeze(1))
        yt_train = torch.tensor(label_transformer.transform(y_train.unsqueeze(1)).flatten(), dtype=torch.float64)

        model, likelihood, losses = train_gp(xt_train, yt_train, training_iterations=200)

        with torch.no_grad():
            pred = model(xt_test)
            y_pred = label_transformer.inverse_transform(pred.mean.unsqueeze(1)).squeeze()
            y_pred = np.where(y_pred > 0, y_pred, 0)

        test_df[f"{LABEL}_pred"] = y_pred
        test_df["abs_rel_deviation"] = np.abs((test_df[LABEL] - test_df[f"{LABEL}_pred"]) / test_df[LABEL] * 100)
        split_info.append(test_df)

    results = pd.concat(split_info, ignore_index=True)

    oof_file = Path(f"/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/oof_T{TEMP}_p{pressure:g}.csv")
    results.to_csv(oof_file, index=False)
    print(f"Saved OOF predictions -> {oof_file}")
    
    pooled_r2, boot_std, (ci_low, ci_high), _ = pooled_r2_with_bootstrap_ci(
        results=results, label=LABEL, n_boot=N_BOOT, seed=BOOT_SEED, ci=95
    )

    pooled_entry = pd.DataFrame([{
        "temperature_kelvin": TEMP,
        "pressure_bar": pressure,
        "label": LABEL,
        "n_samples": len(results),
        "pooled_r2": pooled_r2,
        "bootstrap_std": boot_std,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }])
    pooled_entry.to_csv(POOLED_LOG_FILE, mode="a", header=not POOLED_LOG_FILE.exists(), index=False)

    print(f"Pooled R²  : {pooled_r2:.4f}  (95% CI [{ci_low:.4f}, {ci_high:.4f}])")
    print(f"Saved -> {POOLED_LOG_FILE}")

 
    # =============================
    # MAIN (inside your pressure loop)
    # =============================
    global_best_value = float(data[LABEL].max())
    n_candidates = len(data)
    
    max_iter = BO_MAX_ITER
    patience = BO_PATIENCE
    init = INIT_STRATEGY
    acq = ACQ
    
    n_initial = max(3, min(10, n_candidates - 1))
    
    # global scaling (fit once per state)
    feature_scaler = MinMaxScaler().fit(data[feature_columns].values)
    label_scaler   = MinMaxScaler().fit(data[[LABEL]].values)
    
    for run_seed in range(N_BO_RUNS):

        # deterministic seeds
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
    
        # ---- initial selection (SAME for BO and Random) ----
        if init == "kmeans":
            initial_indices = initial_indices_kmeans(data, feature_columns, n_initial, random_state=run_seed)
        else:
            initial_indices = data.sample(n=n_initial, replace=False, random_state=run_seed).index
    
        initial_structures = data.loc[initial_indices, "structure_name"].astype(str).tolist()
    
        # shared base info
        base_row = {
            "seed": run_seed,
            "temperature_kelvin": TEMP,
            "pressure_bar": pressure,
            "label": LABEL,
            "n_candidates": n_candidates,
            "init_strategy": init,
            "n_initial": n_initial,
            "initial_structures": ",".join(initial_structures),
            "max_iter": max_iter,
            "patience": patience,
            "global_best": global_best_value,
            "acq": acq,  # keep even for random (makes filtering easy)
        }
    
        # ====================================================
        # BO RUN
        # ====================================================
        run_id_bo = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_p{pressure}_s{run_seed}_bo"
        best_bo, hist_bo, acq_param_last_bo = run_one_bo_or_random(
            method="bo",
            run_id=run_id_bo,
            run_seed=run_seed,
            data=data,
            feature_columns=feature_columns,
            label=LABEL,
            max_iter=max_iter,
            patience=patience,
            init_strategy=init,
            acq=acq,
            XI0=XI0, XI_MIN=XI_MIN,
            BETA0=BETA0, BETA_MIN=BETA_MIN,
            feature_scaler=feature_scaler,
            label_scaler=label_scaler,
            initial_indices=initial_indices
        )
    
        final_best_bo = float(best_bo[-1])
        found_global_best_bo = abs(final_best_bo - global_best_value) < 1e-12
        ratio_bo = final_best_bo / global_best_value if global_best_value != 0 else np.nan
    
        row_bo = {
            **base_row,
            "run_id": run_id_bo,
            "method": "bo",
            "acq_param_last": acq_param_last_bo,
            "iters_done": len(best_bo) - 1,
            "stopped_early": (len(best_bo) - 1) < max_iter,
            "final_best": final_best_bo,
            "ratio_to_optimum": ratio_bo,
            "found_global_best": found_global_best_bo,
        }
    
        # ====================================================
        # RANDOM RUN (same initial set, same budget)
        # ====================================================
        run_id_rd = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_p{pressure}_s{run_seed}_random"
        best_rd, hist_rd, acq_param_last_rd = run_one_bo_or_random(
            method="random",
            run_id=run_id_rd,
            run_seed=run_seed,
            data=data,
            feature_columns=feature_columns,
            label=LABEL,
            max_iter=max_iter,
            patience=patience,
            init_strategy=init,
            acq=acq,  # unused
            XI0=XI0, XI_MIN=XI_MIN,
            BETA0=BETA0, BETA_MIN=BETA_MIN,
            feature_scaler=feature_scaler,
            label_scaler=label_scaler,
            initial_indices=initial_indices
        )
    
        final_best_rd = float(best_rd[-1])
        found_global_best_rd = abs(final_best_rd - global_best_value) < 1e-12
        ratio_rd = final_best_rd / global_best_value if global_best_value != 0 else np.nan
    
        row_rd = {
            **base_row,
            "run_id": run_id_rd,
            "method": "random",
            "acq_param_last": np.nan,
            "iters_done": len(best_rd) - 1,
            "stopped_early": (len(best_rd) - 1) < max_iter,
            "final_best": final_best_rd,
            "ratio_to_optimum": ratio_rd,
            "found_global_best": found_global_best_rd,
        }
    
        # ====================================================
        # SAVE (ONE clean file each)
        # ====================================================
        pd.DataFrame([row_bo, row_rd]).to_csv(
            RESULTS_FILE,
            mode="a",
            header=not RESULTS_FILE.exists(),
            index=False
        )
    
        pd.DataFrame(hist_bo + hist_rd).to_csv(
            HISTORY_FILE,
            mode="a",
            header=not HISTORY_FILE.exists(),
            index=False
        )
    
        print(f"[seed {run_seed}] saved BO+Random -> {RESULTS_FILE.name}, {HISTORY_FILE.name}")
    
    print(f"Done. Results: {RESULTS_FILE}")
    print(f"Done. History: {HISTORY_FILE}")

print("\nDONE.")
