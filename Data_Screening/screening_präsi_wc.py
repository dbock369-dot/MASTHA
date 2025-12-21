import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
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
BASE = Path("/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv")

LABEL = "working_capacity"

# Define working-capacity states (high vs low)
# Each tuple: (temp_high, pres_high, temp_low, pres_low)
WC_STATES = [
    (298.0, 5.0, 298.0, 1.0),
    #(298.0, 1.0,   298.0, 0.1),
    (298.0, 1.0,   298.0, 0.1),
    (400.0, 1.0,   298.0, 1.0),
    #(373.0, 1.0,   298.0, 1.0),
    #(348.0, 1.0,   298.0, 1.0)
]

N_SPLITS = 10
N_BOOT = 2000
BOOT_SEED = 42

# BO stability settings
N_BO_RUNS = 20
BO_MAX_ITER = 100
BO_PATIENCE = 10

INIT_STRATEGY = "kmeans"    # "kmeans" or "random"
ACQ = "ucb"                 # "ei" or "ucb"

# EI params
XI0, XI_MIN = 0.05, 0.005
# UCB params
BETA0, BETA_MIN = 3.0, 1.0

# output files
POOLED_LOG_FILE = BASE / "gp_pooled_r2_log_wc_dens_ucb5.csv"
RESULTS_FILE     = BASE / "bo_random_results_wc_dens_ucb5.csv"
HISTORY_FILE     = BASE / "bo_random_history_wc_dens_ucb5.csv"

# input files
DFT_FILE_1 = BASE / "dft_data_temp_pressure_swingswingswing.csv"
DFT_FILE_2 = BASE / "dft_data_temp_pressure_swingswingswing5.csv"
#DFT_FILE_2 = BASE / "dft_data_temp_pressure_präsi_20bin.csv"
VEXT_FILE  = BASE / "Vext_allTEMP_hist_no_pressure_no_chem_20b_swing.csv"

# ============================================================
# HELPERS
# ============================================================
def is_bin_column(col) -> bool:
    # raw bin columns before suffixing might be ints or digit-strings or "bin_###"
    if isinstance(col, (int, np.integer)):
        return True
    s = str(col)
    if s.isdigit():
        return True
    if re.fullmatch(r"bin_\d+", s):
        return True
    return False

def is_suffixed_bin_column(col: str, suffix: str) -> bool:
    # after merge we expect columns like "0_high" / "bin_12_high"
    s = str(col)
    if not s.endswith(suffix):
        return False
    base = s[: -len(suffix)]
    # base might end with "_" due to naming, guard:
    if base.endswith("_"):
        base = base[:-1]
    return is_bin_column(base)

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

def wc_state_label(tH, pH, tL, pL):
    return f"(T={float(tH):.0f}K,p={float(pH):g})-(T={float(tL):.0f}K,p={float(pL):g})"

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

    for _ in range(training_iterations):
        optimizer.zero_grad()
        output = model(xt_train)
        loss = -mll(output, yt_train)
        loss.backward()
        optimizer.step()

    return model.eval(), likelihood.eval()

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
# BUILD WORKING CAPACITY DATASET
# ============================================================
def build_wc_dataset(data_all, temp_high, pres_high, temp_low, pres_low):
    # filter states
    high = data_all[(data_all["temperature_kelvin"] == temp_high) &
                    (data_all["pressure_bar"] == pres_high)].copy()
    low  = data_all[(data_all["temperature_kelvin"] == temp_low) &
                    (data_all["pressure_bar"] == pres_low)].copy()

    # uniqueness per structure+state
    high = high.drop_duplicates(subset=["structure_name", "temperature_kelvin", "pressure_bar"])
    low  = low.drop_duplicates(subset=["structure_name", "temperature_kelvin", "pressure_bar"])

    # bin cols per side (before merge)
    feat_high = [c for c in high.columns if is_bin_column(c)]
    feat_low  = [c for c in low.columns  if is_bin_column(c)]

    # beladung_pro_vol per side
    for df in (high, low):
        df["beladung_pro_vol"] = df["beladung_atoms"].div(df["volume_kubAng"], axis=0)

    # normalize bin features per side
    high[feat_high] = (high[feat_high].multiply(high["grid.dv"], axis=0)
                                  .div(high["volume_kubAng"], axis=0))
    low[feat_low]   = (low[feat_low].multiply(low["grid.dv"], axis=0)
                                .div(low["volume_kubAng"], axis=0))

    # merge only what we need + bins
    high_keep = ["structure_name", "beladung_pro_vol"] + feat_high
    low_keep  = ["structure_name", "beladung_pro_vol"] + feat_low

    merged = pd.merge(
        high[high_keep],
        low[low_keep],
        on="structure_name",
        suffixes=("_high", "_low"),
        how="inner"
    )

    # label
    merged["working_capacity"] = (merged["beladung_pro_vol_high"] - merged["beladung_pro_vol_low"]).abs()
    merged["working_capacity"] = pd.to_numeric(merged["working_capacity"], errors="coerce")

    # feature columns: concat bins_high + bins_low (stable sort)
    feat_high_merged = sorted([c for c in merged.columns if is_suffixed_bin_column(c, "_high")])
    feat_low_merged  = sorted([c for c in merged.columns if is_suffixed_bin_column(c, "_low")])
    feature_columns = feat_high_merged + feat_low_merged

    # state metadata for logging
    merged["temp_high"] = float(temp_high)
    merged["pres_high"] = float(pres_high)
    merged["temp_low"]  = float(temp_low)
    merged["pres_low"]  = float(pres_low)

    # drop NaNs in label + features
    merged = merged.dropna(subset=["working_capacity"] + feature_columns).copy()

    return merged, feature_columns

# ============================================================
# BO / RANDOM TRAJECTORY
# ============================================================
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
        acq,                      # "ei" or "ucb"
        XI0, XI_MIN,
        BETA0, BETA_MIN,
        feature_scaler,
        label_scaler,
        initial_indices
    ):
    candidates = data.copy()

    selected = candidates.loc[initial_indices].copy()
    candidates = candidates.drop(initial_indices).copy()

    # numeric WC state
    tH = float(data["temp_high"].iloc[0])
    pH = float(data["pres_high"].iloc[0])
    tL = float(data["temp_low"].iloc[0])
    pL = float(data["pres_low"].iloc[0])

    best = [float(selected[label].max())]
    history = [{
        "run_id": run_id,
        "seed": run_seed,
        "iter": 0,
        "best_so_far": best[0],
        "temp_high": tH,
        "pres_high": pH,
        "temp_low": tL,
        "pres_low": pL,
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

        if method == "random":
            pick = rng.integers(0, len(candidates))

        elif method == "bo":
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

            model, likelihood = train_gp(train_x, train_y, training_iterations=250)

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

        selected = pd.concat([selected, candidates.iloc[[pick]]], ignore_index=False)
        candidates = candidates.drop(candidates.index[pick])

        best.append(float(selected[label].max()))
        history.append({
            "run_id": run_id,
            "seed": run_seed,
            "iter": len(best) - 1,
            "best_so_far": best[-1],
            "temp_high": tH,
            "pres_high": pH,
            "temp_low": tL,
            "pres_low": pL,
            "method": method
        })

    return best, history, acq_param_last

# ============================================================
# LOAD + PREPARE DATA
# ============================================================
dft_data1 = pd.read_csv(DFT_FILE_1)
dft_data2 = pd.read_csv(DFT_FILE_2)

for df in (dft_data1, dft_data2):
    df["density_bulk"] = (
        df["density_bulk"]
        .astype(str)
        .str.strip()
        .str.replace('[', '', regex=False)
        .str.replace(']', '', regex=False)
    )
    df["density_bulk"] = pd.to_numeric(df["density_bulk"], errors="coerce")

dft_data = pd.concat([dft_data1, dft_data2], ignore_index=True).drop_duplicates()

expV_data = pd.read_csv(VEXT_FILE)

# merge
data_all = pd.merge(dft_data, expV_data, "inner", on=["structure_name", "temperature_kelvin"])

# optional cleanup
data_all = data_all[data_all["beladung_mol_per_kg"] > 0].copy()

# ============================================================
# MAIN LOOP: WC states
# ============================================================
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

for (tH, pH, tL, pL) in WC_STATES:
    print("\n" + "=" * 80)
    print(f"WC-STATE: {wc_state_label(tH, pH, tL, pL)}")

    data, feature_columns = build_wc_dataset(data_all, tH, pH, tL, pL)

    if len(data) < 10:
        print(f"Skipping (too few samples): n={len(data)}")
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

        test_df = data.iloc[test_idx].copy()
        test_df["fold"] = fold

        # feature scaling per fold
        feature_transformer = MinMaxScaler()
        feature_transformer.fit(x_train)
        xt_train = torch.tensor(feature_transformer.transform(x_train), dtype=torch.float64)
        xt_test  = torch.tensor(feature_transformer.transform(x_test), dtype=torch.float64)

        # label scaling per fold
        label_transformer = MinMaxScaler()
        label_transformer.fit(y_train.unsqueeze(1))
        yt_train = torch.tensor(label_transformer.transform(y_train.unsqueeze(1)).flatten(), dtype=torch.float64)

        model, likelihood = train_gp(xt_train, yt_train, training_iterations=200)

        with torch.no_grad():
            pred = model(xt_test)
            y_pred = label_transformer.inverse_transform(pred.mean.unsqueeze(1)).squeeze()
            # working_capacity should be >=0
            y_pred = np.where(y_pred > 0, y_pred, 0)

        test_df[f"{LABEL}_pred"] = y_pred
        test_df["abs_rel_deviation"] = np.abs((test_df[LABEL] - test_df[f"{LABEL}_pred"]) / test_df[LABEL] * 100)

        split_info.append(test_df)

    results = pd.concat(split_info, ignore_index=True)

    oof_file = BASE / f"oof_wc_TH{tH:g}_pH{pH:g}_TL{tL:g}_pL{pL:g}.csv"
    results.to_csv(oof_file, index=False)
    print(f"Saved OOF predictions -> {oof_file}")

    pooled_r2, boot_std, (ci_low, ci_high), _ = pooled_r2_with_bootstrap_ci(
        results=results, label=LABEL, n_boot=N_BOOT, seed=BOOT_SEED, ci=95
    )

    pooled_entry = pd.DataFrame([{
        "temp_high": float(tH),
        "pres_high": float(pH),
        "temp_low": float(tL),
        "pres_low": float(pL),
        "wc_state": wc_state_label(tH, pH, tL, pL),
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

    # -------------------------
    # 2) BO vs Random (same init)
    # -------------------------
    global_best_value = float(data[LABEL].max())
    n_candidates = len(data)

    max_iter = BO_MAX_ITER
    patience = BO_PATIENCE
    init = INIT_STRATEGY
    acq = ACQ

    n_initial = max(3, min(10, n_candidates - 1))

    # global scaling per WC state
    feature_scaler = MinMaxScaler().fit(data[feature_columns].values)
    label_scaler   = MinMaxScaler().fit(data[[LABEL]].values)

    for run_seed in range(N_BO_RUNS):
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        # initial selection (same for BO and Random)
        if init == "kmeans":
            initial_indices = initial_indices_kmeans(data, feature_columns, n_initial, random_state=run_seed)
        else:
            initial_indices = data.sample(n=n_initial, replace=False, random_state=run_seed).index

        initial_structures = data.loc[initial_indices, "structure_name"].astype(str).tolist()

        base_row = {
            "seed": run_seed,
            "temp_high": float(tH),
            "pres_high": float(pH),
            "temp_low": float(tL),
            "pres_low": float(pL),
            "wc_state": wc_state_label(tH, pH, tL, pL),
            "label": LABEL,
            "n_candidates": n_candidates,
            "init_strategy": init,
            "n_initial": n_initial,
            "initial_structures": ",".join(initial_structures),
            "max_iter": max_iter,
            "patience": patience,
            "global_best": global_best_value,
            "acq": acq,
        }

        # BO run
        run_id_bo = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_wc_TH{tH:g}_pH{pH:g}_TL{tL:g}_pL{pL:g}_s{run_seed}_bo"
        best_bo, hist_bo, acq_param_last_bo = run_one_bo_or_random(
            method="bo",
            run_id=run_id_bo,
            run_seed=run_seed,
            data=data,
            feature_columns=feature_columns,
            label=LABEL,
            max_iter=max_iter,
            patience=patience,
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

        # Random run
        run_id_rd = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_wc_TH{tH:g}_pH{pH:g}_TL{tL:g}_pL{pL:g}_s{run_seed}_random"
        best_rd, hist_rd, _ = run_one_bo_or_random(
            method="random",
            run_id=run_id_rd,
            run_seed=run_seed,
            data=data,
            feature_columns=feature_columns,
            label=LABEL,
            max_iter=max_iter,
            patience=patience,
            acq=acq,  # unused for random
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

        # SAVE
        pd.DataFrame([row_bo, row_rd]).to_csv(
            RESULTS_FILE,
            mode="a",
            header=not RESULTS_FILE.exists(),
            index=False
        )

        hist_df = pd.DataFrame(hist_bo + hist_rd)
        hist_df.to_csv(
            HISTORY_FILE,
            mode="a",
            header=not HISTORY_FILE.exists(),
            index=False
        )

        print(f"[seed {run_seed}] saved BO+Random -> {RESULTS_FILE.name}, {HISTORY_FILE.name}")

    print(f"Done WC-State. Results: {RESULTS_FILE}")
    print(f"Done WC-State. History: {HISTORY_FILE}")

print("\nDONE.")
