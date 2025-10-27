import pandas as pd
import torch
from settings import Settings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import jax.numpy as jnp
import numpy as np


def is_number(s):
    """Check if a string can be parsed as a number."""
    try:
        float(s)
        return True
    except Exception:
        return False

def load_csv_and_pickle(settings: Settings) -> pd.DataFrame:
    """
    Lädt main- und supp-CSV und merged sie.
    Erwartet in config.json
    """
    data_cfg = settings.data

    dft_path = Path(data_cfg.csv_dft) if data_cfg.csv_dft else None
    Vext_path = Path(data_cfg.csv_vext) if data_cfg.csv_vext else None

    if dft_path and not dft_path.exists():
        raise FileNotFoundError(f"DFT CSV nicht gefunden: {dft_path}")
    if Vext_path and not Vext_path.exists():
        raise FileNotFoundError(f"Vext CSV nicht gefunden: {Vext_path}")

    # Daten
    dft = pd.read_csv(dft_path) 
    dft = dft[dft["pressure_bar"] == settings.data.pressure_bar]
    dft = dft[dft["temperature_kelvin"] == settings.data.temperature_kelvin]
    Vext = pd.read_csv(Vext_path)
    Vext = Vext[Vext["temperature_kelvin"] == settings.data.temperature_kelvin]

    # Merge
    dft_Vext = pd.merge(dft, Vext, on=["structure_name", "temperature_kelvin"], how="inner")

    return dft_Vext


def preprocess_data(dft: pd.DataFrame, settings) -> pd.DataFrame:
    """
    Funktion muss überarbeitet werden, da sich die Strktur ändert
    """

    df = dft.copy()

    label_col = settings.preprocessing.label

    # Label-Normalisierung
    if settings.preprocessing.label_normalize:
        if label_col in df.columns:
            scaler = StandardScaler()
            df[label_col] = scaler.fit_transform(df[[label_col]])
            #print(f"[INFO] Label '{label_col}' normalisiert (StandardScaler).")
        else:
            print(f"[WARN] Label-Normalisierung aktiviert, aber Spalte '{label_col}' nicht gefunden.")

    # Feature-Normalisierung
    feature_columns = [col for col in df.columns if is_number(col)]

    if settings.preprocessing.feature_normalize:
        if feature_columns:
            scaler = StandardScaler()
            df[feature_columns] = scaler.fit_transform(df[feature_columns])
            #print(f"[INFO] Features normalisiert: {feature_columns}")
        else:
            print("[WARN] Feature-Normalisierung aktiviert, aber keine Feature-Spalten gefunden.")

    return df



def make_splits(dft: pd.DataFrame, settings, device="cpu"):
    
    label_col = settings.preprocessing.label
    feature_columns = [col for col in df.columns if is_number(col)]  

    # NumPy -> Torch
    X = #torch.tensor(df[feature_columns].values, dtype=torch.float32, device=device)
    y = #torch.tensor(df[label_col].values, dtype=torch.float32, device=device)

    k = settings.training.kfold_splits
    shuffle = settings.training.shuffle
    seed = settings.training.random_seed

    kf = KFold(n_splits=k, shuffle=shuffle, random_state=seed)

    splits = []
    for train_idx, val_idx in kf.split(X):
        splits.append((
            X[train_idx], y[train_idx],  # Training
            X[val_idx], y[val_idx]       # Validation
        ))

    print(splits)
    return splits

def hist_vext_raw(data, cutoff, bins=100, log_bins=False):
    
    data = jnp.array(data)
    data = data[data < cutoff].flatten()

    if log_bins:
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), bins)

    hist, bin_edges = np.histogram(data, bins=bins)
    return hist, bin_edges


def hist_vext_exp(data, cutoff, bins=100, log_bins=True):

    data = jnp.array(data)
    data = data[data < cutoff].flatten()
    data_exp = jnp.exp(-data)

    if log_bins:
        bins = np.logspace(np.log10(data_exp.min()), np.log10(data_exp.max()), bins)

    hist, bin_edges = np.histogram(data_exp, bins=bins)
    return hist, bin_edges

def label_umrechnen():

    return

def Vext_extraktion_aus_pickle():
    def hist_vext_exp(data, cutoff, bins=100, log_bins=True):
        #data = jnp.array(data)
        #data = data[data < cutoff].flatten()
        #data_exp = jnp.exp(-data)

        if log_bins:
            bins = np.logspace(np.log10(data_exp.min()), np.log10(data_exp.max()), bins)

        hist, bin_edges = np.histogram(data_exp, bins=bins)
        return hist, bin_edges

    # === Hauptordner ===
    base_dir = "/Users/danielbock/MASTERTHESIS/MASTA/DataArchiv/Vext_allTEMP"
    pattern = re.compile(r"Vext_([A-Z]{3})_(\d+)\.pkl")

    # === Einstellungen ===
    cutoff = 5.0   # ggf. anpassen
    bins = 100

    # === Ergebnisse sammeln ===
    all_dfs = []

    for temp_folder in os.listdir(base_dir):
        temp_path = os.path.join(base_dir, temp_folder)
        if not os.path.isdir(temp_path) or not temp_folder.startswith("Vext_"):
            continue

        print(f"\n📂 Bearbeite Temperatur-Ordner: {temp_folder}")
        df_list = []

        for filename in os.listdir(temp_path):
            if not filename.endswith(".pkl"):
                continue

            match = pattern.match(filename)
            if not match:
                print(f"  ⚠️ Übersprungen (kein gültiger Name): {filename}")
                continue

            struct_name, temp = match.groups()
            file_path = os.path.join(temp_path, filename)

            # Datei laden
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            # in numpy array + squeeze
            data = jnp.array(data)
            mask = data < cutoff
            data = data[mask]
            #print(data)

            # Histogramm berechnen
            hist, bin_edges = hist_vext_exp(data, cutoff=cutoff, bins=bins, log_bins=True)

            # In DataFrame-Zeile schreiben
            df_entry = {"structure": struct_name, "temperature": int(temp)}
            for i in range(len(hist)):
                df_entry[f"{i}"] = hist[i]

            df_list.append(df_entry)

        if df_list:
            df_temp = pd.DataFrame(df_list)
            all_dfs.append(df_temp)
            print(f"  ✅ {len(df_temp)} Strukturen verarbeitet")

    # Alles zusammenführen
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        print(f"\n✅ Gesamt-DataFrame mit {len(df_all)} Strukturen erstellt")

        # Optional speichern
        # df_all.to_csv(os.path.join(base_dir, "Vext_allTEMP_histograms.csv"), index=False)

    return