import logging
import pandas as pd
from pathlib import Path


def setup_logger(config_path: str, log_dir: str = "logs") -> logging.Logger:
    """
    Setzt den Logger auf:
    - Ausgabe in Konsole
    - Ausgabe in Datei (logs/<config_name>.log)
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    config_name = Path(config_path).stem  # nur Dateiname ohne .json
    log_path = Path(log_dir) / f"{config_name}.log"

    logger = logging.getLogger("masta")
    logger.setLevel(logging.INFO)

    # Doppel-Handler vermeiden
    if not logger.handlers:
        # Konsole
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(console_fmt)

        # Datei
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(logging.INFO)
        file_fmt = logging.Formatter("%(asctime)s - [%(levelname)s] %(message)s")
        file_handler.setFormatter(file_fmt)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # Hinweis: Mit welcher Config gestartet
    logger.info(f"Run gestartet mit Config: {config_path}")

    return logger


def save_results_csv(results: list[dict], config_path: str, out_dir: str = "results"):
    """
    Speichert die Ergebnisse aller Folds in eine CSV.
    results: Liste von Dicts mit Keys wie {"fold": 1, "r2": ..., "mae": ...}
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    config_name = Path(config_path).stem
    out_path = Path(out_dir) / f"{config_name}_results.csv"

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Ergebnisse gespeichert unter {out_path}")

