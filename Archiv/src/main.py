import argparse
from settings import Settings
from data_utils import load_csv_and_pickle, preprocess_data, make_splits
from gp_model_kriese import train_gp_model
from logger_utils import setup_logger, save_results_csv


def main(config_file: str):
    settings = Settings(config_file)
    logger = setup_logger(config_file)

    dft, Vext = load_csv_and_pickle(settings)
    dft, Vext = preprocess_data(dft, Vext, settings)
    splits = make_splits(dft, Vext, settings)

    results = []
    for i, (X_train, y_train, X_val, y_val) in enumerate(splits):
        logger.info(f"Training Fold {i+1}/{len(splits)}...")
        _, metrics = train_gp_model(X_train, y_train, X_val, y_val, settings, logger=logger)
        logger.info(f"Fold {i+1}: R²={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}")

        metrics["fold"] = i + 1
        results.append(metrics)

    # Mittelwerte
    mean_r2 = sum(m["r2"] for m in results) / len(results)
    mean_mae = sum(m["mae"] for m in results) / len(results)
    logger.info(f"Mittelwerte über alle Folds: R²={mean_r2:.3f}, MAE={mean_mae:.3f}")

    # Ergebnisse speichern
    save_results_csv(results, config_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Pfad zur JSON-Konfigurationsdatei"
    )
    args = parser.parse_args()

    config_file = args.config if args.config else "config_2.json"
    main(config_file)





