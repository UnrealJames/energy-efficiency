import argparse
import warnings
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "model" / "energy_model.joblib"


def generate_predictions(input_features, model_bundle):
    features_subset = input_features[model_bundle["features"]].astype(float)

    out_of_grid_violations = []
    for _, row in features_subset.iterrows():
        invalid_columns = [
            col for col in model_bundle["features"] 
            if row[col] not in model_bundle["valid_levels"][col]
        ]
        out_of_grid_violations.append(", ".join(invalid_columns))

    if any(out_of_grid_violations):
        warnings.warn("Some inputs are outside the simulated design grid (extrapolation).")

    predictions_dataframe = pd.DataFrame(index=features_subset.index)

    for target_variable, model_pipeline in model_bundle["models"].items():
        predicted_values = model_pipeline.predict(features_subset)
        interval_halfwidth = model_bundle["interval_halfwidth_95"][target_variable]

        predictions_dataframe[target_variable] = predicted_values.round(2)
        predictions_dataframe[f"{target_variable}_lo95"] = (predicted_values - interval_halfwidth).round(2)
        predictions_dataframe[f"{target_variable}_hi95"] = (predicted_values + interval_halfwidth).round(2)

    heating_threshold = model_bundle["threshold_high_heating"]
    class_probabilities = model_bundle["high_heating_classifier"].predict_proba(features_subset)[:, 1]
    predictions_dataframe[f"p_heating_gt_{heating_threshold}"] = class_probabilities.round(3)

    predictions_dataframe["out_of_grid"] = out_of_grid_violations
    return predictions_dataframe


if __name__ == "__main__":
    bundle = joblib.load(MODEL_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--out")

    for feature_name in bundle["features"]:
        parser.add_argument(f"--{feature_name}", type=float)

    arguments = parser.parse_args()

    if arguments.csv:
        features_dataframe = pd.read_csv(arguments.csv)
    else:
        single_row_dictionary = {feature: getattr(arguments, feature) for feature in bundle["features"]}
        features_dataframe = pd.DataFrame([single_row_dictionary])

    predictions_output = generate_predictions(features_dataframe, bundle)

    combined_results_table = pd.concat(
        [features_dataframe.reset_index(drop=True), predictions_output.reset_index(drop=True)], 
        axis=1
    )

    if arguments.out:
        combined_results_table.to_csv(arguments.out, index=False)
        print("Wrote", arguments.out)
    else:
        print(combined_results_table.transpose().to_string(header=False))
