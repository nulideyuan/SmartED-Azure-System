from services.adls_reader import read_latest_features_from_adls
from services.inference import recursive_forecast_from_latest_features

if __name__ == "__main__":
    df = read_latest_features_from_adls()
    pred_df = recursive_forecast_from_latest_features(df, device="cpu")
    print(pred_df)