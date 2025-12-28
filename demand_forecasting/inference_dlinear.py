#!/usr/bin/env python3
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.append('/home/quang_ai/DemandForecasting')

from data_utils.load_data import load_and_preprocess_data
from model.baseline_models import get_baseline_model
from model.dlinear import Model

DATA_PATH = "/home/quang_ai/DemandForecasting/data/imputed_data.csv"
CKPT_PATH = "/home/quang_ai/DemandForecasting/demand_forecasting/checkpoints/imputed_decoder/best_dlinear_model.pth"

def get_forecast_data(
    data: pd.DataFrame,
    store_id: int,
    product_id: int,
    start_day: int,
    month: int,
    year: int,
    days_ahead: int = 7
):
    forecast_data = data[
        (data['store_id'] == store_id) &
        (data['product_id'] == product_id)
    ].copy()

    forecast_data['dt'] = pd.to_datetime(forecast_data['dt'])

    start_date = pd.Timestamp(year=year, month=month, day=start_day)
    previous_date = start_date - pd.Timedelta(days=30)
    end_date = start_date + pd.Timedelta(days=days_ahead)

    range_forecast_date = pd.date_range(
        start=start_date,
        end=end_date - pd.Timedelta(days=1)
    ).strftime('%m/%d/%Y').tolist()

    forecast_data = forecast_data[
        (forecast_data['dt'] >= previous_date) &
        (forecast_data['dt'] < end_date)
    ]

    return forecast_data, range_forecast_date


def get_torch_forecast_data(forecast_data: pd.DataFrame):
    # arguments
    series_num = 1
    horizon = 37
    window_size = 30 * 16

    forecast_data['hours_sale'] = forecast_data['hours_sale'].map(
        lambda x: x[1:-1].split(', ')
    )
    forecast_data['dayofweek'] = forecast_data['dt'].dt.dayofweek
    forecast_data['day'] = forecast_data['dt'].dt.day

    numerical_features = [
        'discount', 'precpt',
        'avg_temperature', 'avg_humidity', 'avg_wind_level'
    ]
    binary_features = ['holiday_flag', 'activity_flag']
    time_features = ['dayofweek', 'day']

    hours_sale = np.array(
        forecast_data['hours_sale'].tolist(),
        dtype=float
    )
    hours_sale = hours_sale.reshape(series_num, horizon, 24)[..., 6:22]

    numerical_data = forecast_data[numerical_features].values.astype(float)
    scaler = StandardScaler()
    numerical_normalized = scaler.fit_transform(numerical_data)

    time_data = forecast_data[time_features].values.astype(float)
    time_data[:, 0] = time_data[:, 0] / 6
    time_data[:, 1] = (time_data[:, 1] - 1) / 30

    binary_data = forecast_data[binary_features].values.astype(float)

    features_combined = np.concatenate(
        [numerical_normalized, binary_data, time_data],
        axis=1
    )
    features = features_combined.reshape(series_num, horizon, -1)

    hours_sale = np.expand_dims(hours_sale, axis=-1)
    features = np.expand_dims(features, axis=2)
    features = np.broadcast_to(
        features,
        (series_num, horizon, hours_sale.shape[2], features.shape[-1])
    )

    hour_encoding = np.broadcast_to(
        np.arange(16)[None, None, :, None] / 15,
        (series_num, horizon, 16, 1)
    )

    ds = np.concatenate(
        [features, hour_encoding, hours_sale],
        axis=-1
    )
    ds = ds.reshape(series_num, horizon * 16, -1)
    ds = ds.squeeze(0)

    n_features = ds.shape[-1] - 1

    x = torch.tensor(ds[:window_size, :], dtype=torch.float32)
    x_dec = torch.tensor(ds[window_size:, :n_features], dtype=torch.float32)
    y = torch.tensor(ds[window_size:, -1:], dtype=torch.float32)

    return x, x_dec, y


class Config:
    def __init__(self, use_decoder=True):
        self.model = 'dlinear'
        self.patience = 5
        self.enable_scheduler = True
        self.seq_len = 480
        self.pred_len = 112
        self.enc_in = 11
        self.dec_in = 10
        self.use_decoder = use_decoder
        self.individual = True

        self.batch_size = 1024
        self.lr = 0.001
        self.epochs = 20
        self.train_ratio = 0.99


def main(args):
    data = pd.read_csv(DATA_PATH)

    forecast_data, range_forecast_date = get_forecast_data(
        data,
        store_id=args.store_id,
        product_id=args.product_id,
        start_day=args.start_day,
        month=args.month,
        year=args.year,
        days_ahead=7
    )

    x, x_dec, y = get_torch_forecast_data(forecast_data)

    configs = Config(use_decoder=True)
    model = Model(configs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    state_dict = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        x = x.unsqueeze(0).to(device)
        x_dec = x_dec.unsqueeze(0).to(device)

        output = model(x, x_dec)
        if not configs.use_decoder:
            output = output[:, :, -1:]

        y_pred_np = output.squeeze(0).cpu().numpy()

    y_pred_np = y_pred_np.squeeze()
    y_pred_daily = y_pred_np.reshape(-1, 16).sum(axis=1).tolist()

    final_forecast = pd.DataFrame({
        'date': range_forecast_date,
        'qty': y_pred_daily
    })

    final_forecast.to_csv('final_forecast.csv', index=False)
    print("Saved forecast to final_forecast.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run DLinear Forecast")

    parser.add_argument("--store_id", type=int, required=True)
    parser.add_argument("--product_id", type=int, required=True)
    parser.add_argument("--start_day", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--year", type=int, required=True)

    args = parser.parse_args()
    main(args)
