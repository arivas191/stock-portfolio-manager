import pandas as pd
import numpy as np
import math
from copy import deepcopy
from typing import Iterable, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Setting up LSTM model
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)

class IdentityScaler:
    def fit(self, X): return self
    def transform(self, X): return X.astype(np.float32)
    def inverse_transform(self, X): return X

class SeqDS(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def build_sequences(arr: np.ndarray, lookback: int, target_idx: int):
    """arr: (N,F). Predict y at i using window [i-lookback, i-1]."""
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback:i, :])   
        y.append(arr[i, target_idx])       
    return np.asarray(X, np.float32), np.asarray(y, np.float32)

class LSTMRegressor(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)              
        return self.head(out[:, -1, :]).squeeze(-1) 

def fit_model(model: nn.Module,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader],
              device: torch.device,
              epochs: int = 100,
              lr: float = 1e-3,
              weight_decay: float = 1e-5,
              patience: int = 10):
    """If val_loader provided → early stopping; else fixed-epoch training."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_state, best_val, wait = None, float("inf"), 0

    for ep in range(1, epochs + 1):
        model.train(); tr_sum=0.0; n=0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tr_sum += loss.item()*len(xb); n += len(xb)
        #print(f"Epoch {ep:03d} | train {tr_sum/max(1,n):.6f}", end="")

        if val_loader is None:
            print()
            continue

        model.eval(); va_sum=0.0; m=0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_sum += loss_fn(model(xb), yb).item()*len(xb); m += len(xb)
        vloss = va_sum/max(1,m)
        #print(f" | val {vloss:.6f} | best {min(best_val, vloss):.6f}")

        if vloss < best_val - 1e-7:
            best_val, wait, best_state = vloss, 0, deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                #print("Early stopping.")
                break

    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model

def inverse_target(scaled_vec: np.ndarray, scaler, target_idx: int, num_features: int) -> np.ndarray:
    tmp = np.zeros((len(scaled_vec), num_features), dtype=np.float32)
    tmp[:, target_idx] = scaled_vec
    return scaler.inverse_transform(tmp)[:, target_idx]

@torch.no_grad()
def forecast_rolling_multifeat(model: nn.Module,
                               tv_arr: np.ndarray,   
                               test_arr: np.ndarray,  
                               lookback: int,
                               device: torch.device) -> np.ndarray:
    """
    One-step walk-forward: for step k, predict with window ending at k-1 (all REAL data).
    After predicting k, append the REAL row k into the window.
    Returns preds_scaled: (N,)
    """
    preds = []
    w = tv_arr[-lookback:, :].copy()
    model.eval()
    for k in range(len(test_arr)):
        x = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(device) 
        yhat = model(x).cpu().numpy().ravel()[0]
        preds.append(yhat)
        w = np.vstack([w[1:], test_arr[k][None, :]])
    return np.array(preds, dtype=np.float32)

def cv_lookback_multi_rolling(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    lookback_grid: Iterable[int],
    device: torch.device,
    hidden=64, layers=2, dropout=0.1,
    lr=1e-3, batch=64, epochs=100, patience=10,
    scale=True
) -> Tuple[int, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Train on TRAIN. Evaluate lookback on VALIDATION via rolling walk-forward:
    seed window with the last 'lookback' rows of TRAIN (scaled), then roll across VALIDATION (scaled).
    Returns:
      best_lb,
      cv_table (lookback, val_rmse, val_mae),
      best_val_pred_df (index=df_val.index, cols=[Close_true, Close_pred]),
      best_val_metrics (MAE, RMSE)
    """
    cols = list(feature_cols)
    tr = df_train.sort_index()[cols].dropna().copy()
    va = df_val.sort_index()[cols].dropna().copy()

    target_idx = cols.index(target_col)
    num_features = len(cols)

    scaler = (MinMaxScaler() if scale else IdentityScaler()).fit(tr.values) 
    tr_arr = scaler.transform(tr.values)
    va_arr = scaler.transform(va.values)

    results = []
    best_rmse = float("inf")
    best_lb = None
    best_val_pred_df = None
    best_val_metrics = None

    for lb in lookback_grid:
        if lb >= len(tr_arr):
            #print(f"\n=== Lookback {lb} -> skip (lb >= len(train)) ===")
            continue

        #print(f"\n=== Lookback {lb} ===")

        X_tr, y_tr = build_sequences(tr_arr, lb, target_idx)
        if len(X_tr) == 0:
            print("Skip (insufficient train sequences)."); continue
        tr_loader = DataLoader(SeqDS(X_tr, y_tr), batch_size=batch, shuffle=True, drop_last=True)

        X_va, y_va = build_sequences(va_arr, lb, target_idx)
        va_loader = DataLoader(SeqDS(X_va, y_va), batch_size=batch, shuffle=False) if len(X_va) > 0 else None

        model = LSTMRegressor(num_features, hidden, layers, dropout).to(device)
        model = fit_model(model, tr_loader, va_loader, device, epochs, lr, patience=patience)

        preds_s = forecast_rolling_multifeat(model, tr_arr, va_arr, lb, device)
        y_pred  = inverse_target(preds_s, scaler, target_idx, num_features)          
        y_true  = va[target_col].values.astype(np.float32)                          

        rmse_v = math.sqrt(mean_squared_error(y_true, y_pred))
        mae_v  = mean_absolute_error(y_true, y_pred)
        #print(f"Val RMSE (rolling): {rmse_v:.4f} | Val MAE: {mae_v:.4f}")
        results.append((lb, rmse_v, mae_v))

        if rmse_v < best_rmse:
            best_rmse = rmse_v
            best_lb = lb
            best_val_pred_df = pd.DataFrame({"Close_true": y_true, "Close_pred": y_pred}, index=va.index)
            best_val_metrics = {"MAE": float(mae_v), "RMSE": float(rmse_v)}

    if not results:
        raise ValueError("CV produced no valid results.")

    cv_table = pd.DataFrame(results, columns=["lookback", "val_rmse", "val_mae"])\
                 .sort_values("val_rmse").reset_index(drop=True)
    return best_lb, cv_table, best_val_pred_df, best_val_metrics

def run_lstm_cv_and_forecast_multi_rolling(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: Iterable[str] = ("Open","High","Low","Close","Volume"),
    target_col: str = "Close",
    lookback_grid: Iterable[int] = (40, 60, 90, 120, 180, 252),
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    batch: int = 64,
    epochs: int = 100,
    patience: int = 10,
    scale: bool = True,
    final_early_stop_frac: Optional[float] = 0.10,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Multivariate LSTM with rolling evaluation on VALIDATION (CV),
    and rolling forecast on TRAIN + TEST.
    Returns:
      best_lb, cv_table,
      train_pred_df, train_metrics,
      val_pred_df, val_metrics,
      test_metrics, pred_df, model, scaler, feature_cols
    """
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cols = list(feature_cols)
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
    if target_col not in cols:
        raise ValueError("target_col must be in feature_cols.")

    df_train = df_train.sort_index()[cols].dropna().copy()
    df_val   = df_val.sort_index()[cols].dropna().copy()
    df_test  = df_test.sort_index()[cols].dropna().copy()

    best_lb, cv_table, val_pred_df, val_metrics = cv_lookback_multi_rolling(
        df_train, df_val, cols, target_col, lookback_grid, device,
        hidden, layers, dropout, lr, batch, epochs, patience, scale
    )
    #print(f"\n>>> Best lookback = {best_lb}")

    # Final training on TRAIN+VAL
    df_tv = pd.concat([df_train, df_val]).sort_index()
    if len(df_tv) <= best_lb:
        raise ValueError(f"Not enough history in train+val ({len(df_tv)}) for lookback={best_lb}")

    scaler_final = (MinMaxScaler() if scale else IdentityScaler()).fit(df_tv.values)
    tr_arr   = scaler_final.transform(df_train.values)
    tv_arr   = scaler_final.transform(df_tv.values)
    test_arr = scaler_final.transform(df_test.values)

    target_idx = cols.index(target_col)
    num_features = len(cols)

    X_tv, y_tv = build_sequences(tv_arr, best_lb, target_idx)

    # Final fit: ES on tail of train+val
    if final_early_stop_frac and 0.0 < final_early_stop_frac < 1.0:
        split = max(1, int(len(X_tv) * (1 - final_early_stop_frac)))
        tv_train = DataLoader(SeqDS(X_tv[:split], y_tv[:split]), batch_size=batch, shuffle=True, drop_last=True)
        tv_val   = DataLoader(SeqDS(X_tv[split:], y_tv[split:]), batch_size=batch, shuffle=False)
        model = LSTMRegressor(num_features, hidden, layers, dropout).to(device)
        model = fit_model(model, tv_train, tv_val, device, epochs, lr, patience=patience)
    else:
        tv_loader = DataLoader(SeqDS(X_tv, y_tv), batch_size=batch, shuffle=True, drop_last=True)
        model = LSTMRegressor(num_features, hidden, layers, dropout).to(device)
        model = fit_model(model, tv_loader, None, device, epochs, lr, patience=patience)

    # Rolling forecast on TRAIN (seeded from TRAIN itself)
    preds_tr_s = forecast_rolling_multifeat(model, tr_arr[:best_lb], tr_arr[best_lb:], best_lb, device)
    y_pred_tr  = inverse_target(preds_tr_s, scaler_final, target_idx, num_features)
    y_true_tr  = df_train[target_col].values[best_lb:].astype(np.float32)

    mae_tr  = mean_absolute_error(y_true_tr, y_pred_tr)
    rmse_tr = math.sqrt(mean_squared_error(y_true_tr, y_pred_tr))
    train_pred_df = pd.DataFrame({"Close_true": y_true_tr, "Close_pred": y_pred_tr}, 
                                 index=df_train.index[best_lb:])

    # Rolling forecast on TEST (REAL rows)
    preds_test_s = forecast_rolling_multifeat(model, tv_arr, test_arr, best_lb, device)
    y_pred_test  = inverse_target(preds_test_s, scaler_final, target_idx, num_features)
    y_true_test  = df_test[target_col].values.astype(np.float32)

    mae_t  = mean_absolute_error(y_true_test, y_pred_test)
    rmse_t = math.sqrt(mean_squared_error(y_true_test, y_pred_test))
    pred_df = pd.DataFrame({"Close_true": y_true_test, "Close_pred": y_pred_test}, index=df_test.index)

    return {
        "best_lb": best_lb,
        "cv_table": cv_table,
        "train_pred_df": train_pred_df,
        "train_metrics": {"MAE": mae_tr, "RMSE": rmse_tr},
        "val_pred_df": val_pred_df,
        "val_metrics": val_metrics,
        "test_metrics": {"MAE": mae_t, "RMSE": rmse_t},
        "pred_df": pred_df,
        "model": model,
        "scaler": scaler_final,
        "feature_cols": cols,
    }

# Read the prices data 
df1 = pd.read_csv("prices_all.csv")
df1 =df1.drop(columns="Open.1")
df1["Date"] = pd.to_datetime(df1["Date"])

# Create datasets from AAPL
df_aapl1 = df1[df1["Ticker"] == "AAPL"]
train_aapl1 = df_aapl1.iloc[:int(len(df_aapl1) * 0.7),:]
val_aapl1 = df_aapl1.iloc[len(train_aapl1):len(train_aapl1) + int((len(df_aapl1) - len(train_aapl1)) * 0.67), :]
test_aapl1 = df_aapl1.iloc[len(train_aapl1) + int((len(df_aapl1) - len(train_aapl1)) * 0.67):,:]

# Train LSTM for AAPL 
out_aapl = run_lstm_cv_and_forecast_multi_rolling(
    train_aapl1, val_aapl1, test_aapl1,
    feature_cols=("Open","High","Low","Close","Volume"),
    lookback_grid=(40,60,90,120,160, 180),
    hidden=64, layers=1, dropout=0.15,
    lr=1.7e-3, batch=64, epochs=120, patience=10,
    scale=True,
    final_early_stop_frac=0.15,
    seed=42
)
print("Performance on AAPL stock:")
print("Best lookback:", out_aapl["best_lb"])
print("Test metrics:", out_aapl["test_metrics"])
print("-------------------------------------------------")

# Create datasets for GOOG
df_goog1 = df1[df1["Ticker"] == "GOOG"]
train_goog1 = df_goog1.iloc[:int(len(df_goog1) * 0.7),:]
val_goog1 = df_goog1.iloc[len(train_goog1):len(train_goog1) + int((len(df_goog1) - len(train_goog1)) * 0.67), :]
test_goog1 = df_goog1.iloc[len(train_goog1) + int((len(df_goog1) - len(train_goog1)) * 0.67):,:]

out_goog = run_lstm_cv_and_forecast_multi_rolling(
    train_goog1, val_goog1, test_goog1,
    feature_cols=("Open","High","Low","Close","Volume"),
    lookback_grid=(40,60,90,120,160),
    hidden=64, layers=1, dropout=0.15,
    lr=1.65e-3, batch=64, epochs=120, patience=10,
    scale=True,
    final_early_stop_frac=0.15,
    seed=42
)
print("Performance on GOOG stock:")
print("Best lookback:", out_goog["best_lb"])
print("Test metrics:", out_goog["test_metrics"])
print("-------------------------------------------------")

# Create datasets for MSFT
df_msft1 = df1[df1["Ticker"] == "MSFT"]
train_msft1 = df_msft1.iloc[:int(len(df_msft1) * 0.7),:]
val_msft1 = df_msft1.iloc[len(train_msft1):len(train_msft1) + int((len(df_msft1) - len(train_msft1)) * 0.67), :]
test_msft1 = df_msft1.iloc[len(train_msft1) + int((len(df_msft1) - len(train_msft1)) * 0.67):,:]

# Train LSTM for MSFT
out_msft = run_lstm_cv_and_forecast_multi_rolling(
    train_msft1, val_msft1, test_msft1,
    feature_cols=("Open","High","Low","Close","Volume"),
    lookback_grid=(40,60,90,120,160),
    hidden=64, layers=1, dropout=0.15,
    lr=1.5e-3, batch=64, epochs=120, patience=10,
    scale=True,
    final_early_stop_frac=0.15,  # set None/0 to disable
    seed=42
)
print("Performance on MSFT stock:")
print("Best lookback:", out_msft["best_lb"])
print("Test metrics:", out_msft["test_metrics"])
print("-------------------------------------------------")

# Create function to read prediction dataset from ARIMA model
def create_df(arima_df, stock, excel = True):
    if excel:
        pred_df = pd.read_excel(arima_df)
        pred_df = pred_df[pred_df["Ticker"] == stock]
    else:
        pred_df = pd.read_csv(arima_df)

    pred_df["Date"] = pd.to_datetime(pred_df["Date"])
    val_start = "2024-03-19"
    val_end = "2025-03-20"
    mask = ((pred_df["Date"] >= pd.to_datetime(val_start)) & (pred_df["Date"] <= pd.to_datetime(val_end)))
    pred_arima_val = pred_df.loc[mask]
    pred_arima_test = pred_df.iloc[-125:,:]
    pred_arima_train = pred_df.iloc[:878,:]
    return pred_arima_val, pred_arima_test, pred_df, pred_arima_train

# Create function to run linear regression to combine models
def lr_model(pred_arima_lstm_val, pred_arima_lstm_test, pred_arima_lstm_train):
    FEATURES = ["ARIMA", "LSTM"]
    TARGET = "TARGET"
    X_tr = pred_arima_lstm_val[FEATURES].to_numpy()
    y_tr = pred_arima_lstm_val[TARGET].to_numpy()
    X_te = pred_arima_lstm_test[FEATURES].to_numpy()
    y_te = pred_arima_lstm_test[TARGET].to_numpy()

    model = LinearRegression().fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_pred_val = model.predict(X_tr)

    X_train = pred_arima_lstm_train[FEATURES].dropna().to_numpy()
    y_pred_train = model.predict(X_train)

    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))

    print(f"MAE: {mae:.4f}  RMSE: {rmse:.4f}")
    print("Coefficients:", dict(zip(FEATURES, model.coef_)))
    print("Intercept:", model.intercept_)

    return y_pred, y_pred_val, y_pred_train

# Read arima prediction for AAPL and concate with prediction from LSTM
pred_arima_val_aapl, pred_arima_test_aapl, pred_df, pred_arima_train_aapl = create_df("prices_prediction_arima.xlsx", "AAPL")
pred_arima_lstm_val_aapl = pd.concat([pred_arima_val_aapl["ARIMA_PRED"].reset_index(drop=True).rename("ARIMA"), out_aapl['val_pred_df']["Close_pred"].reset_index(drop=True).rename("LSTM"), pred_arima_val_aapl["Close"].reset_index(drop=True).rename("TARGET")], axis = 1)
pred_arima_lstm_test_aapl = pd.concat([pred_arima_test_aapl["ARIMA_PRED"].reset_index(drop=True).rename("ARIMA"), out_aapl["pred_df"]["Close_pred"].reset_index(drop=True).rename("LSTM"), pred_arima_test_aapl["Close"].reset_index(drop=True).rename("TARGET")], axis = 1)
pred_arima_lstm_train_aapl = pd.concat([pred_arima_train_aapl["ARIMA_PRED"].reset_index(drop=True).rename("ARIMA"), out_aapl["train_pred_df"]["Close_pred"].reset_index(drop=True).rename("LSTM"), pred_arima_train_aapl["Close"].reset_index(drop=True).rename("TARGET")], axis = 1)

# Run linear regression on combined dataset to get new predicted test set values
print("-------------------------------------------------")
print("Performance on hybrid model ARIMA-LSTM on AAPL stock:")
output_arima_lstm_appl_test, output_arima_lstm_appl_val, output_arima_lstm_appl_train = lr_model(pred_arima_lstm_val_aapl, pred_arima_lstm_val_aapl, pred_arima_lstm_train_aapl)

# Read arima prediction for GOOG and concate with prediction from LSTM
pred_arima_val_goog, pred_arima_test_goog, pred_df, pred_arima_train_goog = create_df("prices_prediction_arima.xlsx", "GOOG")
pred_arima_lstm_val_goog = pd.concat([pred_arima_val_goog["ARIMA_PRED"].reset_index(drop=True).rename("ARIMA"), out_goog['val_pred_df']["Close_pred"].reset_index(drop=True).rename("LSTM"), pred_arima_val_goog["Close"].reset_index(drop=True).rename("TARGET")], axis = 1)
pred_arima_lstm_test_goog = pd.concat([pred_arima_test_goog["ARIMA_PRED"].reset_index(drop=True).rename("ARIMA"), out_goog["pred_df"]["Close_pred"].reset_index(drop=True).rename("LSTM"), pred_arima_test_goog["Close"].reset_index(drop=True).rename("TARGET")], axis = 1)
pred_arima_lstm_train_goog = pd.concat([pred_arima_train_goog["ARIMA_PRED"].reset_index(drop=True).rename("ARIMA"), out_goog["train_pred_df"]["Close_pred"].reset_index(drop=True).rename("LSTM"), pred_arima_train_goog["Close"].reset_index(drop=True).rename("TARGET")], axis = 1)

# Run linear regression on combined dataset to get new predicted test set values
print("-------------------------------------------------")
print("Performance on hybrid model ARIMA-LSTM on GOOG stock:")
output_arima_lstm_goog_test, output_arima_lstm_goog_val, output_arima_lstm_goog_train = lr_model(pred_arima_lstm_val_goog, pred_arima_lstm_test_goog, pred_arima_lstm_train_goog)

# Read arima prediction and moving aveage prediction for MSFT and concate them
pred_arima_val_msft, pred_arima_test_msft, pred_df, pred_arima_train_msft = create_df("prices_prediction_arima.xlsx", "MSFT")
pred_mv_val_msft, pred_mv_test_msft, pred_df_mv, pred_mv_train_msft = create_df("mv_result.csv", "a", excel=False)
pred_arima_mv_val_msft = pd.concat([pred_arima_val_msft["ARIMA_PRED"].reset_index(drop=True).rename("ARIMA"), pred_mv_val_msft['MSFT_mv_pred'].reset_index(drop=True).rename("LSTM"), pred_arima_val_msft["Close"].reset_index(drop=True).rename("TARGET")], axis = 1)
pred_arima_mv_test_msft = pd.concat([pred_arima_test_msft["ARIMA_PRED"].reset_index(drop=True).rename("ARIMA"), pred_mv_test_msft["MSFT_mv_pred"].reset_index(drop=True).rename("LSTM"), pred_arima_test_msft["Close"].reset_index(drop=True).rename("TARGET")], axis = 1)
pred_arima_lstm_train_msft = pd.concat([pred_arima_train_msft["ARIMA_PRED"].reset_index(drop=True).rename("ARIMA"), pred_mv_train_msft["MSFT_mv_pred"].reset_index(drop=True).rename("LSTM"), pred_arima_train_msft["Close"].reset_index(drop=True).rename("TARGET")], axis = 1)

# Run linear regression on combined dataset to get new predicted test set values
print("-------------------------------------------------")
print("Performance on hybrid model ARIMA-MA on MSFT stock:")
output_arima_lstm_msft_test, output_arima_lstm_msft_val, output_arima_lstm_msft_train = lr_model(pred_arima_mv_val_msft, pred_arima_mv_test_msft, pred_arima_lstm_train_msft)

# concate predicted values for hybrid arima_lstm model for train, validation, and test sets
arima_lstm_pred = np.concatenate([output_arima_lstm_appl_train, output_arima_lstm_appl_val, output_arima_lstm_appl_test])
arima_lstm_pred = np.concatenate([np.full((len(df_aapl1) - len(arima_lstm_pred)), np.nan), arima_lstm_pred])

# replace the arima prediction column in the prediction dataset
pred_df.loc[pred_df.index[:len(arima_lstm_pred)], "ARIMA_PRED"] = arima_lstm_pred

# create new signal column
signal_list = []
for i in range(len(arima_lstm_pred) - 1):
    if np.isnan(arima_lstm_pred[i]):
        signal_list.append(np.nan)
    elif arima_lstm_pred[i + 1] > arima_lstm_pred[i]:
        signal_list.append("increase")
    elif arima_lstm_pred[i + 1] == arima_lstm_pred[i]:
        signal_list.append("unchanged")
    elif arima_lstm_pred[i + 1] < arima_lstm_pred[i]:
        signal_list.append("decrease")

pred_df.loc[pred_df.index[:len(signal_list)], "SIGNAL"] = signal_list
final_prediction_df = pred_df.rename(columns={"ARIMA_PRED": "PREDICTION"})
final_prediction_df.to_csv("prices_prediction.csv")
