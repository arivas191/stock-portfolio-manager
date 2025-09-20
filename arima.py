import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm

# Clean price series
def ensure_series(s):
    s = s.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index().astype(float)
    return s.dropna()

# Align by index and return (RMSE, MAE)
def score_aligned(y_true, y_pred):
    pair = pd.concat([y_true.rename("y"), y_pred.rename("hat")],
                     axis=1, join="inner").dropna()
    if pair.empty:
        return np.nan, np.nan
    rmse = float(np.sqrt(mean_squared_error(pair["y"], pair["hat"])))
    mae  = float(mean_absolute_error(pair["y"], pair["hat"]))
    return rmse, mae

# Walk-forward ARIMA/SARIMAX
def arima_walk_forward(series, order=(1,1,1), seasonal_order=(0,0,0,0), trend="n",
    init=250,
    window="expanding",
    update_mode="append",
    refit_every=None,
    maxiter=50
):
    s = ensure_series(series)
    n = len(s)
    if init < 1 or init >= n:
        raise ValueError("Invalid 'init': choose 1 <= init < len(series).")

    preds, idx = [], []

    # initial training window
    tr = s.iloc[:init] if window == "expanding" else s.iloc[max(0, init-int(window)):init]

    # initial fit
    model = sm.tsa.statespace.SARIMAX(
        tr, order=order, seasonal_order=seasonal_order, trend=trend,
        enforce_stationarity=True, enforce_invertibility=True
    )
    cur = model.fit(disp=False, maxiter=maxiter)

    for i in range(init, n):
        # 1-step-ahead forecast for time i
        yhat = cur.get_forecast(steps=1).predicted_mean.iloc[0]
        preds.append(yhat)
        idx.append(s.index[i])

        # observe actual and update
        y_obs = s.iloc[i]
        if update_mode == "append" and window == "expanding":
            cur = cur.append(endog=[y_obs], refit=False)  # fast state update
        else:
            # refit or filter on a new window
            tr = s.iloc[:i+1] if window == "expanding" else s.iloc[max(0, i+1-int(window)):i+1]
            model = sm.tsa.statespace.SARIMAX(
                tr, order=order, seasonal_order=seasonal_order, trend=trend,
                enforce_stationarity=True, enforce_invertibility=True
            )
            do_refit = True
            if update_mode == "refit" and isinstance(refit_every, int) and refit_every > 1:
                if (i - init + 1) % refit_every != 0:
                    cur = model.filter(cur.params)  # reuse params, only filter
                    do_refit = False
            if do_refit:
                cur = model.fit(disp=False, maxiter=maxiter)

    return pd.Series(preds, index=idx, name="hat")

# Validation grid
def grid_search_on_validation(y, train_len, val_len, order_grid,
                              trend="n",
                              window="expanding",
                              update_mode="append",
                              refit_every=None,
                              seasonal_order=(0,0,0,0)):
    val_slice = y.iloc[train_len: train_len + val_len]
    scores, preds = {}, {}
    for order in order_grid:
        wf_val = arima_walk_forward(
            y, order=order, seasonal_order=seasonal_order, trend=trend,
            init=train_len, window=window, update_mode=update_mode, refit_every=refit_every
        )
        # evaluate only on validation period
        val_pred = wf_val.reindex(val_slice.index)
        rmse, _ = score_aligned(val_slice, val_pred)
        scores[order] = rmse
        preds[order]  = val_pred
    leaderboard = pd.Series(scores).sort_values()
    best_order  = leaderboard.index[0]
    return best_order, leaderboard, preds[best_order]


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("prices.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"])

    # --- NEW: collect all tickers & all splits
    all_rows = []
    tickers = sorted(df["Ticker"].unique())

    for TICKER in tickers:
        y = (df[df.Ticker==TICKER].sort_values("Date")
               .set_index("Date")["Close"].astype(float))
        y = ensure_series(y)

        # Split train/val/test set
        n_test = 125
        n_val  = 252
        n_tr   = len(y) - n_val - n_test
        y_tr   = y.iloc[:n_tr]
        y_val  = y.iloc[n_tr:n_tr+n_val]
        y_te   = y.iloc[-n_test:]

        # Shared settings
        trend = "c"
        seasonal_order = (0,0,0,0)
        window = "expanding"
        update_mode = "append"
        refit_every = None

        # Hyperparam selection on validation (walk-forward)
        order_grid = [(0,1,1), (1,1,2)]
        best_order, leaderboard, val_pred = grid_search_on_validation(
            y, train_len=n_tr, val_len=n_val,
            order_grid=order_grid, trend=trend,
            window=window, update_mode=update_mode, refit_every=refit_every,
            seasonal_order=seasonal_order
        )
        print("Validation leaderboard (RMSE):")
        print(leaderboard.round(4))
        print(f"Selected order (p,d,q) for {TICKER}:", best_order)

        # 1) TRAIN predictions: fit on train only
        #    - hat_in  : in-sample fitted values (optimistic)
        #    - hat_dyn : dynamic one-step-ahead inside train (uses only past)
        res_tr = sm.tsa.statespace.SARIMAX(
            y_tr, order=best_order, seasonal_order=seasonal_order, trend=trend,
            enforce_stationarity=True, enforce_invertibility=True
        ).fit(disp=False)

        y_tr_hat_in  = res_tr.get_prediction(dynamic=False).predicted_mean
        p, d, q = best_order
        burn = max(5, p + q + d)    # warm-up
        burn = min(burn, max(1, len(y_tr)//3))
        start_dyn = y_tr.index[burn]
        y_tr_hat_dyn = res_tr.get_prediction(start=start_dyn, dynamic=True).predicted_mean

        df_train_preds = pd.DataFrame({
            "y": y_tr,
            "hat_in":  y_tr_hat_in.reindex(y_tr.index),
            "hat_dyn": y_tr_hat_dyn.reindex(y_tr.index)
        })

        # 2) VALIDATION predictions: walk-forward from end of train
        df_val_preds = pd.DataFrame({
            "y": y_val,
            "hat_wf": val_pred.reindex(y_val.index)
        })

        # 3) TEST predictions: final walk-forward with init at end of train+val
        y_wf_hat = arima_walk_forward(
            y, order=best_order, seasonal_order=seasonal_order, trend=trend,
            init=n_tr + n_val,         # forecast starts at first test date
            window=window, update_mode=update_mode, refit_every=refit_every
        )
        y_te_hat = y_wf_hat.reindex(y_te.index)

        df_test_preds = pd.DataFrame({
            "y": y_te,
            "hat_wf": y_te_hat
        })

        # 4) Metrics for each split
        def rmse_mae(y_true, y_pred):
            pair = pd.concat([y_true.rename("y"), y_pred.rename("hat")], axis=1, join="inner").dropna()
            if pair.empty:
                return np.nan, np.nan
            rmse = float(np.sqrt(mean_squared_error(pair["y"], pair["hat"])))
            mae  = float(mean_absolute_error(pair["y"], pair["hat"]))
            return rmse, mae

        tr_rmse_in,  tr_mae_in  = rmse_mae(df_train_preds["y"], df_train_preds["hat_in"])
        tr_rmse_dyn, tr_mae_dyn = rmse_mae(df_train_preds["y"], df_train_preds["hat_dyn"])
        val_rmse,     val_mae   = rmse_mae(df_val_preds["y"],   df_val_preds["hat_wf"])
        te_rmse,      te_mae    = rmse_mae(df_test_preds["y"],  df_test_preds["hat_wf"])

        print("\n=== Metrics ===")
        print(f"{TICKER} | Train  (in-sample)   RMSE={tr_rmse_in:.4f} | MAE={tr_mae_in:.4f}")
        print(f"{TICKER} | Train  (dynamic)     RMSE={tr_rmse_dyn:.4f} | MAE={tr_mae_dyn:.4f}")
        print(f"{TICKER} | Val    (walk-forward)RMSE={val_rmse:.4f} | MAE={val_mae:.4f}")
        print(f"{TICKER} | Test   (walk-forward)RMSE={te_rmse:.4f}  | MAE={te_mae:.4f}")

        # 5) Save to CSV
        all_rows += [
            pd.DataFrame({"Ticker": TICKER,
                          "Split": "Train",
                          "Date": y_tr.index,
                          "Close": y_tr.values,
                          "ARIMA_PRED": df_train_preds["hat_dyn"].reindex(y_tr.index).values}),
            pd.DataFrame({"Ticker": TICKER,
                          "Split": "Validation",
                          "Date": y_val.index,
                          "Close": y_val.values,
                          "ARIMA_PRED": df_val_preds["hat_wf"].reindex(y_val.index).values}),
            pd.DataFrame({"Ticker": TICKER,
                          "Split": "Test",
                          "Date": y_te.index,
                          "Close": y_te.values,
                          "ARIMA_PRED": df_test_preds["hat_wf"].reindex(y_te.index).values}),
        ]


    out = pd.concat(all_rows, ignore_index=True).sort_values(["Ticker", "Date"])
    out["PrevClose"] = out.groupby("Ticker")["Close"].shift(1)

    # SIGNAL = compare ARIMA_PRED(t) with Close(t-1)
    eps = 0.0
    cond_na = out["PrevClose"].isna()
    cond_eq = (np.abs(out["ARIMA_PRED"] - out["PrevClose"]) / out["PrevClose"]) <= eps
    cond_up = out["ARIMA_PRED"] > out["PrevClose"]

    out["SIGNAL"] = np.select(
        [cond_na, cond_eq, cond_up],
        ["unchanged", "unchanged", "increase"],
        default="decrease"
    )

    final_out = out[["Ticker", "Date", "Close", "ARIMA_PRED", "SIGNAL"]]
    final_out.to_excel("prices_prediction_arima.xlsx", index=False)
