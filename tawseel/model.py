import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

TARGET = "failed"
FEATURE_COLUMNS = ["customer_id", "merchant_id", "area", "created_at", "promised_slot", "cod_amount", "n_items", "address_text"]

def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    if df_out.empty:
        df_out["customer_prior_parcels"] = pd.Series(dtype=int)
        df_out["customer_prior_fail_rate"] = pd.Series(dtype=float)
        return df_out
    res_parcels = pd.Series(index=df_out.index, dtype=int)
    res_fail_rates = pd.Series(index=df_out.index, dtype=float)
    for cust_id, group in df_out.groupby("customer_id", sort=False):
        g_sorted = group.sort_values("created_at")
        times = g_sorted["created_at"].values
        fails = g_sorted["failed"].values.astype(float)
        n = len(g_sorted)
        p_parcels = np.searchsorted(times, times, side='left')
        cum_fails = np.concatenate([[0.0], np.cumsum(fails)])
        sum_earlier_fails = cum_fails[p_parcels]
        p_fail_rates = np.full(n, np.nan, dtype=float)
        mask_has_prior = p_parcels > 0
        p_fail_rates[mask_has_prior] = sum_earlier_fails[mask_has_prior] / p_parcels[mask_has_prior]
        res_parcels.loc[g_sorted.index] = p_parcels
        res_fail_rates.loc[g_sorted.index] = p_fail_rates
    df_out["customer_prior_parcels"] = res_parcels.astype(int)
    df_out["customer_prior_fail_rate"] = res_fail_rates.astype(float)
    return df_out

def time_split(df: pd.DataFrame, time_col: str = "created_at", test_fraction: float = 0.2):
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be between 0 and 1.")
    if df.empty:
        return df.copy(), df.copy()
    df_sorted = df.sort_values(time_col)
    n = len(df_sorted)
    n_test = int(np.ceil(n * test_fraction))
    boundary_idx = n - n_test
    boundary_time = df_sorted.iloc[boundary_idx][time_col]
    return df_sorted[df_sorted[time_col] < boundary_time], df_sorted[df_sorted[time_col] >= boundary_time]

def build_pipeline(seed: int = 0) -> Pipeline:
    num_cols = ["cod_amount", "n_items", "customer_prior_parcels", "customer_prior_fail_rate"]
    cat_cols = ["customer_id", "merchant_id", "area", "promised_slot"]
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
    ], remainder='drop')
    return Pipeline(steps=[('col_transformer', preprocessor), ('classifier', LogisticRegression(random_state=seed, max_iter=1000))])

def evaluate(model, test_df: pd.DataFrame) -> dict:
    y_true = test_df[TARGET].values
    probs = model.predict_proba(test_df)[:, 1]
    return {"roc_auc": float(roc_auc_score(y_true, probs)), "pr_auc": float(average_precision_score(y_true, probs)), "brier": float(brier_score_loss(y_true, probs)), "base_rate": float(np.mean(y_true)), "n": int(len(test_df))}

def compare_to_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0) -> dict:
    model = build_pipeline(seed=seed)
    model.fit(train_df, train_df[TARGET])
    m_metrics = evaluate(model, test_df)
    dummy = DummyClassifier(strategy="prior", random_state=seed)
    dummy.fit(train_df, train_df[TARGET])
    d_probs = dummy.predict_proba(test_df)[:, 1]
    y_true = test_df[TARGET].values
    d_metrics = {"roc_auc": float(roc_auc_score(y_true, d_probs)), "pr_auc": float(average_precision_score(y_true, d_probs)), "brier": float(brier_score_loss(y_true, d_probs)), "base_rate": float(np.mean(y_true)), "n": int(len(test_df))}
    return {"model": m_metrics, "dummy": d_metrics, "lift_pr_auc": float(m_metrics["pr_auc"] - d_metrics["pr_auc"])}