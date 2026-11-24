import numbers
import numpy as np
import pandas as pd

TARGET = "delivery_minutes"
MAX_MINUTES = 180

NUMERIC_FEATURES = [
    "total_items", "subtotal", "num_distinct_items", "min_item_price", "max_item_price",
    "avg_item_price", "price_range",
    "total_onshift_dashers", "total_busy_dashers", "total_outstanding_orders",
    "available_dashers", "busy_ratio", "orders_per_dasher",
    "est_place_min", "est_driving_min",
    "hour", "day_of_week", "is_weekend",
]
CATEGORICAL_FEATURES = ["market_id", "order_protocol", "store_primary_category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean_data(df):
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"])

    # 1. target: minutes from order creation to delivery
    df[TARGET] = (df["actual_delivery_time"] - df["created_at"]).dt.total_seconds() / 60

    # 2. remove rows without target and unrealistic deliveries
    df = df.dropna(subset=[TARGET])
    df = df[(df[TARGET] > 0) & (df[TARGET] <= MAX_MINUTES)]

    # 3. remove the single stray order from 2014
    df = df[df["created_at"] >= "2015-01-01"]

    # 4. negative counts / prices are impossible -> missing
    for col in ["total_onshift_dashers", "total_busy_dashers",
                "total_outstanding_orders", "min_item_price"]:
        df.loc[df[col] < 0, col] = np.nan

    # 5. fill missing store info with the most common value for the same store
    for col in ["store_primary_category", "market_id", "order_protocol"]:
        store_mode = df.groupby("store_id")[col].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        df[col] = df[col].fillna(df["store_id"].map(store_mode))

    return df.sort_values("created_at").reset_index(drop=True)


def to_category(value):
    if pd.isna(value):
        return "unknown"
    if isinstance(value, numbers.Number):
        return str(int(value))
    return str(value)


def add_features(df):
    df = df.copy()
    created = pd.to_datetime(df["created_at"])

    # time
    df["hour"] = created.dt.hour
    df["day_of_week"] = created.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # marketplace load
    onshift = df["total_onshift_dashers"].replace(0, np.nan)
    df["available_dashers"] = df["total_onshift_dashers"] - df["total_busy_dashers"]
    df["busy_ratio"] = (df["total_busy_dashers"] / onshift).clip(0, 1)
    df["orders_per_dasher"] = df["total_outstanding_orders"] / onshift

    # order
    df["avg_item_price"] = df["subtotal"] / df["total_items"]
    df["price_range"] = df["max_item_price"] - df["min_item_price"]

    # seconds -> minutes
    df["est_place_min"] = df["estimated_order_place_duration"] / 60
    df["est_driving_min"] = df["estimated_store_to_consumer_driving_duration"] / 60

    # categories as text
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].apply(to_category)
    return df