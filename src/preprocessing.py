import os
import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates Haversine distance between two points in kilometers."""
    r = 6371.0  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def run_preprocessing_pipeline(data_dir: str) -> pd.DataFrame:
    """End-to-end data processing pipeline converting raw relational CSVs into a ML-ready DataFrame."""
    print("Loading raw CSV tables...")
    orders = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv"))
    items = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
    customers = pd.read_csv(os.path.join(data_dir, "olist_customers_dataset.csv"))
    sellers = pd.read_csv(os.path.join(data_dir, "olist_sellers_dataset.csv"))
    products = pd.read_csv(os.path.join(data_dir, "olist_products_dataset.csv"))
    geo = pd.read_csv(os.path.join(data_dir, "olist_geolocation_dataset.csv"))

    # 1. Datetime Parsing & Filter completed deliveries
    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col])

    orders_clean = orders[
        (orders["order_status"] == "delivered")
        & (orders["order_delivered_customer_date"].notna())
    ].copy()

    # 2. Define Binary Target Variable (is_delayed)
    orders_clean["is_delayed"] = (
        orders_clean["order_delivered_customer_date"]
        > orders_clean["order_estimated_delivery_date"]
    ).astype(int)

    # 3. Item-Level Aggregation (Prevent 1-to-many duplication)
    items_agg = (
        items.groupby("order_id")
        .agg(
            total_freight_value=("freight_value", "sum"),
            total_item_price=("price", "sum"),
            num_items=("order_item_id", "count"),
            seller_id=("seller_id", "first"),
        )
        .reset_index()
    )

    df = pd.merge(orders_clean, items_agg, on="order_id", how="inner")

    # 4. Merge Customers, Sellers, and Product Specs
    df = pd.merge(
        df,
        customers[
            [
                "customer_id",
                "customer_zip_code_prefix",
                "customer_city",
                "customer_state",
            ]
        ],
        on="customer_id",
        how="left",
    )

    df = pd.merge(
        df,
        sellers[
            [
                "seller_id",
                "seller_zip_code_prefix",
                "seller_city",
                "seller_state",
            ]
        ],
        on="seller_id",
        how="left",
    )

    items_with_specs = pd.merge(
        items[["order_id", "product_id"]], products, on="product_id", how="left"
    )
    product_agg = (
        items_with_specs.groupby("order_id")
        .agg(
            product_weight_g=("product_weight_g", "mean"),
            product_length_cm=("product_length_cm", "mean"),
            product_height_cm=("product_height_cm", "mean"),
            product_width_cm=("product_width_cm", "mean"),
        )
        .reset_index()
    )

    df = pd.merge(df, product_agg, on="order_id", how="left")

    # 5. Geolocation Merging & Haversine Distance
    geo_clean = (
        geo.groupby("geolocation_zip_code_prefix")
        .agg(
            lat=("geolocation_lat", "mean"), lng=("geolocation_lng", "mean")
        )
        .reset_index()
    )

    df = pd.merge(
        df,
        geo_clean.rename(
            columns={
                "geolocation_zip_code_prefix": "customer_zip_code_prefix",
                "lat": "customer_lat",
                "lng": "customer_lng",
            }
        ),
        on="customer_zip_code_prefix",
        how="left",
    )

    df = pd.merge(
        df,
        geo_clean.rename(
            columns={
                "geolocation_zip_code_prefix": "seller_zip_code_prefix",
                "lat": "seller_lat",
                "lng": "seller_lng",
            }
        ),
        on="seller_zip_code_prefix",
        how="left",
    )

    df["distance_km"] = haversine_distance(
        df["customer_lat"],
        df["customer_lng"],
        df["seller_lat"],
        df["seller_lng"],
    )
    df["is_same_state"] = (df["customer_state"] == df["seller_state"]).astype(
        int
    )

    # 6. Temporal & Volumetric Features
    df["estimated_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 3600)
    df["purchase_dow"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["product_volume_cm3"] = (
        df["product_length_cm"]
        * df["product_height_cm"]
        * df["product_width_cm"]
    )

    # 7. Select Final Columns & Impute Missing Values with Median
    feature_cols = [
        "total_freight_value",
        "total_item_price",
        "num_items",
        "product_weight_g",
        "product_volume_cm3",
        "distance_km",
        "is_same_state",
        "estimated_days",
        "purchase_dow",
        "purchase_hour",
    ]
    target_col = "is_delayed"

    for col in ["product_weight_g", "product_volume_cm3", "distance_km"]:
        df[col] = df[col].fillna(df[col].median())

    final_df = df[feature_cols + [target_col]].copy()
    print(
        f"Preprocessing complete! Dataset shape: {final_df.shape}. Target delay rate: {final_df['is_delayed'].mean():.2%}"
    )
    return final_df


if __name__ == "__main__":
    # Test script directly when run from CLI
    df_processed = run_preprocessing_pipeline(data_dir="data/raw")
    os.makedirs("data/processed", exist_ok=True)
    df_processed.to_csv("data/processed/olist_processed.csv", index=False)