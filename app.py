import json
from datetime import datetime, time

import joblib
import pandas as pd
import streamlit as st

from features import FEATURES, TARGET, add_features

st.set_page_config(page_title="Delivery time predictor", page_icon="🛵", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load("models/eta_model.joblib")
    with open("models/meta.json") as f:
        meta = json.load(f)
    demo = pd.read_csv("models/demo_orders.csv")
    return model, meta, demo


model, meta, demo = load_model()
mae = meta["test_mae"]
cats = meta["categories"]


def predict(order):
    row = add_features(pd.DataFrame([order]))
    return float(model.predict(row[FEATURES])[0])


st.title("🛵 Restaurant order delivery time predictor")
st.caption(f"LightGBM model trained on real DoorDash orders · average error on the test week ≈ {mae:.1f} min")

tab1, tab2 = st.tabs(["New order", "Real order from the test week"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Restaurant and order**")
        category = st.selectbox("Store category", cats["store_primary_category"])
        market = st.selectbox("Market (city id)", [c for c in cats["market_id"] if c != "unknown"])
        protocol = st.selectbox("Order protocol", [c for c in cats["order_protocol"] if c != "unknown"])
        total_items = st.number_input("Total items", 1, 50, 3)
        distinct_items = st.number_input("Distinct items", 1, 20, 2)
    with c2:
        st.markdown("**Prices, $**")
        subtotal = st.number_input("Order subtotal", 1.0, 300.0, 25.0)
        min_price = st.number_input("Cheapest item", 0.0, 150.0, 6.0)
        max_price = st.number_input("Most expensive item", 0.0, 150.0, 12.0)
        st.markdown("**Distance**")
        driving = st.slider("Estimated driving time, min", 1, 35, 9)
    with c3:
        st.markdown("**Marketplace load right now**")
        onshift = st.slider("Couriers on shift", 1, 170, 40)
        busy = st.slider("Busy couriers", 0, 170, 35)
        outstanding = st.slider("Outstanding orders", 0, 285, 50)
        st.markdown("**Time (UTC)**")
        order_date = st.date_input("Date", datetime(2015, 2, 14))
        order_time = st.time_input("Time", time(2, 0))

    order = {
        "created_at": datetime.combine(order_date, order_time),
        "market_id": market,
        "order_protocol": protocol,
        "store_primary_category": category,
        "total_items": total_items,
        "num_distinct_items": distinct_items,
        "subtotal": subtotal * 100,            # dollars -> cents
        "min_item_price": min_price * 100,
        "max_item_price": max_price * 100,
        "total_onshift_dashers": onshift,
        "total_busy_dashers": min(busy, onshift),
        "total_outstanding_orders": outstanding,
        "estimated_order_place_duration": meta["median_place_duration"],
        "estimated_store_to_consumer_driving_duration": driving * 60,   # minutes -> seconds
    }

    if st.button("Predict delivery time", type="primary"):
        p = predict(order)
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted delivery time", f"{p:.0f} min")
        m2.metric("Likely range (± MAE)", f"{max(p - mae, 0):.0f}–{p + mae:.0f} min")
        m3.metric("Orders per courier", f"{outstanding / onshift:.1f}")

        st.markdown("**What if the marketplace were busier or quieter?**")
        scenarios = []
        for label, k in [("Quiet (½ orders)", 0.5), ("Current", 1.0), ("Busy (1.5×)", 1.5), ("Peak (2×)", 2.0)]:
            o = dict(order, total_outstanding_orders=outstanding * k,
                     total_busy_dashers=min(onshift, busy * k))
            scenarios.append({"Scenario": label, "Predicted minutes": round(predict(o), 1)})
        st.dataframe(pd.DataFrame(scenarios), hide_index=True)

with tab2:
    st.write("Real orders from the last week of data, which the model never saw during training.")
    if "i" not in st.session_state:
        st.session_state.i = 0
    if st.button("🎲 Pick a random real order"):
        st.session_state.i = int(demo.sample(1).index[0])
    row = demo.loc[[st.session_state.i]]
    p = float(model.predict(add_features(row)[FEATURES])[0])
    actual = float(row[TARGET].iloc[0])
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted", f"{p:.0f} min")
    m2.metric("Actual", f"{actual:.0f} min")
    m3.metric("Error", f"{actual - p:+.0f} min")
    st.dataframe(row[["created_at", "store_primary_category", "total_items", "subtotal",
                      "total_onshift_dashers", "total_busy_dashers", "total_outstanding_orders",
                      "estimated_store_to_consumer_driving_duration"]].T.astype(str))