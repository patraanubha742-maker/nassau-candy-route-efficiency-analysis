
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Nassau Candy Route Efficiency",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}
FACTORY_COORDS = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.119140, -96.181150),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107),
}
STATE_ABBR = {
"Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO",
"Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID",
"Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
"Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN",
"Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
"New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC",
"North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA",
"Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN","Texas":"TX",
"Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV",
"Wisconsin":"WI","Wyoming":"WY","District of Columbia":"DC"
}

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        candidates = [Path("Nassau Candy Distributor.csv"), Path("data/Nassau Candy Distributor.csv")]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            st.error("CSV not found. Put 'Nassau Candy Distributor.csv' beside app.py or upload it below.")
            st.stop()
        df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    df["Shipping Lead Time (Days)"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Factory"] = df["Product Name"].map(FACTORY_MAP)
    df["Route"] = df["Factory"] + " → " + df["State/Province"]
    return df

def route_summary(df):
    r = df.groupby(["Factory","State/Province","Region"], as_index=False).agg(
        Shipments=("Order ID","nunique"),
        Average_Lead_Time=("Shipping Lead Time (Days)","mean"),
        Lead_Time_Std=("Shipping Lead Time (Days)","std"),
        Total_Sales=("Sales","sum"),
        Total_Units=("Units","sum"),
    )
    r["Lead_Time_Std"] = r["Lead_Time_Std"].fillna(0)
    mn, mx = r["Average_Lead_Time"].min(), r["Average_Lead_Time"].max()
    r["Efficiency_Score"] = 100 if mx == mn else 100*(mx-r["Average_Lead_Time"])/(mx-mn)
    r["Route"] = r["Factory"] + " → " + r["State/Province"]
    return r

st.title("🚚 Factory-to-Customer Shipping Route Efficiency Analysis")
st.caption("Nassau Candy Distributor | Data Science Internship Project")

with st.sidebar:
    st.header("Filters")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    df = load_data(uploaded)

    min_date = df["Order Date"].min().date()
    max_date = df["Order Date"].max().date()
    date_range = st.date_input("Order date range", (min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    regions = st.multiselect("Region", sorted(df["Region"].dropna().unique()), default=sorted(df["Region"].dropna().unique()))
    states = st.multiselect("State / Province", sorted(df["State/Province"].dropna().unique()))
    modes = st.multiselect("Ship mode", sorted(df["Ship Mode"].dropna().unique()), default=sorted(df["Ship Mode"].dropna().unique()))
    threshold = st.slider("Lead-time delay threshold (days)", 0, max(2000, int(df["Shipping Lead Time (Days)"].max())), 1300)
    min_shipments = st.slider("Minimum shipments for route ranking", 1, 50, 3)

    mask = (
        (df["Order Date"].dt.date >= start_date) &
        (df["Order Date"].dt.date <= end_date) &
        (df["Region"].isin(regions)) &
        (df["Ship Mode"].isin(modes))
    )
    if states:
        mask &= df["State/Province"].isin(states)
    fdf = df.loc[mask].copy()

st.markdown("### Executive KPI Snapshot")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Shipments", f"{fdf['Order ID'].nunique():,}")
k2.metric("Avg lead time", f"{fdf['Shipping Lead Time (Days)'].mean():,.1f} days" if len(fdf) else "—")
k3.metric("Median lead time", f"{fdf['Shipping Lead Time (Days)'].median():,.0f} days" if len(fdf) else "—")
k4.metric("Delayed shipments", f"{(fdf['Shipping Lead Time (Days)'] > threshold).mean()*100:,.1f}%" if len(fdf) else "—")
k5.metric("Route count", f"{fdf['Route'].nunique():,}")

if len(fdf) == 0:
    st.warning("No records match the selected filters.")
    st.stop()

st.info(
    "Data-quality note: the supplied dataset contains order dates from 2024–2025 and ship dates from 2026–2030. "
    "This produces lead times of roughly 904–1,642 days, which is not typical for physical parcel shipping. "
    "The dashboard therefore treats the dates as analytical input and explicitly flags this issue for validation."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Route Efficiency Overview", "Geographic Shipping Map", "Ship Mode Comparison", "Route Drill-Down"
])

with tab1:
    st.subheader("Route Performance Leaderboard")
    r = route_summary(fdf)
    ranked = r[r["Shipments"] >= min_shipments].copy()
    c1, c2 = st.columns(2)
    with c1:
        top = ranked.sort_values("Average_Lead_Time").head(10)
        fig = px.bar(top.sort_values("Average_Lead_Time", ascending=True),
                     x="Average_Lead_Time", y="Route", orientation="h",
                     title="Top 10 Most Efficient Routes",
                     labels={"Average_Lead_Time":"Average lead time (days)"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        bottom = ranked.sort_values("Average_Lead_Time", ascending=False).head(10)
        fig = px.bar(bottom.sort_values("Average_Lead_Time"),
                     x="Average_Lead_Time", y="Route", orientation="h",
                     title="Bottom 10 Least Efficient Routes",
                     labels={"Average_Lead_Time":"Average lead time (days)"})
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        r.sort_values(["Efficiency_Score"], ascending=False)[
            ["Route","Factory","State/Province","Region","Shipments",
             "Average_Lead_Time","Lead_Time_Std","Efficiency_Score","Total_Sales","Total_Units"]
        ].round(2),
        use_container_width=True, hide_index=True
    )

with tab2:
    st.subheader("US Geographic Shipping Efficiency")
    geo = fdf.groupby("State/Province", as_index=False).agg(
        Shipments=("Order ID","nunique"),
        Average_Lead_Time=("Shipping Lead Time (Days)","mean"),
        Total_Sales=("Sales","sum")
    )
    geo["StateCode"] = geo["State/Province"].map(STATE_ABBR)
    us_geo = geo.dropna(subset=["StateCode"]).copy()
    if len(us_geo):
        fig = px.choropleth(
            us_geo, locations="StateCode", locationmode="USA-states",
            color="Average_Lead_Time", scope="usa",
            hover_name="State/Province",
            hover_data={"Shipments":True,"Average_Lead_Time":":.1f","Total_Sales":":.2f","StateCode":False},
            title="Average Shipping Lead Time by State"
        )
        fig.update_layout(margin=dict(l=0,r=0,t=45,b=0))
        st.plotly_chart(fig, use_container_width=True)
    non_us = geo[geo["StateCode"].isna()]
    if len(non_us):
        st.caption("Non-US locations are shown separately because the map uses the USA-states geometry.")
        st.dataframe(non_us.round(2), use_container_width=True, hide_index=True)

    factory_table = pd.DataFrame(
        [{"Factory": f, "Latitude": lat, "Longitude": lon} for f,(lat,lon) in FACTORY_COORDS.items()]
    )
    st.markdown("**Factory locations used for route context**")
    st.dataframe(factory_table, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Ship Mode Performance")
    mode = fdf.groupby("Ship Mode", as_index=False).agg(
        Shipments=("Order ID","nunique"),
        Average_Lead_Time=("Shipping Lead Time (Days)","mean"),
        Median_Lead_Time=("Shipping Lead Time (Days)","median"),
        Lead_Time_Std=("Shipping Lead Time (Days)","std"),
        Total_Sales=("Sales","sum"),
    )
    mode["Delay_%"] = fdf.groupby("Ship Mode")["Shipping Lead Time (Days)"].apply(
        lambda x: (x > threshold).mean()*100
    ).reindex(mode["Ship Mode"]).values
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(mode, x="Ship Mode", y="Average_Lead_Time",
                     title="Average Lead Time by Ship Mode",
                     labels={"Average_Lead_Time":"Average lead time (days)"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(mode, x="Ship Mode", y="Delay_%",
                     title=f"Delay Frequency (> {threshold} days)",
                     labels={"Delay_%":"Delayed shipments (%)"})
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(mode.round(2), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Route Drill-Down")
    routes = sorted(fdf["Route"].dropna().unique())
    selected_route = st.selectbox("Select route", routes)
    rdf = fdf[fdf["Route"] == selected_route].copy()
    a, b, c, dcol = st.columns(4)
    a.metric("Orders", f"{rdf['Order ID'].nunique():,}")
    b.metric("Avg lead time", f"{rdf['Shipping Lead Time (Days)'].mean():,.1f} d")
    c.metric("Delay frequency", f"{(rdf['Shipping Lead Time (Days)'] > threshold).mean()*100:,.1f}%")
    dcol.metric("Sales", f"${rdf['Sales'].sum():,.2f}")

    timeline = rdf.sort_values("Order Date").copy()
    fig = px.scatter(
        timeline, x="Order Date", y="Shipping Lead Time (Days)",
        color="Ship Mode", hover_data=["Order ID","Product Name","State/Province","Ship Date"],
        title="Order-Level Shipping Timeline"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        timeline[["Order ID","Order Date","Ship Date","Ship Mode","Factory","State/Province",
                  "Product Name","Sales","Units","Shipping Lead Time (Days)"]].sort_values("Order Date"),
        use_container_width=True, hide_index=True
    )

st.markdown("---")
st.caption("Built as a Data Science Internship project using Python, Pandas, Plotly and Streamlit.")
