"""
app.py - House Price Prediction Streamlit Application

A modern, interactive dashboard that lets users input housing features
and receive an estimated California house price from a pre-trained
Linear Regression model.

Usage:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.inspection import permutation_importance


# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS for a premium look
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ---------- Global ---------- */
    .main { background-color: #0e1117; }

    /* ---------- Metric card ---------- */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .prediction-card h2 {
        color: #ffffff;
        font-size: 2.4rem;
        margin: 0;
        font-weight: 700;
    }
    .prediction-card p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin-top: 0.25rem;
    }

    /* ---------- Info card ---------- */
    .info-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
    }
    .info-card h4 { color: #667eea; margin: 0 0 0.3rem 0; }
    .info-card p  { color: rgba(255,255,255,0.7); margin: 0; font-size: 0.9rem; }

    /* ---------- Section headers ---------- */
    .section-header {
        background: linear-gradient(90deg, rgba(102,126,234,0.15), transparent);
        padding: 0.6rem 1rem;
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper: load the trained model (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    Load the serialized pipeline. If the file is missing or a version mismatch
    causes unpickling to fail, retrain the model on the fly for 100% compatibility.
    """
    model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
    
    # Try loading the existing model first
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            # Silent fallback to retraining if loading fails
            pass
            
    # Retrain on the fly as fallback
    try:
        from sklearn.datasets import fetch_california_housing
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # Load dataset
        housing = fetch_california_housing(as_frame=True)
        X_raw = housing.data
        y = housing.target
        
        # Apply the same feature engineering defined in app.py
        X = engineer_features(X_raw)
        
        # Build and train the pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", HistGradientBoostingRegressor(random_state=42)),
        ])
        pipeline.fit(X, y)
        
        # Save it back locally for subsequent loads if writable
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(pipeline, model_path)
        except Exception:
            pass
            
        return pipeline
    except Exception as retrain_error:
        st.error(f"❌ Failed to load or retrain the model: {str(retrain_error)}")
        st.stop()


# Coordinates for California's two main economic centers
SF_COORD = (37.7749, -122.4194)
LA_COORD = (34.0522, -118.2437)


def engineer_features(df):
    """
    Engineer advanced features from raw California Housing inputs.
    Supports both full DataFrames (training) and single-row DataFrames (inference).
    """
    new_df = df.copy()

    # 1. Proximity to major urban/economic hubs (approx. Euclidean distance)
    new_df["dist_to_SF"] = np.sqrt(
        (new_df["Latitude"] - SF_COORD[0]) ** 2
        + (new_df["Longitude"] - SF_COORD[1]) ** 2
    )
    new_df["dist_to_LA"] = np.sqrt(
        (new_df["Latitude"] - LA_COORD[0]) ** 2
        + (new_df["Longitude"] - LA_COORD[1]) ** 2
    )

    # 2. Structural/demographic ratios
    new_df["bedrms_per_room"] = new_df["AveBedrms"] / new_df["AveRooms"]
    new_df["rooms_per_occup"] = new_df["AveRooms"] / new_df["AveOccup"]

    return new_df


@st.cache_data
def load_dataset():
    """Load the California Housing dataset and engineer features for analytics."""
    housing = fetch_california_housing(as_frame=True)
    df = engineer_features(housing.data)
    df["MedHouseVal"] = housing.target
    
    # We want to return the engineered feature names (excluding target)
    feature_names = list(df.columns.drop("MedHouseVal"))
    return df, feature_names


# ─────────────────────────────────────────────
# Feature descriptions (for user guidance)
# ─────────────────────────────────────────────
FEATURE_INFO = {
    "MedInc":     ("Median Income", "Median income of households in the block group (in $10k)"),
    "HouseAge":   ("House Age", "Median house age in the block group (years)"),
    "AveRooms":   ("Average Rooms", "Average number of rooms per household"),
    "AveBedrms":  ("Average Bedrooms", "Average number of bedrooms per household"),
    "Population": ("Population", "Block group population"),
    "AveOccup":   ("Average Occupancy", "Average number of occupants per household"),
    "Latitude":   ("Latitude", "Latitude of the block group"),
    "Longitude":  ("Longitude", "Longitude of the block group"),
}

# Sensible default values derived from dataset medians / means
DEFAULTS = {
    "MedInc": 3.88,
    "HouseAge": 29.0,
    "AveRooms": 5.43,
    "AveBedrms": 1.05,
    "Population": 1425.0,
    "AveOccup": 3.07,
    "Latitude": 35.63,
    "Longitude": -119.57,
}


# ─────────────────────────────────────────────
# Load assets
# ─────────────────────────────────────────────
model = load_model()
df, feature_names = load_dataset()


# ─────────────────────────────────────────────
# Sidebar: About + Feature Guide
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 About This App")
    st.markdown(
        "Predict California house prices using a **Random Forest Regressor** model "
        "trained on the [California Housing dataset]"
        "(https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)."
    )
    st.divider()
    st.markdown("### 📖 Feature Guide")
    for key, (title, desc) in FEATURE_INFO.items():
        st.markdown(
            f'<div class="info-card"><h4>{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────
st.markdown("# 🏠 House Price Prediction")
st.markdown(
    "Enter the housing features below and click **Predict** to estimate "
    "the median house value for a California block group."
)
st.divider()

# ─────────────────────────────────────────────
# Input Form (two-column layout)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📝 Enter Housing Features</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input(
        "💰 Median Income ($10k)", min_value=0.0, max_value=20.0,
        value=DEFAULTS["MedInc"], step=0.1, format="%.2f",
        help=FEATURE_INFO["MedInc"][1],
    )
    ave_rooms = st.number_input(
        "🚪 Average Rooms", min_value=0.5, max_value=50.0,
        value=DEFAULTS["AveRooms"], step=0.1, format="%.2f",
        help=FEATURE_INFO["AveRooms"][1],
    )
    population = st.number_input(
        "👥 Population", min_value=1.0, max_value=40000.0,
        value=DEFAULTS["Population"], step=10.0, format="%.0f",
        help=FEATURE_INFO["Population"][1],
    )
    latitude = st.number_input(
        "🌐 Latitude", min_value=32.0, max_value=42.0,
        value=DEFAULTS["Latitude"], step=0.01, format="%.2f",
        help=FEATURE_INFO["Latitude"][1],
    )

with col2:
    house_age = st.number_input(
        "🏗️ House Age (years)", min_value=1.0, max_value=52.0,
        value=DEFAULTS["HouseAge"], step=1.0, format="%.0f",
        help=FEATURE_INFO["HouseAge"][1],
    )
    ave_bedrms = st.number_input(
        "🛏️ Average Bedrooms", min_value=0.3, max_value=30.0,
        value=DEFAULTS["AveBedrms"], step=0.1, format="%.2f",
        help=FEATURE_INFO["AveBedrms"][1],
    )
    ave_occup = st.number_input(
        "🏘️ Average Occupancy", min_value=0.5, max_value=20.0,
        value=DEFAULTS["AveOccup"], step=0.1, format="%.2f",
        help=FEATURE_INFO["AveOccup"][1],
    )
    longitude = st.number_input(
        "🌐 Longitude", min_value=-125.0, max_value=-114.0,
        value=DEFAULTS["Longitude"], step=0.01, format="%.2f",
        help=FEATURE_INFO["Longitude"][1],
    )

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
st.divider()

predict_col, map_col = st.columns([1, 1])

with predict_col:
    if st.button("🔮 Predict House Price", use_container_width=True, type="primary"):
        # Build DataFrame with the exact column names the model was trained on
        input_df = pd.DataFrame([[
            med_inc, house_age, ave_rooms, ave_bedrms,
            population, ave_occup, latitude, longitude,
        ]], columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])

        # Basic validation
        if ave_bedrms > ave_rooms:
            st.warning("⚠️ Average bedrooms exceeds average rooms — please double-check your inputs.")

        # Apply the exact same feature engineering
        engineered_df = engineer_features(input_df)
        
        prediction = model.predict(engineered_df)[0]
        price_usd = prediction * 100_000  # Convert from $100k units to dollars

        st.markdown(
            f"""
            <div class="prediction-card">
                <p>Estimated Median House Value</p>
                <h2>${price_usd:,.0f}</h2>
                <p>Model output: {prediction:.4f} (×$100k)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with map_col:
    st.markdown('<div class="section-header">📍 Location Preview</div>', unsafe_allow_html=True)
    map_df = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
    st.map(map_df, zoom=6)


# ─────────────────────────────────────────────
# Analytics & Visualizations
# ─────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-header">📊 Model Analytics & Visualizations</div>', unsafe_allow_html=True)

viz_tab1, viz_tab2 = st.tabs(["📈 Feature Importance", "🔥 Correlation Heatmap"])

# --- Tab 1: Feature Importance (permutation importance) ---
with viz_tab1:
    st.markdown("#### Permutation Feature Importance")
    st.caption(
        "Permutation importance measures the increase in the prediction error of the model "
        "after we shuffle the values of a feature, which breaks the relationship between the "
        "feature and the target. A larger decrease in R2 score indicates a more important feature."
    )

    # We run permutation importance on a sample of the data to keep it fast and cached
    @st.cache_data
    def compute_permutation_importance(_model, _df, feature_names):
        # Sample 1000 records for speed
        sample_df = _df.sample(n=min(1000, len(_df)), random_state=42)
        X_sample = sample_df[feature_names]
        y_sample = sample_df["MedHouseVal"]
        
        # Run permutation importance using the pipeline (which handles scaling)
        result = permutation_importance(
            _model, X_sample, y_sample, n_repeats=5, random_state=42, n_jobs=-1
        )
        return result.importances_mean

    importance_vals = compute_permutation_importance(model, df, feature_names)
    
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_vals,
    }).sort_values("Importance", ascending=True)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    colors = ["#764ba2" if c < 0 else "#667eea" for c in coef_df["Importance"]]
    ax1.barh(coef_df["Feature"], coef_df["Importance"], color=colors, edgecolor="none", height=0.6)
    ax1.set_xlabel("Mean Decrease in R2 Score", fontsize=11)
    ax1.set_title("Permutation Feature Importance (on Test Sample)", fontsize=13, fontweight="bold")
    ax1.axvline(0, color="white", linewidth=0.8, alpha=0.5)
    ax1.set_facecolor("#0e1117")
    fig1.patch.set_facecolor("#0e1117")
    ax1.tick_params(colors="white")
    ax1.xaxis.label.set_color("white")
    ax1.title.set_color("white")
    for spine in ax1.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig1)

# --- Tab 2: Correlation Heatmap ---
with viz_tab2:
    st.markdown("#### Feature Correlation Heatmap")
    st.caption("Pearson correlation between all features and the target variable (MedHouseVal).")

    corr = df.corr(numeric_only=True)
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=12, center="dark")
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        linewidths=0.5,
        linecolor="#1a1a2e",
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        ax=ax2,
    )
    ax2.set_title("Correlation Heatmap", fontsize=14, fontweight="bold", color="white")
    ax2.set_facecolor("#0e1117")
    fig2.patch.set_facecolor("#0e1117")
    ax2.tick_params(colors="white")
    cbar = ax2.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.set_label("Correlation", color="white")
    plt.tight_layout()
    st.pyplot(fig2)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:rgba(255,255,255,0.35); font-size:0.85rem;'>"
    "Built with ❤️ using Streamlit & Scikit-learn · California Housing Dataset"
    "</p>",
    unsafe_allow_html=True,
)
