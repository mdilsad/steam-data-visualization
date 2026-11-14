from __future__ import annotations

from typing import NamedTuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MAX_RF_ROWS = 25_000
MAX_SCATTER_POINTS = 10_000
ELBOW_SAMPLE_SIZE = 15_000
SILHOUETTE_SAMPLE_SIZE = 3_000
PERMUTATION_REPEATS = 5
ForestEstimator = Union[Pipeline, TransformedTargetRegressor]

KMEANS_FEATURES = (
    "price",
    "review_score",
    "positive_ratio",
    "is_free",
    "release_year",
    "concurrent_users",
    "recommendations",
    "metacritic_score",
)
KMEANS_DEFAULT_FEATURES = (
    "price",
    "review_score",
    "positive_ratio",
    "is_free",
    "release_year",
    "concurrent_users",
)
COUNT_FEATURES = {"recommendations", "concurrent_users"}
RF_FEATURES = (
    "price",
    "total",
    "review_score",
    "metacritic_score",
    "is_free",
    "release_year",
    "concurrent_users",
    "recommendations",
)
RF_DEFAULT_FEATURES = (
    "price",
    "total",
    "review_score",
    "is_free",
    "release_year",
    "concurrent_users",
)
FEATURE_LABELS = {
    "price": "Price",
    "review_score": "Review score",
    "positive_ratio": "Positive review ratio",
    "is_free": "Free-to-play",
    "release_year": "Release year",
    "concurrent_users": "Concurrent users",
    "recommendations": "Recommendations",
    "metacritic_score": "Metacritic score",
    "total": "Total reviews",
}


class KMeansResult(NamedTuple):
    scatter_data: pd.DataFrame
    profile: pd.DataFrame
    standardized_medians: pd.DataFrame
    elbow_data: pd.DataFrame
    silhouette: float
    silhouette_sample_size: int
    pca_variance: tuple[float, float]
    centers: pd.DataFrame
    clustered_game_count: int


class RandomForestResult(NamedTuple):
    test_results: pd.DataFrame
    comparison: pd.DataFrame
    impurity_importance: pd.DataFrame
    permutation_importance: pd.DataFrame
    selected_model: str
    selected_metrics: dict[str, float]
    train_count: int
    test_count: int
    sampled: bool


def _available_features(df: pd.DataFrame, candidates: tuple[str, ...]) -> list[str]:
    """Return candidate columns that contain at least one usable numeric value."""
    available = []
    for column in candidates:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).any():
            available.append(column)
    return available


@st.cache_data(show_spinner=False)
def _prepare_kmeans_data(
    df: pd.DataFrame,
    selected_features: tuple[str, ...],
) -> tuple[pd.DataFrame, np.ndarray, PCA]:
    if len(selected_features) < 2:
        raise ValueError("Select at least two numeric features to run PCA and clustering.")
    if "app_id" in selected_features or "estimated_owners" in selected_features:
        raise ValueError("Identifiers and estimated_owners cannot be clustering features.")

    raw_features = pd.DataFrame(index=df.index)
    for column in selected_features:
        if column not in df.columns:
            continue
        raw_features[column] = pd.to_numeric(df[column], errors="coerce")
    raw_features = raw_features.replace([np.inf, -np.inf], np.nan)
    raw_features = raw_features.dropna(axis=1, how="all")
    if raw_features.shape[1] < 2:
        raise ValueError("At least two selected columns must contain numeric values.")

    # A zero Steam review score means no score was assigned for games with very few reviews.
    if "review_score" in raw_features.columns:
        raw_features.loc[raw_features["review_score"] == 0, "review_score"] = np.nan

    valid_rows = raw_features.notna().any(axis=1)
    raw_features = raw_features.loc[valid_rows]
    if len(raw_features) < 2:
        raise ValueError("There are not enough usable games for clustering.")

    transformed = raw_features.copy()
    skewed_features = COUNT_FEATURES.union({"price"}).intersection(transformed.columns)
    for column in skewed_features:
        transformed[column] = np.log1p(transformed[column].clip(lower=0))

    # Imputation is fitted on available values; no rows are discarded for price outliers.
    imputed = SimpleImputer(strategy="median").fit_transform(transformed)
    scaled = StandardScaler().fit_transform(imputed)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(scaled)
    return raw_features, scaled, pca


@st.cache_data(show_spinner=False)
def _calculate_elbow(scaled_features: np.ndarray) -> pd.DataFrame:
    sample_count = min(ELBOW_SAMPLE_SIZE, len(scaled_features))
    sample_indices = np.random.default_rng(RANDOM_STATE).choice(
        len(scaled_features),
        size=sample_count,
        replace=False,
    )
    sample = scaled_features[sample_indices]
    maximum_k = min(8, sample_count - 1)
    results = []
    for cluster_count in range(2, maximum_k + 1):
        model = KMeans(
            n_clusters=cluster_count,
            random_state=RANDOM_STATE,
            n_init=5,
            max_iter=200,
        ).fit(sample)
        results.append({"K": cluster_count, "Inertia": float(model.inertia_)})
    return pd.DataFrame(results)


def _stratified_sample_indices(
    labels: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    if len(labels) <= sample_size:
        return np.arange(len(labels))
    rng = np.random.default_rng(RANDOM_STATE)
    clusters = np.unique(labels)
    representatives = np.array(
        [rng.choice(np.flatnonzero(labels == cluster)) for cluster in clusters]
    )
    remaining = np.setdiff1d(np.arange(len(labels)), representatives, assume_unique=False)
    additional = rng.choice(
        remaining,
        size=sample_size - len(representatives),
        replace=False,
    )
    return np.concatenate([representatives, additional])


@st.cache_data(show_spinner=False)
def _fit_kmeans(
    df: pd.DataFrame,
    selected_features: tuple[str, ...],
    cluster_count: int,
) -> KMeansResult:
    """Cluster all usable rows and calculate sampled evaluation diagnostics."""
    raw_features, scaled, pca = _prepare_kmeans_data(df, selected_features)
    if cluster_count < 2 or cluster_count > min(8, len(scaled)):
        raise ValueError("Choose a cluster count between 2 and the number of usable games.")

    model = KMeans(
        n_clusters=cluster_count,
        random_state=RANDOM_STATE,
        n_init=10,
    ).fit(scaled)
    labels = model.labels_
    cluster_names = [f"Cluster {label + 1}" for label in labels]
    pca_coordinates = pca.transform(scaled)

    scatter_data = pd.DataFrame(
        {
            "PC 1": pca_coordinates[:, 0],
            "PC 2": pca_coordinates[:, 1],
            "Cluster": cluster_names,
        },
        index=raw_features.index,
    )
    if "name" in df.columns:
        scatter_data["Game"] = df.loc[raw_features.index, "name"].fillna("Unknown title")

    cluster_sizes = (
        pd.Series(cluster_names, index=raw_features.index)
        .value_counts()
        .reindex([f"Cluster {i + 1}" for i in range(cluster_count)], fill_value=0)
    )
    if int(cluster_sizes.sum()) != len(raw_features) or cluster_sizes.eq(0).any():
        raise RuntimeError("K-Means returned an empty cluster or inconsistent cluster counts.")
    profile = raw_features.assign(Cluster=cluster_names).groupby("Cluster").median()
    profile.insert(0, "Games", cluster_sizes)

    medians = profile.drop(columns="Games")
    standard_deviations = medians.std(axis=0, ddof=0).replace(0, 1)
    standardized_medians = (medians - medians.mean(axis=0)) / standard_deviations
    standardized_medians.index.name = "Cluster"
    centers_pca = pca.transform(model.cluster_centers_)
    centers = pd.DataFrame(
        {
            "PC 1": centers_pca[:, 0],
            "PC 2": centers_pca[:, 1],
            "Cluster": [f"Cluster {i + 1}" for i in range(cluster_count)],
            "Games": cluster_sizes.to_numpy(),
        }
    )

    silhouette_indices = _stratified_sample_indices(
        labels,
        min(SILHOUETTE_SAMPLE_SIZE, len(labels)),
    )
    silhouette_value = silhouette_score(
        scaled[silhouette_indices],
        labels[silhouette_indices],
    )
    variance = pca.explained_variance_ratio_
    return KMeansResult(
        scatter_data=scatter_data,
        profile=profile,
        standardized_medians=standardized_medians,
        elbow_data=_calculate_elbow(scaled),
        silhouette=float(silhouette_value),
        silhouette_sample_size=len(silhouette_indices),
        pca_variance=(float(variance[0]), float(variance[1])),
        centers=centers,
        clustered_game_count=len(raw_features),
    )


@st.cache_data(show_spinner=False)
def _fit_random_forest(
    df: pd.DataFrame,
    selected_features: tuple[str, ...],
) -> RandomForestResult:
    """Compare raw/log-target forests and a baseline on one held-out test split."""
    target = "estimated_owners"
    if any(
        feature == target or feature.startswith(f"{target}_") or feature == "app_id"
        for feature in selected_features
    ):
        raise ValueError("The target, its transformations, and identifiers cannot be model inputs.")
    if not selected_features:
        raise ValueError("Select at least one numeric input feature.")
    if target not in df.columns:
        raise ValueError("The dataset does not contain the estimated_owners target column.")

    usable_features = [
        column for column in selected_features if column in df.columns
    ]
    if not usable_features:
        raise ValueError("None of the selected features are present in the dataset.")

    source = df[usable_features + [target]].apply(pd.to_numeric, errors="coerce")
    source = source.replace([np.inf, -np.inf], np.nan)
    source = source.dropna(subset=[target])
    source = source.loc[source[target] >= 0]
    usable_features = [
        column for column in usable_features if source[column].notna().any()
    ]
    if not usable_features:
        raise ValueError("The selected features do not contain usable numeric values.")
    source = source[usable_features + [target]]
    if len(source) < 10:
        raise ValueError("At least 10 games with estimated owners are needed to train and test.")

    available_rows = len(source)
    sampled = available_rows > MAX_RF_ROWS
    if sampled:
        source = source.sample(n=MAX_RF_ROWS, random_state=RANDOM_STATE)

    x = source[usable_features]
    y = source[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    def make_forest(log_target: bool) -> ForestEstimator:
        forest = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestRegressor(
                n_estimators=150,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )
        if log_target:
            return TransformedTargetRegressor(
                regressor=forest,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=False,
            )
        return forest

    def score_predictions(predictions: np.ndarray) -> dict[str, float]:
        return {
            "MAE": float(mean_absolute_error(y_test, predictions)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "R²": float(r2_score(y_test, predictions)),
        }

    baseline = DummyRegressor(strategy="median").fit(x_train, y_train)
    comparison_rows = [
        {
            "Model": "Training-median baseline",
            **score_predictions(np.asarray(baseline.predict(x_test))),
        }
    ]
    main_models: list[
        tuple[str, ForestEstimator, np.ndarray, dict[str, float]]
    ] = []
    selected_variant = (
        "With total reviews" if "total" in usable_features else "Selected features"
    )
    feature_variants = [(selected_variant, tuple(usable_features))]
    without_total = tuple(feature for feature in usable_features if feature != "total")
    if "total" in usable_features and without_total:
        feature_variants.append(("Without total reviews", without_total))

    for variant, model_features in feature_variants:
        for log_target in (False, True):
            target_description = "log1p target" if log_target else "original target"
            model_name = f"Random Forest ({target_description}; {variant.lower()})"
            model = make_forest(log_target)
            model.fit(x_train[list(model_features)], y_train)
            predictions = np.asarray(model.predict(x_test[list(model_features)]))
            metrics = score_predictions(predictions)
            comparison_rows.append(
                {
                    "Model": model_name,
                    **metrics,
                    "Input features": len(model_features),
                }
            )
            if variant == selected_variant:
                main_models.append((model_name, model, predictions, metrics))

    selected_model, fitted_model, predictions, selected_metrics = min(
        main_models,
        key=lambda item: item[3]["MAE"],
    )
    test_results = pd.DataFrame(
        {
            "Actual": y_test.to_numpy(),
            "Predicted": predictions,
            "Game": (
                df.loc[y_test.index, "name"].fillna("Unknown title").to_numpy()
                if "name" in df.columns
                else y_test.index.astype(str)
            ),
        },
        index=y_test.index,
    )

    base_pipeline = (
        fitted_model.regressor_
        if isinstance(fitted_model, TransformedTargetRegressor)
        else fitted_model
    )
    imputer = base_pipeline.named_steps["simpleimputer"]
    forest = base_pipeline.named_steps["randomforestregressor"]
    model_features = imputer.get_feature_names_out(usable_features)
    impurity = pd.DataFrame(
        {"Feature": model_features, "Importance": forest.feature_importances_}
    ).sort_values("Importance", ascending=False)

    baseline_mae = mean_absolute_error(y_test, predictions)
    rng = np.random.default_rng(RANDOM_STATE)
    permutation_rows = []
    for feature in usable_features:
        increases = []
        original_values = x_test[feature].to_numpy(copy=True)
        for _ in range(PERMUTATION_REPEATS):
            shuffled_test = x_test.copy()
            shuffled_test[feature] = rng.permutation(original_values)
            shuffled_predictions = np.asarray(fitted_model.predict(shuffled_test))
            increases.append(
                float(mean_absolute_error(y_test, shuffled_predictions) - baseline_mae)
            )
        permutation_rows.append(
            {
                "Feature": feature,
                "Importance": float(np.mean(increases)),
                "Std": float(np.std(increases)),
            }
        )
    permutation_results = pd.DataFrame(
        permutation_rows
    ).sort_values("Importance", ascending=False)
    return RandomForestResult(
        test_results=test_results,
        comparison=pd.DataFrame(comparison_rows),
        impurity_importance=impurity,
        permutation_importance=permutation_results,
        selected_model=selected_model,
        selected_metrics=selected_metrics,
        train_count=len(x_train),
        test_count=len(x_test),
        sampled=sampled,
    )


@st.cache_data(show_spinner=False)
def _feature_relationships(
    df: pd.DataFrame,
    selected_features: tuple[str, ...],
) -> pd.DataFrame:
    """Summarize pairwise-available Spearman correlations with the estimate."""
    if "estimated_owners" not in df.columns:
        return pd.DataFrame(columns=["Feature", "Spearman rho", "Games"])
    columns = [
        column for column in selected_features + ("estimated_owners",)
        if column in df.columns
    ]
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    rows = []
    for feature in selected_features:
        if feature not in numeric.columns:
            continue
        pair = numeric[[feature, "estimated_owners"]].dropna()
        if pair.empty:
            continue
        rows.append(
            {
                "Feature": FEATURE_LABELS.get(feature, feature),
                "Spearman rho": pair[feature].corr(
                    pair["estimated_owners"], method="spearman"
                ),
                "Games": len(pair),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "Spearman rho", key=lambda values: values.abs(), ascending=False
    )


def _display_feature_name(feature: str) -> str:
    missing_prefix = "missingindicator_"
    if feature.startswith(missing_prefix):
        source = feature[len(missing_prefix) :]
        return f"Missing: {FEATURE_LABELS.get(source, source)}"
    return FEATURE_LABELS.get(feature, feature)


def _render_kmeans(df: pd.DataFrame) -> None:
    st.markdown(
        "K-Means groups games by similarity across selected numeric characteristics. "
        "PCA projects those characteristics into two dimensions for display. "
        "Cluster labels describe patterns in the selected features, not game genres."
    )
    available = _available_features(df, KMEANS_FEATURES)
    defaults = [feature for feature in KMEANS_DEFAULT_FEATURES if feature in available]
    controls = st.columns([1, 3])
    cluster_count = controls[0].slider(
        "Number of clusters (K)",
        min_value=2,
        max_value=8,
        value=4,
        key="ml_kmeans_cluster_count",
    )
    selected_features = controls[1].multiselect(
        "Numeric features",
        options=available,
        default=defaults,
        format_func=lambda feature: FEATURE_LABELS.get(feature) or feature,
        key="ml_kmeans_features",
    )
    st.caption(
        "Identifiers and estimated owners are excluded. Price and skewed count inputs "
        "use log1p before StandardScaler; original values are retained for summaries. "
        "Missing values are median-imputed after transformation. A review_score of 0 "
        "is treated as unavailable in this dataset (fewer than 10 reviews), not as a "
        "zero score."
    )
    if "concurrent_users" in df.columns:
        zeros = int(pd.to_numeric(df["concurrent_users"], errors="coerce").eq(0).sum())
        remaining_missing = int(df["concurrent_users"].isna().sum())
        st.caption(
            f"concurrent_users contains {zeros:,} zeros and {remaining_missing:,} "
            "remaining missing values. The preprocessing step maps unavailable source "
            "values to zero, so the cleaned file cannot distinguish those from genuine "
            "zero counts; zeros are preserved, and remaining missing values are imputed."
        )

    if len(selected_features) < 2:
        st.info("Select at least two numeric features to display the clustering analysis.")
        return
    try:
        result = _fit_kmeans(
            df,
            tuple(selected_features),
            cluster_count,
        )
    except (ValueError, RuntimeError) as error:
        st.warning(str(error))
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Games clustered", f"{result.clustered_game_count:,}")
    metric_columns[1].metric("Silhouette score", f"{result.silhouette:.3f}")
    metric_columns[2].metric("PCA component 1", f"{result.pca_variance[0]:.1%}")
    metric_columns[3].metric("PCA component 2", f"{result.pca_variance[1]:.1%}")
    st.caption(
        f"Silhouette score uses a reproducible stratified sample of "
        f"{result.silhouette_sample_size:,} games. Cluster sizes sum to "
        f"{int(result.profile['Games'].sum()):,}; all {cluster_count} clusters are non-empty."
    )

    chart_sample = result.scatter_data
    if len(chart_sample) > MAX_SCATTER_POINTS:
        chart_sample = chart_sample.sample(
            n=MAX_SCATTER_POINTS,
            random_state=RANDOM_STATE,
        )
    if "Game" in chart_sample.columns:
        scatter = px.scatter(
            chart_sample,
            x="PC 1",
            y="PC 2",
            color="Cluster",
            hover_name="Game",
            title="PCA Projection of K-Means Clusters",
            opacity=0.7,
        )
    else:
        scatter = px.scatter(
            chart_sample,
            x="PC 1",
            y="PC 2",
            color="Cluster",
            title="PCA Projection of K-Means Clusters",
            opacity=0.7,
        )
    scatter.add_trace(
        go.Scatter(
            x=result.centers["PC 1"],
            y=result.centers["PC 2"],
            mode="markers+text",
            text=result.centers["Cluster"],
            textposition="top center",
            customdata=result.centers[["Games"]],
            name="Cluster centers",
            marker={
                "symbol": "x",
                "size": 15,
                "color": "#111827",
                "line": {"width": 2, "color": "#ffffff"},
            },
            hovertemplate=(
                "%{text}<br>Center<br>Games: %{customdata[0]:,}<extra></extra>"
            ),
        )
    )
    scatter.update_layout(
        xaxis_title="Principal component 1",
        yaxis_title="Principal component 2",
    )
    st.plotly_chart(scatter, use_container_width=True)
    if len(result.scatter_data) > len(chart_sample):
        st.caption(
            f"The PCA chart samples {len(chart_sample):,} of {len(result.scatter_data):,} "
            "clustered games for display; clustering uses every usable game."
        )

    elbow_chart = px.line(
        result.elbow_data,
        x="K",
        y="Inertia",
        markers=True,
        title=f"Elbow Method (reproducible sample of {min(ELBOW_SAMPLE_SIZE, result.clustered_game_count):,} games)",
    )
    elbow_chart.add_vline(
        x=cluster_count,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Selected K = {cluster_count}",
    )
    elbow_chart.update_layout(
        xaxis_title="Number of clusters (K)",
        yaxis_title="Within-cluster sum of squares (inertia)",
    )
    st.plotly_chart(elbow_chart, use_container_width=True)
    st.caption(
        "The Elbow curve is calculated on a fixed random sample to keep interaction "
        "responsive; the selected K-Means model is fitted on all usable games."
    )

    distribution = result.profile["Games"].rename_axis("Cluster").reset_index()
    distribution_chart = px.bar(
        distribution,
        x="Cluster",
        y="Games",
        color="Cluster",
        title="Cluster Distribution",
        text_auto=True,
    )
    distribution_chart.update_layout(showlegend=False, yaxis_title="Number of games")
    st.plotly_chart(distribution_chart, use_container_width=True)

    st.subheader("Cluster Characteristics")
    heatmap = px.imshow(
        result.standardized_medians,
        x=[
            FEATURE_LABELS.get(column, column)
            for column in result.standardized_medians.columns
        ],
        y=result.standardized_medians.index,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        aspect="auto",
        title="Relative Feature Medians by Cluster",
        labels={"x": "Feature", "y": "Cluster", "color": "Standardized median"},
    )
    st.plotly_chart(heatmap, use_container_width=True)
    cluster_sizes = result.profile["Games"]
    variable_features = result.standardized_medians.std(axis=0).loc[lambda values: values > 0]
    for cluster in result.standardized_medians.index:
        row = result.standardized_medians.loc[cluster, variable_features.index]
        if row.empty:
            st.write(
                f"**{cluster}** contains {int(cluster_sizes.loc[cluster]):,} games; "
                "selected feature medians do not differ across clusters."
            )
            continue
        highest_column = str(row.idxmax())
        lowest_column = str(row.idxmin())
        highest_feature = FEATURE_LABELS.get(highest_column, highest_column)
        lowest_feature = FEATURE_LABELS.get(lowest_column, lowest_column)
        highest_median = result.profile.loc[cluster, highest_column]
        lowest_median = result.profile.loc[cluster, lowest_column]
        st.write(
            f"**{cluster}** contains {int(cluster_sizes.loc[cluster]):,} games "
            f"({cluster_sizes.loc[cluster] / result.clustered_game_count:.1%}). "
            f"Its most elevated selected-feature median is **{highest_feature}** "
            f"({highest_median:,.2f}); its lowest relative median is **{lowest_feature}** "
            f"({lowest_median:,.2f}), compared with the other clusters."
        )
    display_profile = result.profile.rename(
        columns={
            column: FEATURE_LABELS.get(column, column)
            for column in result.profile.columns
        }
    )
    st.dataframe(display_profile, use_container_width=True)
    st.caption(
        "Interpretations compare selected-feature medians only; they do not assign "
        "game types or imply causation. The table reports medians in original units."
    )

    if "price" in df.columns:
        prices = pd.to_numeric(df["price"], errors="coerce").dropna()
        if not prices.empty:
            with st.expander("Price distribution and retained high-price games"):
                price_metrics = st.columns(4)
                price_metrics[0].metric("Price skewness", f"{prices.skew():.1f}")
                price_metrics[1].metric("Median price", f"{prices.median():,.2f}")
                price_metrics[2].metric("99th percentile", f"{prices.quantile(0.99):,.2f}")
                price_metrics[3].metric("Maximum price", f"{prices.max():,.2f}")
                top_indices = prices.nlargest(10).index
                display_columns = [
                    column for column in ("name", "price") if column in df.columns
                ]
                price_outliers = df.loc[top_indices, display_columns].copy()
                price_outliers["Current cluster"] = (
                    result.scatter_data["Cluster"]
                    .reindex(price_outliers.index)
                    .fillna("Not clustered")
                )
                st.dataframe(price_outliers, hide_index=True, use_container_width=True)
                st.caption(
                    "The two highest-priced games formed the 2-game cluster in the "
                    "previous K=5 run with raw price. They are retained here with their "
                    "original prices; log1p is applied only to clustering inputs."
                )


def _render_random_forest(df: pd.DataFrame) -> None:
    st.markdown(
        "Random Forest Regression estimates the dataset's estimated_owners midpoint "
        "from game characteristics. Original-target and log1p-target models use the "
        "same reproducible held-out split; all displayed metrics are in the original "
        "owner-estimate scale."
    )
    st.warning(
        "estimated_owners is an estimate, not actual sales or verified player counts. "
        "In the cleaned dataset, preprocess.py converts SteamSpy owner-range bounds "
        "to their arithmetic midpoint (for example, 10–20 million becomes 15 million). "
        "The model predicts that midpoint estimate, not an exact owner count."
    )
    if "estimated_owners" not in df.columns:
        st.warning("The dataset does not contain estimated_owners.")
        return

    target_values = pd.to_numeric(df["estimated_owners"], errors="coerce")
    target_values = target_values.replace([np.inf, -np.inf], np.nan).dropna()
    target_values = target_values.loc[target_values >= 0]
    if target_values.empty:
        st.info("No non-negative estimated_owners values are available.")
        return

    target_metrics = st.columns(4)
    target_metrics[0].metric("Available estimates", f"{len(target_values):,}")
    target_metrics[1].metric("Target skewness", f"{target_values.skew():.1f}")
    target_metrics[2].metric("Median estimate", f"{target_values.median():,.0f}")
    target_metrics[3].metric("Maximum estimate", f"{target_values.max():,.0f}")
    transformed_target = np.log1p(target_values)
    distribution = px.histogram(
        x=transformed_target,
        nbins=50,
        title="Estimated Owners Distribution (log1p display scale)",
        labels={"x": "log1p(estimated_owners)", "count": "Games"},
    )
    tick_raw_values = np.geomspace(
        max(1.0, float(target_values.min())),
        float(target_values.max()),
        num=6,
    )
    distribution.update_xaxes(
        tickmode="array",
        tickvals=np.log1p(tick_raw_values),
        ticktext=[f"{value:,.0f}" for value in tick_raw_values],
        title_text="Estimated owners (ticks shown in original estimate units)",
    )
    distribution.update_yaxes(title_text="Number of games")
    st.plotly_chart(distribution, use_container_width=True)
    st.caption(
        f"The 99th percentile is {target_values.quantile(0.99):,.0f}. The log1p "
        "transformation above is for display only; no high-owner games are removed."
    )

    available = _available_features(df, RF_FEATURES)
    defaults = [feature for feature in RF_DEFAULT_FEATURES if feature in available]
    selected_features = st.multiselect(
        "Numeric model features",
        options=available,
        default=defaults,
        format_func=lambda feature: FEATURE_LABELS.get(feature) or feature,
        key="ml_rf_features",
    )
    st.caption(
        "Selected inputs: "
        + (
            ", ".join(FEATURE_LABELS.get(feature, feature) for feature in selected_features)
            if selected_features
            else "none"
        )
    )

    if "recommendations" in df.columns:
        missing_recommendations = df["recommendations"].isna().mean()
        st.caption(
            "The default inputs use total reviews instead of positive and negative "
            "counts (total is their exact sum). Positive ratio is also omitted because "
            "it is computed from review counts. Recommendations are optional: "
            f"{missing_recommendations:.1%} of rows are missing them, and they overlap "
            "strongly with review counts. Metacritic scores are optional and have "
            "substantial missingness. app_id, names, free text, estimated_owners, and "
            "transformations of the target are never inputs."
        )
    else:
        st.caption(
            "Positive ratio is computed from review counts, so the model uses total "
            "reviews instead of overlapping review-count fields. Identifiers, names, "
            "free-text fields, and estimated_owners are excluded from model inputs."
        )

    if not selected_features:
        st.info("Select at least one numeric feature to train the regression model.")
        return
    if "estimated_owners" in selected_features:
        st.error("estimated_owners cannot be used as an input feature.")
        return
    try:
        result = _fit_random_forest(df, tuple(selected_features))
    except ValueError as error:
        st.warning(str(error))
        return

    st.subheader("Model Performance Summary")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected model", result.selected_model)
    metric_columns[1].metric("MAE", f"{result.selected_metrics['MAE']:,.0f}")
    metric_columns[2].metric("RMSE", f"{result.selected_metrics['RMSE']:,.0f}")
    metric_columns[3].metric("R² score", f"{result.selected_metrics['R²']:.3f}")
    st.caption(
        f"The selected model is chosen by the lower held-out MAE in original estimate "
        f"units. Training / test games: {result.train_count:,} / {result.test_count:,}."
    )
    if result.selected_metrics["R²"] > 0:
        st.caption(
            f"On this held-out test set, R² = {result.selected_metrics['R²']:.3f}; "
            "this is predictive association, not evidence of causation or exact owners."
        )
    else:
        st.caption(
            f"On this held-out test set, R² = {result.selected_metrics['R²']:.3f}; "
            "the model does not improve on a test-set mean baseline by this metric."
        )
    if result.sampled:
        st.caption(
            f"A reproducible random sample of up to {MAX_RF_ROWS:,} eligible games was "
            "used to keep training practical; the 80/20 split was made after sampling."
        )
    comparison = result.comparison.copy()
    for metric in ("MAE", "RMSE"):
        comparison[metric] = comparison[metric].map(lambda value: f"{value:,.0f}")
    comparison["R²"] = comparison["R²"].map(lambda value: f"{value:.3f}")
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    if "total" not in selected_features:
        st.info(
            "Select Total reviews to calculate the paired comparison with and without "
            "this feature."
        )
    elif len(selected_features) == 1:
        st.info(
            "Select at least one additional feature to calculate the model without "
            "Total reviews."
        )
    st.caption(
        "The training-median baseline and all model variants use the exact same "
        "sample, split, and held-out test rows. The log-target model applies expm1 "
        "before scoring so every metric remains in owner-estimate units."
    )
    selected_rmse_to_mae = (
        result.selected_metrics["RMSE"] / result.selected_metrics["MAE"]
        if result.selected_metrics["MAE"] > 0
        else float("nan")
    )
    st.caption(
        f"RMSE is {selected_rmse_to_mae:.1f}× MAE for the selected model because "
        "squared errors penalize a few large misses much more. The target distribution "
        "is strongly right-skewed, and legitimate high-owner estimates are retained."
    )

    log_axis = st.checkbox(
        "Use logarithmic axes for actual vs predicted",
        value=False,
        key="ml_rf_log_axes",
    )
    plot_data = result.test_results
    if log_axis:
        positive_values = plot_data["Actual"].gt(0) & plot_data["Predicted"].gt(0)
        excluded_count = int((~positive_values).sum())
        plot_data = plot_data.loc[positive_values]
        st.caption(
            f"Log axes omit {excluded_count:,} test rows with non-positive actual or "
            "predicted estimates; values are not shifted to make them positive."
        )
    if plot_data.empty:
        st.info("No positive actual/predicted pairs are available for logarithmic axes.")
    else:
        actual_predicted = px.scatter(
            plot_data,
            x="Actual",
            y="Predicted",
            hover_name="Game",
            hover_data={"Actual": ":,.0f", "Predicted": ":,.0f"},
            title="Actual vs Predicted Estimated Owners",
            opacity=0.55,
            labels={
                "Actual": "Actual estimated_owners midpoint",
                "Predicted": "Predicted estimated_owners",
            },
        )
        line_min = (
            float(min(plot_data["Actual"].min(), plot_data["Predicted"].min()))
            if log_axis
            else 0.0
        )
        line_max = float(max(plot_data["Actual"].max(), plot_data["Predicted"].max()))
        actual_predicted.add_trace(
            go.Scatter(
                x=[line_min, line_max],
                y=[line_min, line_max],
                mode="lines",
                name="Perfect prediction",
                line={"dash": "dash", "color": "#64748b"},
                hoverinfo="skip",
            )
        )
        if log_axis:
            actual_predicted.update_xaxes(type="log")
            actual_predicted.update_yaxes(type="log")
        st.plotly_chart(actual_predicted, use_container_width=True)

    impurity = result.impurity_importance.copy()
    impurity["Feature"] = impurity["Feature"].map(_display_feature_name)
    importance_chart = px.bar(
        impurity.sort_values("Importance", ascending=True),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Impurity-Based Feature Importance",
        labels={"Importance": "Impurity-based importance", "Feature": "Model feature"},
    )
    st.plotly_chart(importance_chart, use_container_width=True)
    permutation = result.permutation_importance.copy()
    permutation["Feature"] = permutation["Feature"].map(_display_feature_name)
    permutation_chart = px.bar(
        permutation.sort_values("Importance", ascending=True),
        x="Importance",
        y="Feature",
        error_x="Std",
        orientation="h",
        title="Permutation Importance on Held-Out Test Data",
        labels={
            "Importance": "Increase in test MAE after permutation",
            "Feature": "Input feature",
        },
    )
    st.plotly_chart(permutation_chart, use_container_width=True)
    st.caption(
        "Impurity importance can favor features with many split points. Permutation "
        "importance reports the held-out MAE increase after shuffling each feature "
        "(five repeats). Neither measure implies causation."
    )

    with st.expander("Selected feature relationships with the estimate"):
        relationships = _feature_relationships(df, tuple(selected_features))
        if relationships.empty:
            st.info("No numeric feature relationships are available.")
        else:
            st.dataframe(relationships, hide_index=True, use_container_width=True)
            st.caption(
                "Spearman correlations are descriptive associations calculated from "
                "available pairs. Total reviews is related to popularity but is not "
                "the target or a direct transformation of it; target-derived inputs "
                "remain excluded."
            )


def render_machine_learning(df: pd.DataFrame) -> None:
    """Render the machine-learning analyses using the shared loaded dataset."""
    st.subheader("Machine Learning Analysis")
    clustering_tab, regression_tab = st.tabs(
        ["K-Means Clustering", "Random Forest Regression"]
    )
    with clustering_tab:
        _render_kmeans(df)
    with regression_tab:
        _render_random_forest(df)
