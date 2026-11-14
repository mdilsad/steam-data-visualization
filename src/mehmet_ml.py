from __future__ import annotations

from collections import Counter
from typing import NamedTuple, cast

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

RANDOM_STATE = 42
MAX_SCATTER_POINTS = 10_000
MEHMET_MODEL_SAMPLE_SIZE = 25_000
MEHMET_TAG_MIN_TRAIN_SUPPORT = 1_000
NETWORK_BETWEENNESS_SOURCES = 100

class GenreProfileClusterResult(NamedTuple):
    assignments: pd.DataFrame
    cluster_profiles: pd.DataFrame
    cluster_counts: pd.DataFrame
    pca_data: pd.DataFrame
    silhouette: float | None
    game_count: int


class GenreTagNodeClusterResult(NamedTuple):
    assignments: pd.DataFrame
    cluster_profiles: pd.DataFrame
    cluster_type_counts: pd.DataFrame
    important_nodes: pd.DataFrame
    pca_data: pd.DataFrame
    genre_count: int
    tag_count: int
    edge_count: int
    betweenness_source_count: int


class ReviewProfileClusterResult(NamedTuple):
    scatter_data: pd.DataFrame
    review_distribution: pd.DataFrame
    cluster_profiles: pd.DataFrame
    cluster_counts: pd.DataFrame
    game_count: int


class MultiLabelForestResult(NamedTuple):
    metrics: dict[str, float]
    label_metrics: pd.DataFrame
    train_count: int
    test_count: int
    sampled_count: int
    all_label_count: int
    excluded_labels: list[str]
    used_features: list[str]
    skipped_features: list[str]
    unseen_input_labels: dict[str, int]


class ReviewScoreForestResult(NamedTuple):
    metrics: dict[str, float]
    test_results: pd.DataFrame
    feature_importance: pd.DataFrame
    train_count: int
    test_count: int
    sampled_count: int
    used_features: list[str]
    skipped_features: list[str]
    unseen_input_labels: dict[str, int]

def _style_ml_chart(figure: go.Figure) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        colorway=px.colors.qualitative.Plotly,
        font={"color": "#f1f5f9", "size": 13},
        title={"font": {"color": "#f8fafc", "size": 20}},
        margin={"l": 55, "r": 30, "t": 55, "b": 50},
        legend={
            "font": {"color": "#e2e8f0", "size": 12},
            "title": {"font": {"color": "#e2e8f0", "size": 13}},
            "bgcolor": "rgba(15,23,42,0.8)",
        },
        hoverlabel={
            "bgcolor": "#1e293b",
            "font": {"color": "#f8fafc", "size": 12},
        },
        coloraxis_colorbar={
            "tickfont": {"color": "#cbd5e1", "size": 11},
            "title": {"font": {"color": "#e2e8f0", "size": 13}},
        },
    )
    figure.update_xaxes(
        gridcolor="#334155",
        zerolinecolor="#64748b",
        linecolor="#64748b",
        tickfont={"color": "#cbd5e1", "size": 11},
        title_font={"color": "#e2e8f0", "size": 13},
    )
    figure.update_yaxes(
        gridcolor="#334155",
        zerolinecolor="#64748b",
        linecolor="#64748b",
        tickfont={"color": "#cbd5e1", "size": 11},
        title_font={"color": "#e2e8f0", "size": 13},
    )
    figure.update_annotations(font={"color": "#e2e8f0", "size": 12})
    return figure


def _split_multilabel(value: object) -> list[str]:
    if value is None or value is pd.NA or value is pd.NaT:
        return []
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return []
    return sorted({label.strip() for label in str(value).split(",") if label.strip()})


@st.cache_data(show_spinner=False)
def _fit_genre_profile_clusters(
    df: pd.DataFrame,
    cluster_count: int,
) -> GenreProfileClusterResult:
    required = {
        "app_id",
        "genres",
        "price",
        "review_score",
        "positive_ratio",
        "recommendations",
        "concurrent_users",
        "total",
        "is_free",
        "release_year",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Genre clustering is missing columns: {', '.join(sorted(missing))}")

    source = df[
        [
            "app_id",
            "genres",
            "price",
            "review_score",
            "positive_ratio",
            "recommendations",
            "concurrent_users",
            "total",
            "is_free",
            "release_year",
        ]
    ].dropna(subset=["app_id"]).drop_duplicates("app_id").copy()
    numeric_features = [
        "price",
        "review_score",
        "positive_ratio",
        "recommendations",
        "concurrent_users",
        "total",
        "is_free",
        "release_year",
    ]
    for column in numeric_features:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source = source.replace([np.inf, -np.inf], np.nan)
    source["log1p_total"] = np.log1p(source["total"].clip(lower=0))
    features = [
        "price",
        "review_score",
        "positive_ratio",
        "recommendations",
        "concurrent_users",
        "log1p_total",
        "is_free",
        "release_year",
    ]
    expanded = source.assign(Genre=source["genres"].map(_split_multilabel)).explode("Genre")
    expanded = expanded.dropna(subset=["Genre"]).drop_duplicates(["app_id", "Genre"])
    profiles = expanded.groupby("Genre", sort=True)[features].mean()
    if len(profiles) < 2 or not 2 <= cluster_count <= min(8, len(profiles)):
        raise ValueError("Choose 2 to 8 clusters, with fewer clusters than genre profiles.")

    imputed = SimpleImputer(strategy="median").fit_transform(profiles)
    scaled = StandardScaler().fit_transform(imputed)
    labels = KMeans(
        n_clusters=cluster_count,
        random_state=RANDOM_STATE,
        n_init=20,
    ).fit_predict(scaled)
    distinct_clusters = np.unique(labels)
    silhouette = (
        float(silhouette_score(scaled, labels))
        if 1 < len(distinct_clusters) < len(profiles)
        else None
    )
    pca_coordinates = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)

    assignments = profiles.reset_index()
    assignments["Cluster"] = [f"Cluster {label + 1}" for label in labels]
    cluster_profiles = assignments.groupby("Cluster")[features].mean()
    cluster_profiles.insert(0, "Genres", assignments.groupby("Cluster").size())
    cluster_counts = (
        assignments["Cluster"]
        .value_counts()
        .rename_axis("Cluster")
        .rename("Genres")
        .reset_index()
        .sort_values("Cluster")
    )
    pca_data = pd.DataFrame(
        {
            "PC 1": pca_coordinates[:, 0],
            "PC 2": pca_coordinates[:, 1],
            "Genre": profiles.index,
            "Cluster": assignments["Cluster"],
        }
    )
    return GenreProfileClusterResult(
        assignments=assignments,
        cluster_profiles=cluster_profiles,
        cluster_counts=cluster_counts,
        pca_data=pca_data,
        silhouette=silhouette,
        game_count=len(source),
    )


@st.cache_data(show_spinner=False)
def _fit_genre_tag_node_clusters(
    edges: pd.DataFrame,
    genre_counts: dict[str, int],
    tag_counts: dict[str, int],
    cluster_count: int,
) -> GenreTagNodeClusterResult:
    graph = nx.Graph()
    graph.add_nodes_from(("Genre", genre) for genre in genre_counts)
    graph.add_nodes_from(("Tag", tag) for tag in tag_counts)
    for genre, tag, shared_games in edges[
        ["genre", "tag", "shared_games"]
    ].itertuples(index=False, name=None):
        weight = max(1, int(shared_games))
        graph.add_edge(
            ("Genre", genre),
            ("Tag", tag),
            weight=weight,
            distance=1 / weight,
        )
    if graph.number_of_nodes() < 2 or not 2 <= cluster_count <= min(8, graph.number_of_nodes()):
        raise ValueError("Choose 2 to 8 clusters for the Genre–Tag network nodes.")

    betweenness_source_count = min(NETWORK_BETWEENNESS_SOURCES, graph.number_of_nodes())
    betweenness = nx.betweenness_centrality(
        graph,
        k=betweenness_source_count if graph.number_of_nodes() > NETWORK_BETWEENNESS_SOURCES else None,
        normalized=True,
        weight="distance",
        seed=RANDOM_STATE,
    )
    page_rank = nx.pagerank(graph, weight="weight", max_iter=500)
    weighted_degree = dict(graph.degree(weight="weight"))
    clustering = cast(
        dict[tuple[str, str], float],
        nx.clustering(graph, weight="weight"),
    )
    degree = dict(graph.degree())

    nodes = sorted(graph.nodes, key=lambda node: (node[0], node[1]))
    feature_rows = [
        {
            "node_type": node[0],
            "name": node[1],
            "degree": degree[node],
            "weighted_degree": weighted_degree[node],
            "pagerank": page_rank[node],
            "betweenness": betweenness[node],
            "clustering_coefficient": clustering[node],
            "game_count": genre_counts[node[1]] if node[0] == "Genre" else tag_counts[node[1]],
        }
        for node in nodes
    ]
    node_features = pd.DataFrame(feature_rows)
    numeric_features = [
        "degree",
        "weighted_degree",
        "pagerank",
        "betweenness",
        "clustering_coefficient",
    ]
    imputed = SimpleImputer(strategy="median").fit_transform(node_features[numeric_features])
    scaled = StandardScaler().fit_transform(imputed)
    model_features = scaled
    labels = KMeans(
        n_clusters=cluster_count,
        random_state=RANDOM_STATE,
        n_init=20,
    ).fit_predict(model_features)
    assignments = node_features.rename(
        columns={
            "node_type": "Node type",
            "name": "Name",
            "degree": "Degree",
            "weighted_degree": "Weighted degree",
            "pagerank": "PageRank",
            "betweenness": "Betweenness",
            "clustering_coefficient": "Clustering coefficient",
            "game_count": "Games",
        }
    )
    assignments["Cluster"] = [f"Cluster {label + 1}" for label in labels]
    cluster_profiles = assignments.groupby("Cluster").agg(
        Nodes=("Name", "size"),
        **{
            "Genre nodes": ("Node type", lambda values: int(values.eq("Genre").sum())),
            "Tag nodes": ("Node type", lambda values: int(values.eq("Tag").sum())),
            "Mean degree": ("Degree", "mean"),
            "Mean weighted degree": ("Weighted degree", "mean"),
            "Mean PageRank": ("PageRank", "mean"),
            "Mean betweenness": ("Betweenness", "mean"),
            "Mean clustering coefficient": ("Clustering coefficient", "mean"),
        },
    )
    cluster_type_counts = (
        assignments.groupby(["Cluster", "Node type"])
        .size()
        .rename("Nodes")
        .reset_index()
    )
    important_nodes = (
        assignments.sort_values(
            ["Cluster", "PageRank", "Weighted degree"],
            ascending=[True, False, False],
        )
        .groupby("Cluster", sort=True)
        .head(3)
        .reset_index(drop=True)
    )
    pca_coordinates = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(
        model_features
    )
    pca_data = pd.DataFrame(
        {
            "PC 1": pca_coordinates[:, 0],
            "PC 2": pca_coordinates[:, 1],
            "Name": assignments["Name"],
            "Node type": assignments["Node type"],
            "Cluster": assignments["Cluster"],
            "Degree": assignments["Degree"],
            "Weighted degree": assignments["Weighted degree"],
        }
    )
    return GenreTagNodeClusterResult(
        assignments=assignments,
        cluster_profiles=cluster_profiles,
        cluster_type_counts=cluster_type_counts,
        important_nodes=important_nodes,
        pca_data=pca_data,
        genre_count=len(genre_counts),
        tag_count=len(tag_counts),
        edge_count=graph.number_of_edges(),
        betweenness_source_count=betweenness_source_count,
    )


def _encode_train_test_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_candidates: tuple[str, ...],
    categorical_candidates: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, int]]:
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    feature_names: list[str] = []
    skipped_features: list[str] = []
    unseen_input_labels: dict[str, int] = {}

    available_numeric = []
    for column in numeric_candidates:
        if column not in train.columns:
            skipped_features.append(column)
            continue
        train_values = pd.to_numeric(train[column], errors="coerce")
        if not train_values.notna().any():
            skipped_features.append(column)
            continue
        available_numeric.append(column)
    if available_numeric:
        train_numeric = train[available_numeric].apply(pd.to_numeric, errors="coerce")
        test_numeric = test[available_numeric].apply(pd.to_numeric, errors="coerce")
        train_numeric = train_numeric.replace([np.inf, -np.inf], np.nan)
        test_numeric = test_numeric.replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="median")
        train_parts.append(np.asarray(imputer.fit_transform(train_numeric), dtype=float))
        test_parts.append(np.asarray(imputer.transform(test_numeric), dtype=float))
        feature_names.extend(available_numeric)

    for column in categorical_candidates:
        if column not in train.columns:
            skipped_features.append(column)
            continue
        train_labels = train[column].map(_split_multilabel)
        test_labels = test[column].map(_split_multilabel)
        encoder = MultiLabelBinarizer()
        train_encoded = np.asarray(encoder.fit_transform(train_labels), dtype=float)
        known_labels = set(encoder.classes_)
        unseen = {
            label
            for labels in test_labels
            for label in labels
            if label not in known_labels
        }
        unseen_input_labels[column] = len(unseen)
        known_test_labels = [
            [label for label in labels if label in known_labels]
            for labels in test_labels
        ]
        test_encoded = np.asarray(
            encoder.transform(known_test_labels),
            dtype=float,
        )
        if train_encoded.shape[1] == 0:
            skipped_features.append(f"{column} (no training labels)")
            continue
        train_parts.append(train_encoded)
        test_parts.append(test_encoded)
        feature_names.extend(f"{column}: {label}" for label in encoder.classes_)

    if not train_parts:
        raise ValueError("No supported numeric or multi-label input features are available.")
    return (
        np.column_stack(train_parts),
        np.column_stack(test_parts),
        feature_names,
        skipped_features,
        unseen_input_labels,
    )


@st.cache_data(show_spinner=False)
def _fit_multilabel_forest(
    df: pd.DataFrame,
    target_column: str,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    minimum_train_support: int,
) -> MultiLabelForestResult:
    if target_column not in df.columns:
        raise ValueError(f"The dataset does not contain the {target_column} target.")

    all_label_vocab = {
        label
        for value in df[target_column]
        for label in _split_multilabel(value)
    }
    source = df.copy().reset_index(drop=True)
    source["_target_labels"] = source[target_column].map(_split_multilabel)
    source = source.loc[source["_target_labels"].map(bool)]
    if "app_id" in source.columns:
        source = source.drop_duplicates("app_id")
    if source.empty:
        raise ValueError(f"No games have usable {target_column} labels.")
    if len(source) > MEHMET_MODEL_SAMPLE_SIZE:
        source = source.sample(n=MEHMET_MODEL_SAMPLE_SIZE, random_state=RANDOM_STATE)
    source = source.reset_index(drop=True)
    train, test = train_test_split(
        source,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    train_support = Counter(
        label for labels in train["_target_labels"] for label in labels
    )
    selected_labels = sorted(
        label
        for label, count in train_support.items()
        if count >= minimum_train_support
    )
    if not selected_labels:
        raise ValueError(
            f"No {target_column} labels reach the minimum training support of "
            f"{minimum_train_support:,}."
        )
    label_encoder = MultiLabelBinarizer(classes=selected_labels)
    selected_label_set = set(selected_labels)
    train_target_labels = train["_target_labels"].map(
        lambda labels: [label for label in labels if label in selected_label_set]
    )
    test_target_labels = test["_target_labels"].map(
        lambda labels: [label for label in labels if label in selected_label_set]
    )
    y_train = np.asarray(
        label_encoder.fit_transform(train_target_labels),
        dtype=np.uint8,
    )
    y_test = np.asarray(
        label_encoder.transform(test_target_labels),
        dtype=np.uint8,
    )
    x_train, x_test, feature_names, skipped, unseen = _encode_train_test_features(
        train,
        test,
        numeric_features,
        categorical_features,
    )

    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=80,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average=None,
        zero_division=0,
    )
    test_support = y_test.sum(axis=0)
    label_metrics = pd.DataFrame(
        {
            "Label": selected_labels,
            "Train support": [train_support[label] for label in selected_labels],
            "Test support": test_support,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    ).sort_values(["Test support", "Label"], ascending=[False, True])
    metrics = {
        "Micro F1": float(f1_score(y_test, predictions, average="micro", zero_division=0)),
        "Macro F1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "Precision": float(precision_score(y_test, predictions, average="micro", zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, average="micro", zero_division=0)),
    }
    return MultiLabelForestResult(
        metrics=metrics,
        label_metrics=label_metrics.reset_index(drop=True),
        train_count=len(train),
        test_count=len(test),
        sampled_count=len(source),
        all_label_count=len(all_label_vocab),
        excluded_labels=sorted(all_label_vocab.difference(selected_labels)),
        used_features=feature_names,
        skipped_features=skipped,
        unseen_input_labels=unseen,
    )


@st.cache_data(show_spinner=False)
def _fit_review_profile_clusters(
    df: pd.DataFrame,
    cluster_count: int,
) -> ReviewProfileClusterResult:
    required = {
        "review_score",
        "positive_ratio",
        "positive",
        "negative",
        "recommendations",
        "price",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Review clustering is missing columns: {', '.join(sorted(missing))}")

    features = pd.DataFrame(index=df.index)
    for column in ("review_score", "positive_ratio", "positive", "negative", "recommendations", "price"):
        features[column] = pd.to_numeric(df[column], errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features.loc[features["review_score"] <= 0, "review_score"] = np.nan
    for column in ("positive", "negative", "recommendations", "price"):
        features[f"log1p_{column}"] = np.log1p(features[column].clip(lower=0))
    feature_columns = [
        "review_score",
        "positive_ratio",
        "log1p_positive",
        "log1p_negative",
        "log1p_recommendations",
        "log1p_price",
    ]
    raw_features = features[feature_columns]
    valid_rows = raw_features.notna().any(axis=1)
    raw_features = raw_features.loc[valid_rows]
    if not 2 <= cluster_count <= min(8, len(raw_features)):
        raise ValueError("Choose 2 to 8 clusters for the review profiles.")

    imputed = SimpleImputer(strategy="median").fit_transform(raw_features)
    scaled = StandardScaler().fit_transform(imputed)
    model = KMeans(
        n_clusters=cluster_count,
        random_state=RANDOM_STATE,
        n_init=10,
    ).fit(scaled)
    labels = np.array([f"Cluster {label + 1}" for label in model.labels_])
    coordinates = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)
    chart_indices = raw_features.index
    if len(chart_indices) > MAX_SCATTER_POINTS:
        sample_positions = np.random.default_rng(RANDOM_STATE).choice(
            len(chart_indices),
            size=MAX_SCATTER_POINTS,
            replace=False,
        )
    else:
        sample_positions = np.arange(len(chart_indices))
    scatter_data = pd.DataFrame(
        {
            "PC 1": coordinates[sample_positions, 0],
            "PC 2": coordinates[sample_positions, 1],
            "Cluster": labels[sample_positions],
        },
        index=chart_indices[sample_positions],
    )
    if "name" in df.columns:
        scatter_data["Game"] = df.loc[scatter_data.index, "name"].fillna("Unknown title")

    profile_values = pd.DataFrame(imputed, columns=feature_columns, index=raw_features.index)
    profile_values["Cluster"] = labels
    cluster_profiles = profile_values.groupby("Cluster")[feature_columns].median()
    cluster_profiles.insert(0, "Games", profile_values.groupby("Cluster").size())
    cluster_counts = (
        profile_values["Cluster"]
        .value_counts()
        .rename_axis("Cluster")
        .rename("Games")
        .reset_index()
        .sort_values("Cluster")
    )
    review_scores = pd.to_numeric(df.loc[raw_features.index, "review_score"], errors="coerce")
    review_distribution = pd.DataFrame(
        {
            "Review score": review_scores,
            "Cluster": labels,
        },
        index=raw_features.index,
    )
    review_distribution = review_distribution.loc[review_distribution["Review score"] > 0]
    return ReviewProfileClusterResult(
        scatter_data=scatter_data,
        review_distribution=review_distribution,
        cluster_profiles=cluster_profiles,
        cluster_counts=cluster_counts,
        game_count=len(raw_features),
    )


@st.cache_data(show_spinner=False)
def _fit_review_score_forest(df: pd.DataFrame) -> ReviewScoreForestResult:
    target = pd.to_numeric(df["review_score"], errors="coerce")
    target = target.replace([np.inf, -np.inf], np.nan)
    valid_target = target.gt(0) & target.le(10)
    source = df.loc[valid_target].copy()
    source["review_score"] = target.loc[valid_target]
    if "app_id" in source.columns:
        source = source.drop_duplicates("app_id")
    if len(source) > MEHMET_MODEL_SAMPLE_SIZE:
        source = source.sample(n=MEHMET_MODEL_SAMPLE_SIZE, random_state=RANDOM_STATE)
    source = source.reset_index(drop=True)
    train, test = train_test_split(
        source,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    x_train, x_test, feature_names, skipped, unseen = _encode_train_test_features(
        train,
        test,
        ("price", "release_year", "is_free"),
        ("genres", "categories"),
    )
    forbidden_features = {"positive", "negative", "total", "positive_ratio"}
    if forbidden_features.intersection(feature_names):
        raise RuntimeError("Review score model inputs contain target leakage.")

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(x_train, train["review_score"])
    predictions = model.predict(x_test)
    actual = test["review_score"].to_numpy(dtype=float)
    metrics = {
        "MAE": float(mean_absolute_error(actual, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predictions))),
        "R²": float(r2_score(actual, predictions)),
    }
    test_results = pd.DataFrame(
        {
            "Actual": actual,
            "Predicted": predictions,
            "Game": (
                test["name"].fillna("Unknown title").to_numpy()
                if "name" in test.columns
                else test.index.astype(str)
            ),
        }
    )
    feature_importance = pd.DataFrame(
        {"Feature": feature_names, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)
    return ReviewScoreForestResult(
        metrics=metrics,
        test_results=test_results,
        feature_importance=feature_importance,
        train_count=len(train),
        test_count=len(test),
        sampled_count=len(source),
        used_features=feature_names,
        skipped_features=skipped,
        unseen_input_labels=unseen,
    )


def _format_encoded_features(features: list[str]) -> str:
    numeric_features = [feature for feature in features if ": " not in feature]
    encoded_counts = Counter(
        feature.split(": ", maxsplit=1)[0]
        for feature in features
        if ": " in feature
    )
    descriptions = numeric_features.copy()
    descriptions.extend(
        f"{column} (multi-hot; {count} training labels)"
        for column, count in sorted(encoded_counts.items())
    )
    return ", ".join(descriptions)


def _render_multilabel_forest_result(
    result: MultiLabelForestResult,
    target_name: str,
    label_unit: str,
) -> None:
    metrics = st.columns(4)
    metrics[0].metric("Micro F1", f"{result.metrics['Micro F1']:.3f}")
    metrics[1].metric("Macro F1", f"{result.metrics['Macro F1']:.3f}")
    metrics[2].metric("Micro Precision", f"{result.metrics['Precision']:.3f}")
    metrics[3].metric("Micro Recall", f"{result.metrics['Recall']:.3f}")
    st.caption(
        f"Target: {target_name}. Features: {_format_encoded_features(result.used_features)}. "
        f"Training / test games: {result.train_count:,} / {result.test_count:,}; "
        f"{result.sampled_count:,} games in the reproducible sample. "
        "The random split is row-based; multi-label stratification is not applied."
    )
    st.caption(
        f"Evaluated {len(result.label_metrics):,} of {result.all_label_count:,} "
        f"observed {label_unit} labels; {len(result.excluded_labels):,} excluded."
    )
    if result.skipped_features:
        st.caption(
            "Unavailable input columns skipped: " + ", ".join(result.skipped_features)
        )
    for column, count in result.unseen_input_labels.items():
        if count:
            st.caption(
                f"{count:,} {column} labels were absent from training and were not "
                "encoded for test rows."
            )

    chart_labels = result.label_metrics.nlargest(25, "Test support").sort_values(
        "F1", ascending=True
    )
    chart = px.bar(
        chart_labels,
        x="F1",
        y="Label",
        orientation="h",
        title=f"Per-{label_unit} F1 (up to 25 most frequent evaluated labels)",
        labels={"F1": "F1 score", "Label": label_unit.title()},
        hover_data={
            "Test support": ":,",
            "Precision": ":.3f",
            "Recall": ":.3f",
        },
        color_discrete_sequence=[px.colors.qualitative.Plotly[0]],
    )
    chart.update_layout(height=max(400, min(1_200, 30 * len(chart_labels))))
    _style_ml_chart(chart)
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(
        result.label_metrics,
        hide_index=True,
        use_container_width=True,
    )
    if result.excluded_labels:
        with st.expander(f"View excluded {label_unit} labels"):
            st.write(", ".join(result.excluded_labels))


def render_genre_distribution_analysis(df: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Machine Learning — Genre Distribution")
    st.markdown("**K-Means: Genre Profile Clustering**")
    genre_count = len(
        {
            genre
            for value in df["genres"].dropna()
            for genre in _split_multilabel(value)
        }
    )
    maximum_clusters = min(8, genre_count)
    cluster_count = st.slider(
        "Number of genre-profile clusters",
        min_value=2,
        max_value=maximum_clusters,
        value=min(4, maximum_clusters),
        key="mehmet_genre_profile_cluster_count",
    )
    with st.spinner("Clustering genre profiles..."):
        result = _fit_genre_profile_clusters(df, cluster_count)
    st.caption(
        f"The clustering unit is the genre profile, not an individual game: "
        f"{genre_count} genre profiles are aggregated from {result.game_count:,} games. "
        "With only 12 profiles in this dataset, this is exploratory and should not be "
        "treated as a statistically stable taxonomy. The silhouette score uses the "
        "scaled values of these 12 profiles for the selected K."
    )
    summary = st.columns(3)
    summary[0].metric("Genre profiles", f"{genre_count:,}")
    summary[1].metric(
        "Silhouette score",
        f"{result.silhouette:.3f}" if result.silhouette is not None else "N/A",
    )
    summary[2].metric("Selected K", str(cluster_count))
    scatter = px.scatter(
        result.pca_data,
        x="PC 1",
        y="PC 2",
        color="Cluster",
        text="Genre",
        hover_data=["Genre", "Cluster"],
        title="PCA Projection of Genre Profiles",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    scatter.update_traces(textposition="top center")
    _style_ml_chart(scatter)
    st.plotly_chart(scatter, use_container_width=True)
    st.subheader("Genre Cluster Assignments")
    st.dataframe(
        result.assignments.sort_values(["Cluster", "Genre"]),
        hide_index=True,
        use_container_width=True,
    )
    distribution = px.bar(
        result.cluster_counts,
        x="Cluster",
        y="Genres",
        color="Cluster",
        text_auto=True,
        title="Number of Genres in Each Cluster",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    distribution.update_layout(showlegend=False)
    _style_ml_chart(distribution)
    st.plotly_chart(distribution, use_container_width=True)
    cluster_profiles = result.cluster_profiles.rename(
        columns={
            "price": "Mean price",
            "review_score": "Mean review score",
            "positive_ratio": "Mean positive ratio",
            "recommendations": "Mean recommendations",
            "concurrent_users": "Mean concurrent users",
            "log1p_total": "Mean log1p(total)",
            "is_free": "Free-game ratio",
            "release_year": "Mean release year",
        }
    )
    st.subheader("Genre Cluster Characteristics")
    st.dataframe(cluster_profiles, use_container_width=True)


def fit_genre_tag_node_clusters(
    edges: pd.DataFrame,
    genre_counts: dict[str, int],
    tag_counts: dict[str, int],
    cluster_count: int,
) -> GenreTagNodeClusterResult:
    return _fit_genre_tag_node_clusters(
        edges,
        genre_counts,
        tag_counts,
        cluster_count,
    )


def render_genre_tag_node_cluster_results(
    result: GenreTagNodeClusterResult,
) -> None:
    st.subheader("K-Means: Network Node Clustering")
    st.caption(
        f"K-Means uses {result.genre_count:,} Genre nodes and {result.tag_count:,} Tag "
        f"nodes from {result.edge_count:,} observed network edges. Node type is excluded "
        "from K-Means inputs and retained for symbols, hover details, and cluster summaries. "
        f"Betweenness uses {result.betweenness_source_count:,} source nodes"
        f"{' (reproducible approximation)' if result.betweenness_source_count < result.genre_count + result.tag_count else ' (exact)'}. "
        "Clustering coefficients are zero for this bipartite graph because it contains "
        "no triangles."
    )
    pca = px.scatter(
        result.pca_data,
        x="PC 1",
        y="PC 2",
        color="Cluster",
        symbol="Node type",
        hover_name="Name",
        hover_data=["Node type", "Degree", "Weighted degree"],
        title="PCA Projection of Genre–Tag Network Nodes",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    _style_ml_chart(pca)
    st.plotly_chart(pca, use_container_width=True)
    distribution = px.bar(
        result.cluster_type_counts,
        x="Cluster",
        y="Nodes",
        color="Node type",
        barmode="stack",
        title="Cluster Distribution by Node Type",
        color_discrete_map={
            "Genre": px.colors.qualitative.Plotly[0],
            "Tag": px.colors.qualitative.Plotly[1],
        },
    )
    _style_ml_chart(distribution)
    st.plotly_chart(distribution, use_container_width=True)
    st.subheader("Network Cluster Characteristics")
    st.dataframe(result.cluster_profiles, use_container_width=True)
    st.subheader("Important Nodes in Each Cluster")
    st.dataframe(
        result.important_nodes[
            [
                "Cluster",
                "Node type",
                "Name",
                "Degree",
                "Weighted degree",
                "PageRank",
                "Betweenness",
                "Clustering coefficient",
                "Games",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )


def render_tag_prediction_analysis(df: pd.DataFrame) -> None:
    st.subheader("Random Forest: Multi-label Tag Prediction")
    with st.spinner("Training the tag prediction model..."):
        result = _fit_multilabel_forest(
            df,
            "tags",
            (
                "price",
                "review_score",
                "positive_ratio",
                "release_year",
                "is_free",
                "recommendations",
            ),
            ("genres",),
            MEHMET_TAG_MIN_TRAIN_SUPPORT,
        )
    st.caption(
        f"Tag labels require at least {MEHMET_TAG_MIN_TRAIN_SUPPORT:,} occurrences "
        "in the training split. Rare labels are listed below and are not represented "
        "in these metrics."
    )
    _render_multilabel_forest_result(result, "tags", "tag")


def render_genre_distribution_random_forest(df: pd.DataFrame) -> None:
    st.subheader("Random Forest: Multi-label Genre Prediction")
    with st.spinner("Training the genre prediction model..."):
        result = _fit_multilabel_forest(
            df,
            "genres",
            (
                "price",
                "review_score",
                "positive_ratio",
                "recommendations",
                "concurrent_users",
                "release_year",
                "is_free",
            ),
            (),
            1,
        )
    _render_multilabel_forest_result(result, "genres", "genre")


def render_review_profile_analysis(df: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Machine Learning — Genre–Review Score")
    st.markdown("**K-Means: Review Profile Clustering**")
    cluster_count = st.slider(
        "Number of review-profile clusters",
        min_value=2,
        max_value=8,
        value=4,
        key="mehmet_review_profile_cluster_count",
    )
    with st.spinner("Clustering game review profiles..."):
        result = _fit_review_profile_clusters(df, cluster_count)
    st.caption(
        f"K-Means was fit on {result.game_count:,} games. A review_score of 0 is "
        "treated as unscored and median-imputed for clustering; the box plot below "
        "uses only games with an actual positive review score. "
        f"The PCA display samples up to {MAX_SCATTER_POINTS:,} games."
    )
    scatter = px.scatter(
        result.scatter_data,
        x="PC 1",
        y="PC 2",
        color="Cluster",
        hover_name="Game" if "Game" in result.scatter_data.columns else None,
        title="PCA Projection of Review Profiles",
        opacity=0.65,
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    _style_ml_chart(scatter)
    st.plotly_chart(scatter, use_container_width=True)
    review_box = px.box(
        result.review_distribution,
        x="Cluster",
        y="Review score",
        color="Cluster",
        points=False,
        title="Review Score Distribution by K-Means Cluster",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    _style_ml_chart(review_box)
    st.plotly_chart(review_box, use_container_width=True)
    distribution = px.bar(
        result.cluster_counts,
        x="Cluster",
        y="Games",
        color="Cluster",
        text_auto=True,
        title="Review Profile Cluster Sizes",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    distribution.update_layout(showlegend=False)
    _style_ml_chart(distribution)
    st.plotly_chart(distribution, use_container_width=True)
    profile = result.cluster_profiles.rename(
        columns={
            "review_score": "Review score median",
            "positive_ratio": "Positive ratio median",
            "log1p_positive": "log1p(positive) median",
            "log1p_negative": "log1p(negative) median",
            "log1p_recommendations": "log1p(recommendations) median",
            "log1p_price": "log1p(price) median",
        }
    )
    st.subheader("Review Profile Cluster Characteristics")
    st.dataframe(profile, use_container_width=True)


def render_review_score_random_forest(df: pd.DataFrame) -> None:
    st.subheader("Random Forest: Review Score Prediction")
    with st.spinner("Training the review score prediction model..."):
        result = _fit_review_score_forest(df)
    metrics = st.columns(3)
    metrics[0].metric("MAE", f"{result.metrics['MAE']:.3f}")
    metrics[1].metric("RMSE", f"{result.metrics['RMSE']:.3f}")
    metrics[2].metric("R²", f"{result.metrics['R²']:.3f}")
    st.caption(
        "Target: review_score. Features: price, release_year, is_free, and training-only "
        "multi-hot encodings of genres and categories. positive, negative, total, and "
        "positive_ratio are excluded to prevent target leakage. review_score = 0 rows "
        "are treated as unscored and excluded. "
        f"Training / test games: {result.train_count:,} / {result.test_count:,}; "
        f"{result.sampled_count:,} games in the reproducible sample."
    )
    st.caption(
        f"R² = {result.metrics['R²']:.3f} means the leakage-safe inputs explain only a "
        "modest share of held-out score variation. Review counts and positive_ratio, "
        "which could encode the target, are deliberately excluded."
    )
    if result.skipped_features:
        st.caption(
            "Unavailable input columns skipped: " + ", ".join(result.skipped_features)
        )
    for column, count in result.unseen_input_labels.items():
        if count:
            st.caption(
                f"{count:,} {column} labels were absent from training and were not "
                "encoded for test rows."
            )
    predictions = px.scatter(
        result.test_results,
        x="Actual",
        y="Predicted",
        hover_name="Game",
        hover_data={"Actual": ":.2f", "Predicted": ":.2f"},
        title="Actual vs Predicted Review Score",
        opacity=0.55,
        color_discrete_sequence=[px.colors.qualitative.Plotly[0]],
    )
    axis_min = float(
        min(result.test_results["Actual"].min(), result.test_results["Predicted"].min())
    )
    axis_max = float(
        max(result.test_results["Actual"].max(), result.test_results["Predicted"].max())
    )
    predictions.add_trace(
        go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode="lines",
            name="Perfect prediction",
            line={"dash": "dash", "color": "#64748b"},
            hoverinfo="skip",
        )
    )
    _style_ml_chart(predictions)
    st.plotly_chart(predictions, use_container_width=True)
    importance = result.feature_importance.head(25).sort_values(
        "Importance", ascending=True
    )
    chart = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Review Score Feature Importance (Top 25)",
        color_discrete_sequence=[px.colors.qualitative.Plotly[0]],
    )
    _style_ml_chart(chart)
    st.plotly_chart(chart, use_container_width=True)
    with st.expander("Review score model feature importances"):
        st.dataframe(
            result.feature_importance,
            hide_index=True,
            use_container_width=True,
        )
