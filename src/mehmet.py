from collections import Counter
import math
from typing import cast

import networkx as nx
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

NETWORK_NODE_LIMIT = 45
NETWORK_EDGE_LIMIT = 230


def get_top_genre_counts(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """DataFrame içindeki virgülle ayrılmış türleri ayrıştırır ve oyun sayılarını hesaplar."""
    if "genres" not in df.columns:
        raise ValueError("DataFrame içinde 'genres' sütunu bulunamadı.")

    # Virgülle birleştirilmiş türleri satırlara patlat (explode) ve temizle
    exploded_genres = (
        df[["app_id", "genres"]]
        .dropna(subset=["genres"])
        .assign(genre=lambda x: x["genres"].str.split(","))
        .explode("genre")
    )
    exploded_genres["genre"] = exploded_genres["genre"].str.strip()
    exploded_genres = exploded_genres[exploded_genres["genre"] != ""]

    # Mehmet'in orijinal mantığı: Tür başına benzersiz (nunique) oyun sayısı
    counts = (
        exploded_genres.groupby("genre", as_index=False)["app_id"]
        .nunique()
        .rename(columns={"app_id": "game_count"})
        .sort_values(["game_count", "genre"], ascending=[False, True])
        .head(limit)
        .reset_index(drop=True)
    )
    return counts


def create_genre_bar_chart(counts: pd.DataFrame, limit: int = 10) -> go.Figure:
    """Mehmet'in yatay bar grafiği tasarımını oluşturur."""
    chart_data = counts.sort_values(["game_count", "genre"], ascending=[True, False])

    figure = px.bar(
        chart_data,
        x="game_count",
        y="genre",
        orientation="h",
        title=f"Top {limit} Game Genres on Steam",
        labels={"game_count": "Number of Games", "genre": "Genre"},
        hover_name="genre",
        hover_data={"game_count": ":,", "genre": False},
    )
    figure.update_traces(
        hovertemplate="%{y}<br>Games: %{x:,}<extra></extra>"
    )
    figure.update_layout(
        yaxis={"categoryorder": "array", "categoryarray": chart_data["genre"].tolist()},
        showlegend=False,
    )
    return figure


def _split_comma_values(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


@st.cache_data(show_spinner=False)
def get_genre_tag_relationships(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int], dict[str, int]]:
    """Return unique-game genre/tag edges, node counts, and data-quality counts."""
    required = {"app_id", "genres", "tags"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")

    games: dict[object, tuple[set[str], set[str]]] = {}
    missing_app_id_rows = 0
    valid_app_id_rows = 0
    for app_id, genres, tags in df[["app_id", "genres", "tags"]].itertuples(
        index=False, name=None
    ):
        if pd.isna(app_id):
            missing_app_id_rows += 1
            continue
        valid_app_id_rows += 1
        if app_id not in games:
            games[app_id] = (set(), set())
        games[app_id][0].update(_split_comma_values(genres))
        games[app_id][1].update(_split_comma_values(tags))

    genre_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()
    missing_genre_games = 0
    missing_tag_games = 0

    for genres, tags in games.values():
        if not genres:
            missing_genre_games += 1
        if not tags:
            missing_tag_games += 1
        genre_counts.update(genres)
        tag_counts.update(tags)
        edge_counts.update((genre, tag) for genre in genres for tag in tags)

    edges = pd.DataFrame(
        [
            {"genre": genre, "tag": tag, "shared_games": count}
            for (genre, tag), count in edge_counts.items()
        ],
        columns=["genre", "tag", "shared_games"],
    )
    if not edges.empty:
        edges = edges.sort_values(
            ["shared_games", "genre", "tag"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    diagnostics = {
        "unique_games": len(games),
        "missing_app_id_rows": missing_app_id_rows,
        "duplicate_rows": valid_app_id_rows - len(games),
        "missing_genre_games": missing_genre_games,
        "missing_tag_games": missing_tag_games,
    }
    return edges, dict(genre_counts), dict(tag_counts), diagnostics


def create_genre_tag_network(
    edges: pd.DataFrame,
    genre_counts: dict[str, int],
    tag_counts: dict[str, int],
    minimum_shared_games: int,
) -> go.Figure:
    """Render a compact force-directed Genre–Tag community graph."""
    nodes, network_edges = _prepare_genre_tag_network(
        edges, genre_counts, tag_counts, minimum_shared_games
    )
    figure = go.Figure()
    if nodes.empty:
        figure.add_annotation(
            text="No relationships match this minimum shared-games threshold.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        figure.update_layout(
            title={
                "text": "Genre–Tag Community Network",
                "font": {"color": "#f8fafc", "size": 20},
            },
            height=650,
            paper_bgcolor="#0b1120",
            plot_bgcolor="#0b1120",
            font={"color": "#f1f5f9"},
        )
        return figure

    line_x: list[float | None] = []
    line_y: list[float | None] = []
    edge_hover_x: list[float] = []
    edge_hover_y: list[float] = []
    edge_hover_data: list[list[object]] = []
    for edge in network_edges.itertuples(index=False):
        weight = cast(int, edge.shared_games)
        x0 = cast(float, edge.x0)
        y0 = cast(float, edge.y0)
        x1 = cast(float, edge.x1)
        y1 = cast(float, edge.y1)
        line_x.extend([x0, x1, None])
        line_y.extend([y0, y1, None])
        edge_hover_x.append((x0 + x1) / 2)
        edge_hover_y.append((y0 + y1) / 2)
        edge_hover_data.append([edge.genre, edge.tag, weight])

    figure.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line={"color": "#94a3b8", "width": 0.7},
            opacity=0.4,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=edge_hover_x,
            y=edge_hover_y,
            mode="markers",
            marker={"size": 10, "color": "rgba(0,0,0,0.01)"},
            customdata=edge_hover_data,
            hovertemplate=(
                "Genre: %{customdata[0]}<br>Tag: %{customdata[1]}"
                "<br>Shared games: %{customdata[2]:,}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    community_ids = sorted(nodes["community"].unique().tolist())
    community_colors = px.colors.qualitative.Set1
    for index, community_id in enumerate(community_ids):
        community_nodes = nodes[nodes["community"] == community_id]
        figure.add_trace(
            go.Scatter(
                x=community_nodes["x"],
                y=community_nodes["y"],
                mode="markers",
                customdata=community_nodes[
                    ["node_type", "degree", "game_count", "community", "name"]
                ],
                marker={
                    "color": community_colors[index % len(community_colors)],
                    "size": 22,
                    "line": {"width": 0.6, "color": "#ffffff"},
                },
                hovertemplate=(
                    "Name: %{customdata[4]}<br>Type: %{customdata[0]}"
                    "<br>Connections: %{customdata[1]:,}"
                    "<br>Games: %{customdata[2]:,}"
                    "<br>Community: %{customdata[3]}<extra></extra>"
                ),
                name=f"Community {community_id}",
            )
        )

    figure.update_layout(
        title={
            "text": "Genre–Tag Community Network",
            "font": {"color": "#f8fafc", "size": 20},
        },
        height=650,
        hovermode="closest",
        dragmode="pan",
        showlegend=True,
        legend={
            "title": {"text": "Communities", "font": {"color": "#e2e8f0"}},
            "x": 1.02,
            "y": 1,
            "font": {"color": "#e2e8f0"},
            "bgcolor": "rgba(15,23,42,0.8)",
        },
        margin={"l": 20, "r": 150, "t": 55, "b": 20},
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        font={"color": "#f1f5f9"},
        hoverlabel={"bgcolor": "#1e293b", "font": {"color": "#f8fafc"}},
        xaxis={"visible": False, "showgrid": False, "zeroline": False},
        yaxis={"visible": False, "showgrid": False, "zeroline": False},
    )
    return figure


@st.cache_data(show_spinner=False)
def _prepare_genre_tag_network(
    edges: pd.DataFrame,
    genre_counts: dict[str, int],
    tag_counts: dict[str, int],
    minimum_shared_games: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter observed edges, detect communities, and cache the force layout."""
    visible_edges = edges.loc[
        edges["shared_games"] >= minimum_shared_games
    ].copy()
    if visible_edges.empty:
        return pd.DataFrame(), pd.DataFrame()

    degrees: Counter[tuple[str, str]] = Counter()
    for genre, tag in visible_edges[["genre", "tag"]].itertuples(
        index=False, name=None
    ):
        degrees[("Genre", genre)] += 1
        degrees[("Tag", tag)] += 1
    selected_nodes = {
        node
        for node, _ in sorted(
            degrees.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )[:NETWORK_NODE_LIMIT]
    }

    visible_edges["genre_node"] = [
        ("Genre", genre) for genre in visible_edges["genre"]
    ]
    visible_edges["tag_node"] = [("Tag", tag) for tag in visible_edges["tag"]]
    visible_edges = visible_edges.loc[
        visible_edges["genre_node"].isin(selected_nodes)
        & visible_edges["tag_node"].isin(selected_nodes)
    ].copy()
    if visible_edges.empty:
        return pd.DataFrame(), pd.DataFrame()
    visible_edges = visible_edges.sort_values(
        ["shared_games", "genre", "tag"],
        ascending=[False, True, True],
    ).head(NETWORK_EDGE_LIMIT)

    graph = nx.Graph()
    for edge in visible_edges.itertuples(index=False):
        graph.add_edge(
            edge.genre_node,
            edge.tag_node,
            weight=cast(int, edge.shared_games),
        )

    graph = nx.k_core(graph, k=2)
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(), pd.DataFrame()
    core_nodes = set(graph.nodes)
    visible_edges = visible_edges.loc[
        visible_edges["genre_node"].isin(core_nodes)
        & visible_edges["tag_node"].isin(core_nodes)
    ].copy()

    communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        seed=42,
    )
    communities.sort(
        key=lambda community: (
            -len(community),
            min((node[0], node[1]) for node in community),
        )
    )
    community_by_node = {
        node: community_id
        for community_id, community in enumerate(communities, start=1)
        for node in community
    }
    positions = nx.spring_layout(
        graph,
        k=5 / math.sqrt(graph.number_of_nodes()),
        iterations=300,
        weight="weight",
        seed=42,
        scale=2.5,
    )
    node_rows = [
        {
            "node": node,
            "node_type": node[0],
            "name": node[1],
            "degree": graph.degree(node),
            "game_count": (
                genre_counts[node[1]] if node[0] == "Genre" else tag_counts[node[1]]
            ),
            "community": community_by_node[node],
            "x": float(position[0]),
            "y": float(position[1]),
        }
        for node, position in positions.items()
    ]
    edge_rows = []
    for edge in visible_edges.itertuples(index=False):
        genre_position = positions[edge.genre_node]
        tag_position = positions[edge.tag_node]
        edge_rows.append(
            {
                "genre": edge.genre,
                "tag": edge.tag,
                "genre_node": edge.genre_node,
                "tag_node": edge.tag_node,
                "shared_games": cast(int, edge.shared_games),
                "x0": float(genre_position[0]),
                "y0": float(genre_position[1]),
                "x1": float(tag_position[0]),
                "y1": float(tag_position[1]),
            }
        )
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def get_genre_review_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Expand valid unique games to one row per game and distinct genre."""
    required = {
        "app_id",
        "name",
        "genres",
        "review_score",
        "review_score_description",
    }
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")

    selected_columns = [
        "app_id",
        "name",
        "genres",
        "review_score",
        "review_score_description",
    ]
    if "positive_ratio" in df.columns:
        selected_columns.append("positive_ratio")
    source = df[selected_columns].copy()
    missing_app_id_rows = int(source["app_id"].isna().sum())
    source = source.dropna(subset=["app_id"])
    duplicate_rows = int(source.duplicated(subset=["app_id"]).sum())
    games = source.drop_duplicates(subset=["app_id"], keep="first").copy()

    raw_scores = games["review_score"]
    score_text = raw_scores.astype("string").str.strip()
    missing_score_mask = raw_scores.isna() | score_text.eq("")
    numeric_scores = pd.to_numeric(raw_scores, errors="coerce")
    valid_score_mask = numeric_scores.between(0, 10, inclusive="both")
    invalid_score_mask = ~missing_score_mask & ~valid_score_mask
    games["review_score"] = numeric_scores

    missing_genre_games = sum(
        not _split_comma_values(value) for value in games["genres"]
    )
    valid_games = games.loc[valid_score_mask].copy()
    valid_games["review_score_description"] = (
        valid_games["review_score_description"].fillna("Unknown").astype(str)
    )
    valid_games["name"] = valid_games["name"].fillna("Unknown title").astype(str)
    valid_games["genre"] = valid_games["genres"].fillna("").astype(str).str.split(",")
    expanded = valid_games.explode("genre")
    expanded["genre"] = expanded["genre"].astype("string").str.strip()
    expanded = expanded.loc[expanded["genre"].notna() & expanded["genre"].ne("")]
    expanded = expanded.drop_duplicates(subset=["app_id", "genre"])

    diagnostics = {
        "unique_games": len(games),
        "missing_app_id_rows": missing_app_id_rows,
        "duplicate_rows": duplicate_rows,
        "missing_scores": int(missing_score_mask.sum()),
        "invalid_scores": int(invalid_score_mask.sum()),
        "valid_scores": int(valid_score_mask.sum()),
        "missing_genre_games": missing_genre_games,
        "missing_description_games": int(
            games["review_score_description"].isna().sum()
        ),
    }
    if "positive_ratio" in games.columns:
        diagnostics["missing_positive_ratio_games"] = int(
            games["positive_ratio"].isna().sum()
        )
    return expanded, diagnostics


def create_genre_review_box_plot(
    data: pd.DataFrame,
    selected_genres: list[str],
    selected_categories: list[str],
    minimum_game_count: int,
    color_by_positive_ratio: bool,
) -> go.Figure:
    filtered = data[
        data["genre"].isin(selected_genres)
        & data["review_score_description"].isin(selected_categories)
    ].copy()
    genre_counts = filtered.groupby("genre")["app_id"].nunique()
    eligible_genres = genre_counts[genre_counts >= minimum_game_count].index.tolist()
    filtered = filtered.loc[filtered["genre"].isin(eligible_genres)]
    genre_order = (
        genre_counts.loc[eligible_genres].sort_values(ascending=False).index.tolist()
    )

    color_column = None
    ratio_order = ["0–25%", "25–50%", "50–75%", "75–100%", "Unknown"]
    if color_by_positive_ratio and "positive_ratio" in filtered.columns:
        ratio = pd.to_numeric(filtered["positive_ratio"], errors="coerce")
        filtered["positive_ratio_band"] = (
            pd.cut(
                ratio,
                bins=[-0.01, 25, 50, 75, 100],
                labels=ratio_order[:4],
                include_lowest=True,
            )
            .astype("string")
            .fillna("Unknown")
        )
        color_column = "positive_ratio_band"

    hover_data: dict[str, object] = {
        "name": True,
        "genre": False,
        "review_score": ":.2f",
        "review_score_description": True,
        "app_id": False,
    }
    if "positive_ratio" in filtered.columns:
        hover_data["positive_ratio"] = ":.1f"

    figure = px.box(
        filtered,
        x="genre",
        y="review_score",
        color=color_column,
        points="outliers",
        category_orders={
            "genre": genre_order,
            "positive_ratio_band": ratio_order,
        },
        hover_data=hover_data,
        color_discrete_sequence=px.colors.sequential.Viridis if color_column else None,
        title="Review Score Distribution Across Game Genres",
        labels={
            "genre": "Game Genre",
            "review_score": "Review Score",
            "positive_ratio_band": "Positive Ratio",
        },
    )
    figure.update_layout(xaxis_title="Game Genre", yaxis_title="Review Score")
    return figure


def render_genre_distribution(df: pd.DataFrame) -> None:
    st.subheader("1. Steam Game Genre Distribution")

    top_n = st.slider(
        "Select number of top genres to display:",
        min_value=5,
        max_value=25,
        value=10,
        step=5,
        key="mehmet_genre_limit",
    )

    genre_counts = get_top_genre_counts(df, limit=top_n)
    fig = create_genre_bar_chart(genre_counts, limit=top_n)
    st.plotly_chart(fig, use_container_width=True)


def render_genre_tag_network(df: pd.DataFrame) -> None:
    edges, genre_counts, tag_counts, _ = get_genre_tag_relationships(df)
    if not genre_counts or not tag_counts or edges.empty:
        st.info("No genre/tag associations are available in this dataset.")
        return

    max_common_games = max(1, int(edges["shared_games"].max()))
    minimum_shared_games = st.slider(
        "Minimum shared games",
        min_value=1,
        max_value=max_common_games,
        value=min(50, max_common_games),
        key="mehmet_network_min_common",
    )
    figure = create_genre_tag_network(
        edges,
        genre_counts,
        tag_counts,
        minimum_shared_games,
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"scrollZoom": True, "displayModeBar": True},
    )


def render_genre_review_box_plot(df: pd.DataFrame) -> None:
    data, diagnostics = get_genre_review_data(df)
    st.caption(
        "Data quality — "
        f"{diagnostics['unique_games']:,} unique games; "
        f"{diagnostics['missing_app_id_rows']:,} missing app IDs; "
        f"{diagnostics['duplicate_rows']:,} duplicate rows removed; "
        f"{diagnostics['valid_scores']:,} valid scores; "
        f"{diagnostics['missing_scores']:,} missing scores; "
        f"{diagnostics['invalid_scores']:,} invalid/out-of-range scores; "
        f"{diagnostics['missing_genre_games']:,} games missing genres; "
        f"{diagnostics['missing_description_games']:,} missing score descriptions."
    )
    if "missing_positive_ratio_games" in diagnostics:
        st.caption(
            f"{diagnostics['missing_positive_ratio_games']:,} games missing positive ratio."
        )
    if data.empty:
        st.info("No games with a valid review score and genre are available.")
        return

    genre_counts = data.groupby("genre")["app_id"].nunique().sort_values(
        ascending=False
    )
    categories = sorted(data["review_score_description"].dropna().unique().tolist())
    controls = st.columns([2, 1, 2, 1])
    selected_genres = controls[0].multiselect(
        "Genres",
        options=genre_counts.index.tolist(),
        default=genre_counts.index.tolist(),
        key="mehmet_review_genres",
    )
    maximum_game_count = max(1, int(genre_counts.max()))
    minimum_game_count = controls[1].slider(
        "Minimum game count",
        min_value=1,
        max_value=maximum_game_count,
        value=min(20, maximum_game_count),
        key="mehmet_review_min_games",
    )
    selected_categories = controls[2].multiselect(
        "Review score categories",
        options=categories,
        default=categories,
        key="mehmet_review_categories",
    )
    color_by_positive_ratio = controls[3].checkbox(
        "Color by positive ratio",
        value=False,
        disabled="positive_ratio" not in data.columns,
        key="mehmet_review_color_ratio",
    )
    figure = create_genre_review_box_plot(
        data,
        selected_genres,
        selected_categories,
        minimum_game_count,
        color_by_positive_ratio,
    )
    st.plotly_chart(figure, use_container_width=True)


def render_mehmet(df: pd.DataFrame) -> None:
    chart = st.selectbox(
        "Select a visualization:",
        [
            "Genre Distribution",
            "Genre – Tags Relationship Network",
            "Genre – Review Score Box Plot",
        ],
        key="mehmet_chart_selection",
    )
    if chart == "Genre Distribution":
        render_genre_distribution(df)
    elif chart == "Genre – Tags Relationship Network":
        render_genre_tag_network(df)
    else:
        render_genre_review_box_plot(df)