import streamlit as st
import pandas as pd
import plotly.express as px


def prepare_treemap_data(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df[["app_id", "name", "genres", "positive_ratio", "total"]].dropna(
        subset=["name", "genres", "positive_ratio", "total"]
    ).copy()
    work_df = work_df[work_df["total"] > 0]

    # Oyunların ilk mekanik ana türünü seç (Free to Play veya Early Access değilse öncelik ver)
    def pick_primary_genre(genre_str):
        genres = [g.strip() for g in str(genre_str).split(",") if g.strip()]
        for g in genres:
            if g not in ("Free to Play", "Early Access"):
                return g
        return genres[0] if genres else ""

    work_df["primary_genre"] = work_df["genres"].apply(pick_primary_genre)
    work_df = work_df[work_df["primary_genre"] != ""].copy()

    # En çok incelemeye sahip 10 ana türü al
    top_genres = (
        work_df.groupby("primary_genre")["total"]
        .sum()
        .nlargest(10)
        .index.tolist()
    )
    work_df = work_df[work_df["primary_genre"].isin(top_genres)].copy()

    # Her türdeki en popüler 15 oyunu seç
    work_df = (
        work_df.sort_values(by="total", ascending=False)
        .groupby("primary_genre", as_index=False)
        .head(15)
    )
    return work_df


def render_treemap(df: pd.DataFrame):
    st.subheader("Market & Genre Hierarchy (Treemap)")
    st.markdown(
        "Top 10 primary genres and their top 15 most-reviewed games. "
        "Box size represents **Total Reviews**, while color reflects **Positive Review Ratio (%)**."
    )

    treemap_df = prepare_treemap_data(df)

    fig = px.treemap(
        treemap_df,
        path=[px.Constant("Steam Ecosystem"), "primary_genre", "name"],
        values="total",
        color="positive_ratio",
        color_continuous_scale="RdYlGn",
        range_color=[45, 100],
        title="Steam Genre Hierarchy by Total Reviews (Colored by Positive Review %)",
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Total Reviews: %{value:,.0f}<br>"
            "Positive Ratio: %{color:.1f}%<extra></extra>"
        ),
        textinfo="label+value+percent parent",
    )

    fig.update_layout(
        height=680,
        margin=dict(t=50, l=10, r=10, b=10),
        coloraxis_colorbar=dict(title="Positive %"),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_parallel_coordinates(df: pd.DataFrame):
    pass


def render_sankey(df: pd.DataFrame):
    pass


def render_demir(df: pd.DataFrame):
    selected_viz = st.selectbox(
        "Select Visualization Type:",
        options=[
            "Treemap (Market & Genre Hierarchy)",
            "Parallel Coordinates (Multi-Dimensional Success Profile)",
            "Sankey Diagram (Pricing to Review Score Flow)",
        ],
        key="demir_viz_selector",
    )

    st.divider()

    if selected_viz == "Treemap (Market & Genre Hierarchy)":
        render_treemap(df)
    elif selected_viz == "Parallel Coordinates (Multi-Dimensional Success Profile)":
        render_parallel_coordinates(df)
    elif selected_viz == "Sankey Diagram (Pricing to Review Score Flow)":
        render_sankey(df)