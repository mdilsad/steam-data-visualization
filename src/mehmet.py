import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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


def render_mehmet(df: pd.DataFrame):
    st.subheader("1. Steam Game Genre Distribution")
    
    top_n = st.slider(
        "Select number of top genres to display:",
        min_value=5,
        max_value=25,
        value=10,
        step=5,
        key="mehmet_genre_limit"
    )
    
    genre_counts = get_top_genre_counts(df, limit=top_n)
    fig = create_genre_bar_chart(genre_counts, limit=top_n)
    st.plotly_chart(fig, use_container_width=True)