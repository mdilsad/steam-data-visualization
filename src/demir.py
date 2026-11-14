import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- SABİT AYARLAR ---
TREEMAP_TOP_GENRES = 12
TREEMAP_GAMES_PER_GENRE = 15

PARCOORDS_TOP_N = 50
PARCOORDS_MAX_PRICE = 70.0
LINE_WIDTH_DEFAULT = 2.2
LINE_WIDTH_HOVER = 3.5
LINE_OPACITY_DEFAULT = 0.65
LINE_OPACITY_DIM = 0.12


# ==========================================
# 1. TREEMAP FONKSİYONLARI
# ==========================================
def _pick_primary_genre(genre_str: str) -> str:
    """Oyunun ilk mekanik ana türünü seçer (Free to Play / Early Access hariç öncelik verir)."""
    genres = [g.strip() for g in str(genre_str).split(",") if g.strip()]
    for g in genres:
        if g not in ("Free to Play", "Early Access"):
            return g
    return genres[0] if genres else ""


def prepare_treemap_data(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df[["app_id", "name", "genres", "positive_ratio", "total"]].dropna(
        subset=["name", "genres", "positive_ratio", "total"]
    ).copy()
    work_df = work_df[work_df["total"] > 0]

    work_df["primary_genre"] = work_df["genres"].apply(_pick_primary_genre)
    work_df = work_df[work_df["primary_genre"] != ""].copy()

    top_genres = (
        work_df.groupby("primary_genre")["total"]
        .sum()
        .nlargest(TREEMAP_TOP_GENRES)
        .index.tolist()
    )
    work_df = work_df[work_df["primary_genre"].isin(top_genres)].copy()

    return (
        work_df.sort_values(by="total", ascending=False)
        .groupby("primary_genre", as_index=False)
        .head(TREEMAP_GAMES_PER_GENRE)
    )


def render_treemap(df: pd.DataFrame):
    st.subheader("Genre Hierarchy by Total Reviews")
    st.markdown(
        f"Top {TREEMAP_TOP_GENRES} primary genres and their top {TREEMAP_GAMES_PER_GENRE} most-reviewed games. "
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
            "Positive Review Ratio: %{color:.1f}%<extra></extra>"
        ),
        textinfo="label+value+percent parent",
    )

    fig.update_layout(
        height=680,
        margin=dict(t=50, l=10, r=10, b=10),
        coloraxis_colorbar=dict(title="Positive %"),
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 2. PARALLEL COORDINATES FONKSİYONLARI
# ==========================================
def prepare_parallel_coordinates_data(
    df: pd.DataFrame,
    selected_genre: str = "All",
    top_n: int = PARCOORDS_TOP_N,
) -> pd.DataFrame:
    cols = [
        "app_id", "name", "genres", "release_year",
        "price", "estimated_owners", "total", "positive_ratio",
    ]
    work_df = df[cols].dropna(
        subset=["release_year", "price", "estimated_owners", "total", "positive_ratio"]
    ).copy()

    work_df = work_df[
        (work_df["price"] >= 0)
        & (work_df["price"] <= PARCOORDS_MAX_PRICE)
        & (work_df["total"] > 0)
        & (work_df["estimated_owners"] > 0)
    ]

    if selected_genre != "All":
        work_df = work_df[
            work_df["genres"].fillna("").apply(
                lambda g: selected_genre in [i.strip() for i in str(g).split(",")]
            )
        ]

    work_df = work_df.sort_values(by="total", ascending=False).head(top_n).copy()
    work_df["log_owners"] = np.log10(work_df["estimated_owners"])
    work_df["log_reviews"] = np.log10(work_df["total"])
    return work_df


def _build_parcoords_axes(df: pd.DataFrame) -> list[dict]:
    """5 paralel eksenin alt/üst sınırlarını ve etiketlerini yapılandırır."""
    min_year, max_year = int(df["release_year"].min()), int(df["release_year"].max())
    if min_year == max_year:
        min_year, max_year = min_year - 1, max_year + 1
    year_ticks = sorted({min_year, (min_year + max_year) // 2, max_year})

    owner_vals = [10_000, 100_000, 1_000_000, 10_000_000, 75_000_000]
    owner_text = ["10K", "100K", "1M", "10M", "75M"]

    rev_candidates = [
        (500, "500"), (2_500, "2.5K"), (10_000, "10K"),
        (50_000, "50K"), (250_000, "250K"), (1_000_000, "1M"), (2_500_000, "2.5M"),
    ]
    min_rev = float(df["total"].min()) * 0.9
    max_rev = max(2_500_000.0, float(df["total"].max()))
    active_rev = [(v, t) for v, t in rev_candidates if min_rev <= v <= max_rev]

    return [
        {
            "label": "Release Year", "col": "release_year",
            "min": float(min_year), "max": float(max_year),
            "tickvals": [float(y) for y in year_ticks],
            "ticktext": [str(y) for y in year_ticks],
        },
        {
            "label": "Price (€)", "col": "price",
            "min": 0.0, "max": PARCOORDS_MAX_PRICE,
            "tickvals": [0.0, 15.0, 30.0, 50.0, 70.0],
            "ticktext": ["€0", "€15", "€30", "€50", "€70"],
        },
        {
            "label": "Estimated Owners", "col": "log_owners",
            "min": np.log10(10_000), "max": max(np.log10(75_000_000), float(df["log_owners"].max())),
            "tickvals": [float(np.log10(v)) for v in owner_vals],
            "ticktext": owner_text,
        },
        {
            "label": "Total Reviews", "col": "log_reviews",
            "min": np.log10(min_rev), "max": np.log10(max_rev),
            "tickvals": [float(np.log10(v)) for v, _ in active_rev],
            "ticktext": [t for _, t in active_rev],
        },
        {
            "label": "Positive Review Ratio (%)", "col": "positive_ratio",
            "min": 40.0, "max": 100.0,
            "tickvals": [40.0, 55.0, 70.0, 85.0, 100.0],
            "ticktext": ["40%", "55%", "70%", "85%", "100%"],
        },
    ]


def _create_parcoords_figure(df: pd.DataFrame, axes: list[dict], title: str) -> go.Figure:
    """Oyun çizgilerini, dikey eksenleri ve renk skalasını içeren Plotly figürünü üretir."""
    fig = go.Figure()
    num_axes = len(axes)
    dense_steps = 25

    # 1. Oyun çizgilerini ekle
    for _, row in df.iterrows():
        y_norm = [
            float(np.clip((float(row[ax["col"]]) - ax["min"]) / max(ax["max"] - ax["min"], 1e-9), 0.0, 1.0))
            for ax in axes
        ]

        x_pts, y_pts = [], []
        for i in range(num_axes - 1):
            x_pts.extend(np.linspace(i, i + 1, dense_steps, endpoint=False).tolist())
            y_pts.extend(np.linspace(y_norm[i], y_norm[i + 1], dense_steps, endpoint=False).tolist())
        x_pts.append(float(num_axes - 1))
        y_pts.append(y_norm[-1])

        color_ratio = float(np.clip((row["positive_ratio"] - 45.0) / 55.0, 0.0, 1.0))
        line_color = px.colors.sample_colorscale("RdYlGn", [color_ratio])[0]

        hover_text = (
            f"<b>{row['name']}</b><br>"
            f"Release Year: {int(row['release_year'])}<br>"
            f"Price: €{row['price']:.2f}<br>"
            f"Estimated Owners: {int(row['estimated_owners']):,}<br>"
            f"Total Reviews: {int(row['total']):,}<br>"
            f"Positive Review Ratio: {row['positive_ratio']:.1f}%<extra></extra>"
        )

        is_axis_node = [float(x).is_integer() for x in x_pts]
        fig.add_trace(
            go.Scatter(
                x=x_pts,
                y=y_pts,
                mode="lines+markers",
                line=dict(color=line_color, width=LINE_WIDTH_DEFAULT),
                marker=dict(
                    size=[7 if on_ax else 12 for on_ax in is_axis_node],
                    color=line_color,
                    opacity=[0.95 if on_ax else 0.001 for on_ax in is_axis_node],
                ),
                opacity=LINE_OPACITY_DEFAULT,
                hovertemplate=hover_text,
                showlegend=False,
            )
        )

    # 2. Dikey eksenleri, eksen başlıklarını ve kılavuz etiketlerini ekle
    annotations = []
    for idx, ax in enumerate(axes):
        fig.add_shape(
            type="line", x0=idx, y0=0, x1=idx, y1=1,
            line=dict(color="rgba(160, 174, 192, 0.85)", width=2.5),
            layer="below",
        )
        # Eksen üst başlığı (tam eksenin tepesinde)
        annotations.append(
            dict(
                x=idx,
                y=1.11,
                xref="x",
                yref="y",
                text=f"<b>{ax['label']}</b>",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=13, color="#f8fafc"),
                bgcolor="rgba(30, 41, 59, 0.9)",
                bordercolor="rgba(148, 163, 184, 0.5)",
                borderpad=5,
            )
        )
        # Eksen üzerindeki kılavuz değerleri
        for t_val, t_txt in zip(ax["tickvals"], ax["ticktext"]):
            y_pos = float(np.clip((t_val - ax["min"]) / max(ax["max"] - ax["min"], 1e-9), 0.0, 1.0))
            annotations.append(
                dict(
                    x=idx - 0.03,
                    y=y_pos,
                    xref="x",
                    yref="y",
                    text=t_txt,
                    showarrow=False,
                    xanchor="right",
                    font=dict(size=11, color="#f8fafc"),
                    bgcolor="rgba(15, 23, 42, 0.85)",
                    bordercolor="rgba(148, 163, 184, 0.4)",
                    borderpad=3,
                )
            )

    # 3. Renk skalası (Colorbar)
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(
                colorscale="RdYlGn", cmin=45, cmax=100, color=[45, 100],
                showscale=True, colorbar=dict(title="Positive %"),
            ),
            hoverinfo="none", showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#f8fafc", size=17)),
        height=670,
        margin=dict(t=105, l=65, r=60, b=30),
        xaxis=dict(range=[-0.42, num_axes - 0.58], visible=False, fixedrange=True),
        yaxis=dict(range=[-0.05, 1.24], visible=False, fixedrange=True),
        annotations=annotations,
        hovermode="closest",
        hoverlabel=dict(bgcolor="#1e293b", font_size=13, font_color="#f8fafc"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _render_interactive_parcoords(fig: go.Figure, num_games: int):
    """Plotly figürüne JS tabanlı çizgi parlama (highlight) efekti ekleyerek ekrana basar."""
    plot_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id="parcoords_hover_chart",
        config={"displayModeBar": True, "responsive": True},
    )

    html_block = f"""
    <style>
        body {{ margin: 0; background: transparent; color: #f8fafc; font-family: sans-serif; }}
    </style>
    {plot_html}
    <script>
    (function() {{
        const gd = document.getElementById('parcoords_hover_chart');
        const numGames = {num_games};
        if (!gd) return;

        const indices = Array.from({{length: numGames}}, (_, i) => i);

        gd.on('plotly_hover', function(data) {{
            if (!data.points || !data.points.length) return;
            const activeIdx = data.points[0].curveNumber;
            if (activeIdx >= numGames) return;

            const opacities = indices.map(i => i === activeIdx ? 1.0 : {LINE_OPACITY_DIM});
            const widths = indices.map(i => i === activeIdx ? {LINE_WIDTH_HOVER} : 1.3);
            Plotly.restyle(gd, {{'opacity': opacities, 'line.width': widths}}, indices);
        }});

        gd.on('plotly_unhover', function() {{
            const opacities = Array(numGames).fill({LINE_OPACITY_DEFAULT});
            const widths = Array(numGames).fill({LINE_WIDTH_DEFAULT});
            Plotly.restyle(gd, {{'opacity': opacities, 'line.width': widths}}, indices);
        }});
    }})();
    </script>
    """
    components.html(html_block, height=690)


def render_parallel_coordinates(df: pd.DataFrame):
    st.subheader("Game Success Profile")
    st.markdown(
        f"**Top {PARCOORDS_TOP_N} most-reviewed games** priced up to **€{int(PARCOORDS_MAX_PRICE)}** across "
        "**Release Year**, **Price**, **Estimated Owners**, **Total Reviews**, and **Positive Review Ratio (%)**. "
    )

    available_genres = sorted(
        {g.strip() for s in df["genres"].dropna() for g in str(s).split(",") if g.strip()}
    )
    selected_genre = st.selectbox(
        "Filter by Genre:",
        options=["All"] + available_genres,
        index=0,
        key="demir_parcoords_genre",
    )

    parcoords_df = prepare_parallel_coordinates_data(df, selected_genre, PARCOORDS_TOP_N)
    if parcoords_df.empty:
        st.warning("No games match the selected genre.")
        return

    axes = _build_parcoords_axes(parcoords_df)
    title_genre = "All Genres" if selected_genre == "All" else selected_genre
    fig = _create_parcoords_figure(
        parcoords_df, axes, title=f"Multi-Dimensional Profile: Top {len(parcoords_df)} Games ({title_genre})"
    )
    _render_interactive_parcoords(fig, len(parcoords_df))


# ==========================================
# 3. SANKEY VE ANA ROUTER FONKSİYONLARI
# ==========================================
def render_sankey(df: pd.DataFrame):
    pass


def render_demir(df: pd.DataFrame):
    selected_viz = st.selectbox(
        "Select Visualization:",
        options=[
            "Genre Hierarchy by Total Reviews",
            "Game Success Profile",
            "Sankey Diagram (Pricing to Review Score Flow)",
        ],
        key="demir_viz_selector",
    )

    st.divider()

    if selected_viz == "Genre Hierarchy by Total Reviews":
        render_treemap(df)
    elif selected_viz == "Game Success Profile":
        render_parallel_coordinates(df)
    elif selected_viz == "Sankey Diagram (Pricing to Review Score Flow)":
        render_sankey(df)