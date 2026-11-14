
import pandas as pd
import plotly.express as px
import streamlit as st

def render_eda(df: pd.DataFrame):
    # display three charts, one at a time

    chart = st.selectbox(
        "Select a visualization:",
        [
            "Positive Review Ratio by Game Genre (Violin Plot)",
            "Review Volume vs. Positive Rating (Density Contour)",
            "Steam Game Releases by Year (Line Chart)",
        ],
        key="eda_chart_selection",
    )

    # copy the data
    data = df.copy()

    # convert columns to correct data types
    data["total"] = pd.to_numeric(data["total"], errors="coerce")
    data["positive_ratio"] = pd.to_numeric(
        data["positive_ratio"], errors="coerce"
    )
    data["release_date"] = pd.to_datetime(
        data["release_date"], errors="coerce"
    )

    data["year"] = data["release_date"].dt.year
    data["genres"] = data["genres"].fillna("")

    # create interactive filters
    genres = (
        data["genres"]
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(15)
        .index.tolist()
    )

    genre = st.selectbox(
        "Game genre",
        ["All genres"] + genres,
        key="eda_genre"
    )

    min_reviews = st.slider(
        "Minimum number of reviews",
        1, 1000, 20,
        key="eda_min_reviews"
    )

    include_recent = st.checkbox(
        "Only games released from 2010 onwards",
        value=False,
        key="eda_recent"
    )

    # filter games by genre
    if genre != "All genres":
        data = data[
            data["genres"].str.split(",").apply(
                lambda values: genre in [
                    v.strip() for v in values
                ]
            )
        ]

    # filter games by release year
    if include_recent:
        data = data[data["year"] >= 2010]

    # prepare review data
    review_data = data.dropna(
        subset=["positive_ratio", "total"]
    ).copy()

    review_data = review_data[
        (review_data["total"] >= min_reviews)
        & review_data["positive_ratio"].between(0, 100)
    ]

    # chart 1: violin plot
    if chart == "Positive Review Ratio by Game Genre (Violin Plot)":

        st.subheader(
            "1. Positive Review Ratio by Game Genre (Violin Plot)"
        )

        st.caption(
            "The shape shows how positive review percentages "
            "vary between genres. A game may belong to multiple genres."
        )

        genre_rows = review_data[
            ["app_id", "genres", "positive_ratio"]
        ].copy()

        genre_rows["genres"] = genre_rows["genres"].str.split(",")
        genre_rows = genre_rows.explode("genres")
        genre_rows["genres"] = genre_rows["genres"].str.strip()

        genre_rows = genre_rows[genre_rows["genres"] != ""]

        top_genres = (
            genre_rows["genres"]
            .value_counts()
            .head(8)
            .index
        )

        genre_rows = genre_rows[
            genre_rows["genres"].isin(top_genres)
        ]

        if not genre_rows.empty:

            fig = px.violin(
                genre_rows,
                x="genres",
                y="positive_ratio",
                color="genres",
                box=True,
                points=False,
                labels={
                    "genres": "Genre",
                    "positive_ratio": "Positive reviews (%)"
                },
                title="Player Satisfaction Distribution Across Genres"
            )

            fig.update_layout(
                showlegend=False,
                xaxis_tickangle=-35
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.info(
                "Insight: Compare each genre's median and variation, "
                "not just its average review score."
            )

        else:
            st.warning(
                "No review data for these filters. "
                "Reduce the minimum review count."
            )

    # chart 2: density contour plot
    elif chart == "Review Volume vs. Positive Rating (Density Contour)":

        st.subheader(
            "2. Review Volume vs. Positive Rating (Density Contour)"
        )

        st.caption(
            "Contours indicate where games are concentrated. "
            "The X axis uses log10(review count)."
        )

        density = review_data.loc[
            review_data["total"] > 0,
            ["total", "positive_ratio"]
        ].copy()

        if len(density) >= 5:

            import numpy as np

            density["log_reviews"] = np.log10(
                density["total"]
            )

            fig = px.density_contour(
                density,
                x="log_reviews",
                y="positive_ratio",
                nbinsx=35,
                nbinsy=30,
                labels={
                    "log_reviews": "Log10(total reviews)",
                    "positive_ratio": "Positive reviews (%)"
                },
                title="Where Are Most Steam Games Concentrated?"
            )

            fig.update_traces(
                contours_coloring="fill",
                contours_showlabels=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.info(
                "Insight: Identify combinations of review volume "
                "and user satisfaction that appear most often."
            )

        else:
            st.warning(
                "Not enough observations for a density contour."
            )

    # chart 3: line chart
    else:

        st.subheader(
            "3. Steam Game Releases by Year (Line Chart)"
        )

        st.caption(
            "This graph counts games with a valid release date "
            "in the filtered dataset."
        )

        releases = data.dropna(
            subset=["year"]
        ).copy()

        releases = releases[
            releases["year"].between(1990, 2026)
        ]

        yearly = (
            releases.groupby("year")["app_id"]
            .nunique()
            .reset_index(name="game_count")
        )

        if not yearly.empty:

            fig = px.line(
                yearly,
                x="year",
                y="game_count",
                markers=True,
                labels={
                    "year": "Release year",
                    "game_count": "Number of games"
                },
                title="How Has the Number of Steam Game Releases Changed?"
            )

            fig.update_layout(
                hovermode="x unified"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            peak = yearly.loc[
                yearly["game_count"].idxmax()
            ]

            st.info(
                f"Insight: {int(peak['year'])} has the highest "
                f"number of releases in the selected data "
                f"({int(peak['game_count']):,} games)."
            )

        else:
            st.warning(
                "No release dates available for these filters."
            )

    # show number of games after filtering
    st.caption(
        f"Games after filtering: {len(data):,}. "
        "The source dataset may not include all Steam games."
    )
