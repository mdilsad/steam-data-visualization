import streamlit as st
from src.data_loader import load_data
from src.mehmet import render_mehmet
from src.eda import render_eda
from src.demir import render_demir

st.set_page_config(
    page_title="Steam Games Data Analyis",
    page_icon="🎮",
    layout="wide"
)

def main():
    df = load_data()

    st.title("🎮 Steam Games Data Analysis")

    tab_about, tab1, tab2, tab3, tab4 = st.tabs([
        "About",
        "Mehmet",
        "Eda",
        "Demir",
        "Dataset Preview"
    ])

    with tab_about:
        st.subheader("What is this project about?")
        st.markdown("This dashboard analyzes 2024 data about game genres, "
                    "community tags, player counts, review scores, prices, "
                    "developers and similar data of 88,000 games on Steam.")

    with tab1:
        render_mehmet(df)

    with tab2:
        render_eda(df)

    with tab3:
        render_demir(df)

    with tab4:
        st.dataframe(df.head(500), use_container_width=True)

if __name__ == "__main__":
    main()