import streamlit as st
from src.data_loader import load_data
from src.mehmet import render_mehmet
from src.eda import render_eda
from src.demir import render_demir

st.set_page_config(
    page_title="Steam Games Data Visualization",
    page_icon="🎮",
    layout="wide"
)

def main():
    df = load_data()

    st.title("🎮 Steam Dataset Visual Analytics")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Mehmet",
        "Eda",
        "Demir",
        "Dataset Preview"
    ])

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