import os
import pandas as pd
import streamlit as st

DATA_PATH = "clean_steam.parquet"

@st.cache_data(show_spinner="Veri seti yükleniyor...")
def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """Temizlenmiş Steam Parquet verisini okur ve Streamlit oturumunda önbelleğe alır."""
    if not os.path.exists(filepath):
        # Eğer kök dizinde değilse bir üst dizin kontrolü (ör. testler için)
        alt_path = os.path.join("..", filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(
                f"'{filepath}' bulunamadı. Lütfen önce preprocess.py betiğini çalıştırın."
            )
            
    df = pd.read_parquet(filepath)
    
    # Tarih sütununun datetime olduğundan emin ol
    if "release_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["release_date"]):
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        
    return df


def get_unique_genres(df: pd.DataFrame) -> list:
    """Tüm benzersiz oyun türlerini sıralı liste halinde döner (Filtreleme bileşenleri için)."""
    if "genres" not in df.columns:
        return []
    
    genres_series = (
        df["genres"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
    )
    unique_genres = sorted([g for g in genres_series.unique() if g])
    return unique_genres


def get_unique_categories(df: pd.DataFrame) -> list:
    """Tüm benzersiz kategorileri sıralı liste halinde döner."""
    if "categories" not in df.columns:
        return []
        
    cats_series = (
        df["categories"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
    )
    unique_cats = sorted([c for c in cats_series.unique() if c])
    return unique_cats