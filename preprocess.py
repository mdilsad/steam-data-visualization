import os
import re
import json
import numpy as np
import pandas as pd

RAW_DATA_DIR = "data-raw"
OUTPUT_FILE = "clean_steam.parquet"

def clean_html(text):
    """Metin içindeki HTML etiketlerini ve döküntüleri temizler."""
    if pd.isna(text):
        return ""
    clean = re.sub(r"<[^>]+>", "", str(text))
    return clean.replace("*", "").replace("languages with full audio support", "").strip()

def parse_price(price_json_str, is_free):
    """price_overview JSON stringinden nihai fiyatı (EUR) çeker."""
    if is_free == 1 or str(is_free) == "1":
        return 0.0
    if pd.isna(price_json_str):
        return np.nan
    try:
        data = json.loads(price_json_str)
        return float(data.get("final", 0)) / 100.0
    except (json.JSONDecodeError, TypeError, ValueError):
        return np.nan

def parse_owners_midpoint(val):
    """'10,000,000 .. 20,000,000' formatını sayısal ortalamaya çevirir."""
    if pd.isna(val):
        return np.nan
    parts = str(val).replace(",", "").split("..")
    if len(parts) == 2:
        try:
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            return (low + high) / 2.0
        except ValueError:
            return np.nan
    return np.nan

def clean_app_id(df):
    """app_id sütununu int64 yapar, bozuk satırları atar."""
    df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce")
    df = df.dropna(subset=["app_id"]).copy()
    df["app_id"] = df["app_id"].astype("int64")
    return df

def filter_ascii_text(series):
    """Bozuk karakterli (Mojibake) ve İngilizce olmayan tür/etiketleri eler."""
    def clean_item(val):
        if pd.isna(val):
            return ""
        items = [i.strip() for i in str(val).split(",") if re.match(r"^[\w\s\-\'\&]+$", i.strip())]
        return ", ".join(dict.fromkeys(items))
    return series.apply(clean_item)

def main():
    print("Nihai Preprocess işlemi başlatılıyor...")

    # 1. Çoklu Tablolar (genres, categories, tags)
    print("1/5: genres, categories, tags işleniyor...")
    genres_df = clean_app_id(pd.read_csv(os.path.join(RAW_DATA_DIR, "genres.csv"), on_bad_lines="skip", low_memory=False))
    genres_grouped = genres_df.groupby("app_id")["genre"].apply(
        lambda x: ", ".join(x.dropna().astype(str).unique())
    ).rename("genres").reset_index()
    genres_grouped["genres"] = filter_ascii_text(genres_grouped["genres"])

    categories_df = clean_app_id(pd.read_csv(os.path.join(RAW_DATA_DIR, "categories.csv"), on_bad_lines="skip", low_memory=False))
    categories_grouped = categories_df.groupby("app_id")["category"].apply(
        lambda x: ", ".join(x.dropna().astype(str).unique())
    ).rename("categories").reset_index()
    categories_grouped["categories"] = filter_ascii_text(categories_grouped["categories"])

    tags_df = clean_app_id(pd.read_csv(os.path.join(RAW_DATA_DIR, "tags.csv"), on_bad_lines="skip", low_memory=False))
    tags_grouped = tags_df.groupby("app_id")["tag"].apply(
        lambda x: ", ".join(x.dropna().astype(str).unique())
    ).rename("tags").reset_index()
    tags_grouped["tags"] = filter_ascii_text(tags_grouped["tags"])

    # 2. games.csv
    print("2/5: games.csv işleniyor...")
    games_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "games.csv"), escapechar="\\", on_bad_lines="skip", low_memory=False)
    games_df = clean_app_id(games_df)
    games_df = games_df[games_df["type"] == "game"].copy()
    games_df = games_df.drop_duplicates(subset=["app_id"])
    
    games_df["release_date"] = pd.to_datetime(games_df["release_date"], errors="coerce")
    games_df["release_year"] = games_df["release_date"].dt.year
    games_df["is_free"] = pd.to_numeric(games_df["is_free"], errors="coerce").fillna(0).astype(int)
    games_df["price"] = [parse_price(p, f) for p, f in zip(games_df["price_overview"], games_df["is_free"])]
    games_df["clean_languages"] = games_df["languages"].apply(clean_html)
    games_df = games_df.drop(columns=["price_overview", "languages", "type"], errors="ignore")

    # 3. reviews.csv
    print("3/5: reviews.csv işleniyor...")
    reviews_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "reviews.csv"), na_values=["\\N", r"\N", ""], escapechar="\\", on_bad_lines="skip", low_memory=False)
    reviews_df = clean_app_id(reviews_df)
    reviews_df = reviews_df.drop_duplicates(subset=["app_id"])
    
    # reviews, steamspy_user_score, steamspy_score_rank, steamspy_positive, steamspy_negative atılıyor
    cols_to_drop = ["reviews", "steamspy_user_score", "steamspy_score_rank", "steamspy_positive", "steamspy_negative"]
    reviews_df = reviews_df.drop(columns=cols_to_drop, errors="ignore")

    numeric_review_cols = ["review_score", "positive", "negative", "total", "metacritic_score", "recommendations"]
    for col in numeric_review_cols:
        if col in reviews_df.columns:
            reviews_df[col] = pd.to_numeric(reviews_df[col], errors="coerce")
            
    reviews_df["positive_ratio"] = np.where(
        reviews_df["total"] > 0,
        (reviews_df["positive"] / reviews_df["total"]) * 100,
        np.nan
    )

    # 4. steamspy_insights.csv
    print("4/5: steamspy_insights.csv işleniyor...")
    steamspy_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "steamspy_insights.csv"), na_values=["\\N", r"\N", ""], escapechar="\\", on_bad_lines="skip", low_memory=False)
    steamspy_df = clean_app_id(steamspy_df)
    steamspy_df = steamspy_df.drop_duplicates(subset=["app_id"])
    
    steamspy_df["concurrent_users"] = pd.to_numeric(steamspy_df["concurrent_users_yesterday"], errors="coerce").fillna(0)
    steamspy_df["estimated_owners"] = steamspy_df["owners_range"].apply(parse_owners_midpoint)

    # Ölü playtime_* ve diğer gereksiz alanları tamamen attık
    steamspy_df = steamspy_df[["app_id", "developer", "publisher", "estimated_owners", "concurrent_users"]]

    # 5. Birleştirme (Merge)
    print("5/5: Tablolar birleştiriliyor...")
    merged = games_df.merge(reviews_df, on="app_id", how="left")
    merged = merged.merge(steamspy_df, on="app_id", how="left")
    merged = merged.merge(genres_grouped, on="app_id", how="left")
    merged = merged.merge(categories_grouped, on="app_id", how="left")
    merged = merged.merge(tags_grouped, on="app_id", how="left")

    # En az 1 incelemesi olan oyunları tut
    merged = merged[merged["total"].fillna(0) > 0].copy()
    merged = merged.dropna(subset=["name"])

    # Parquet olarak kaydet
    merged.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ Nihai optimize parquet hazır: {OUTPUT_FILE}")
    print(f"Toplam Oyun Sayısı: {len(merged)}")
    print(f"Temizlenmiş Sütun Listesi ({len(merged.columns)} adet): {list(merged.columns)}")

if __name__ == "__main__":
    main()