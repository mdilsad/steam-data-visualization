import os
import re
import json
import unicodedata
import numpy as np
import pandas as pd

RAW_DATA_DIR = "data-raw"
OUTPUT_FILE = "clean_steam.parquet"


# --- 1. GENEL ENCODING (MOJIBAKE) VE METİN DÜZELTİCİ ---
def fix_mojibake(text: str) -> str:
    """UTF-8 metnin Latin-1/CP1252 olarak yanlış okunmasından kaynaklanan bozuklukları (Ð­ÐºÑ...) düzeltir."""
    if pd.isna(text):
        return ""
    s = str(text).strip()
    try:
        s = s.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return s


def strip_accents(text: str) -> str:
    """İspanyolca/Fransızca/Lehçe/Türkçe aksanlı harfleri (ó->o, ą->a, ß->ss) düz ASCII'ye yaklaştırır."""
    s = text.replace("ł", "l").replace("Ł", "L").replace("ß", "ss")
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


# --- 2. SÜTUN BAZLI ÇEVİRİ SÖZLÜKLERİ ---
GENRE_PATTERNS = [
    ("Action", ["action", "azione", "akcja", "accion", "acao", "aktion", "actie", "aksiyon", "akcni", "akcio", "экшен", "动作", "アクション"]),
    ("Adventure", ["adventure", "avventura", "przygod", "aventura", "aventure", "abenteuer", "avontuur", "macera", "dobrodruz", "приключ", "冒险", "アドベンチャー"]),
    ("RPG", ["rpg", "role-playing", "role playing", "ruolo", "rol", "fabular", "rollenspiel", "rolev", "ролев", "角色扮演", "ロールプレイング"]),
    ("Strategy", ["strateg", "estrateg", "stratej", "стратег", "策略", "ストラテジー"]),
    ("Simulation", ["simulat", "simulaz", "symulac", "simulac", "simul", "симул", "模拟", "シミュレーション"]),
    ("Sports", ["sport", "deport", "esport", "spor", "спорт", "体育", "スポーツ"]),
    ("Racing", ["racing", "corse", "wyscig", "carreras", "corrida", "renn", "race", "yaris", "гонк", "竞速", "レース"]),
    ("Casual", ["casual", "passatempo", "rekreac", "ocasion", "gelegenheit", "basit eglence", "казуал", "休闲", "カジュアル"]),
    ("Indie", ["indie", "niezalez", "indep", "bagimsiz", "инди", "独立"]),
    ("Massively Multiplayer", ["massively multiplayer", "multigiocatore di massa", "masivo", "massivement", "ммо", "大型多人"]),
    ("Free to Play", ["free to play", "free-to-play", "gratuit", "kostenlos", "darmo", "f2p", "бесплатн", "免费"]),
    ("Early Access", ["early access", "accesso anticipato", "wczesny dostep", "acceso anticipado", "acces anticipe", "ранний доступ", "抢先体验"]),
]

CATEGORY_PATTERNS = [
    ("Single-player", ["single-player", "single player", "giocatore singolo", "jednoosob", "un jugador", "solo", "einzelspieler", "tek oyunculu", "одиноч", "单人"]),
    ("Multi-player", ["multi-player", "multiplayer", "multigiocatore", "wieloosob", "multijugador", "multijoueur", "mehrspieler", "cok oyunculu", "многопольз", "多人"]),
    ("Co-op", ["co-op", "coop", "cooperat", "kooperac", "кооперат", "合作"]),
    ("PvP", ["pvp", "jcj", "против игрок", "玩家对战"]),
    ("Steam Achievements", ["achievement", "osiagniecia", "logros", "succes", "errungenschaften", "basarim", "достижен", "成就"]),
    ("Steam Cloud", ["steam cloud", "chmura", "nube", "облако", "云"]),
    ("Full controller support", ["full controller", "pelna obsluga kontroler", "compatibilidad total con mando", "manette", "полная поддержка контроллер", "完全支持控制器"]),
    ("Partial Controller Support", ["partial controller", "czesciowa obsluga", "parcial", "частичная поддержка", "部分支持控制器"]),
    ("Steam Trading Cards", ["trading card", "karty kolekcjonerskie", "cromos", "cartes a echanger", "sammelkarten", "коллекционные карточки", "集换式卡牌"]),
    ("Steam Workshop", ["workshop", "warsztat", "atelier", "мастерская", "创意工坊"]),
    ("In-App Purchases", ["in-app purchase", "zakupy w aplikacji", "compras dentro", "achats integres", "käufe im spiel", "покупки внутри", "应用内购买"]),
    ("VR Support", ["vr support", "vr only", "steamvr", "tracked motion", "realidad virtual", "wirtualna rzeczywistosc", "виртуальн"]),
]

LANGUAGE_PATTERNS = [
    ("English", ["english", "ingles", "anglais", "englisch", "angielski", "ingilizce", "английский", "英语"]),
    ("German", ["german", "deutsch", "aleman", "allemand", "niemiecki", "almanca", "немецкий", "德语"]),
    ("French", ["french", "francais", "frances", "franzosisch", "francuski", "fransizca", "французский", "法语"]),
    ("Spanish", ["spanish", "espanol", "espagnol", "spanisch", "hiszpanski", "ispanyolca", "испанский", "西班牙语"]),
    ("Italian", ["italian", "italiano", "italien", "italienisch", "wloski", "italyanca", "итальянский", "意大利语"]),
    ("Russian", ["russian", "ruso", "russe", "russisch", "rosyjski", "rusca", "русский", "俄语"]),
    ("Simplified Chinese", ["simplified chinese", "chinois simplifie", "chino simplificado", "упрощенный китайский", "简体中文"]),
    ("Traditional Chinese", ["traditional chinese", "chinois traditionnel", "chino tradicional", "традиционный китайский", "繁體中文"]),
    ("Japanese", ["japanese", "japones", "japonais", "japanisch", "japonski", "japonca", "японский", "日语"]),
    ("Korean", ["korean", "coreano", "coreen", "koreanisch", "koreanski", "korece", "корейский", "韩语"]),
    ("Portuguese", ["portuguese", "portugues", "portugais", "portugiesisch", "portugalski", "portekizce", "португальский", "葡萄牙语"]),
    ("Polish", ["polish", "polaco", "polonais", "polnisch", "polski", "lehce", "польский", "波兰语"]),
    ("Turkish", ["turkish", "turco", "turc", "turkisch", "turecki", "turkce", "турецкий", "土耳其语"]),
]


def normalize_multi_value_str(val, pattern_list, keep_clean_ascii_unmatched=False) -> str:
    """Virgülle ayrılmış metinleri Mojibake'den kurtarır ve İngilizce karşılıklarına çevirir."""
    if pd.isna(val):
        return ""
    fixed_str = fix_mojibake(str(val))
    results = []
    for raw_token in fixed_str.split(","):
        token_orig = raw_token.strip()
        if not token_orig:
            continue
        token_norm = strip_accents(token_orig).lower()
        matched_canonical = None
        for canonical, keywords in pattern_list:
            if any(kw in token_norm for kw in keywords):
                matched_canonical = canonical
                break
        if matched_canonical:
            results.append(matched_canonical)
        elif keep_clean_ascii_unmatched:
            # Tags veya ekstra kategorilerde zaten düzgün İngilizce (ASCII) olanları koru
            clean_ascii = strip_accents(token_orig).strip()
            if clean_ascii.isascii() and re.match(r"^[A-Za-z0-9\s\-\'\&\+]+$", clean_ascii):
                results.append(clean_ascii)
    return ", ".join(dict.fromkeys(results))


def clean_languages_col(text) -> str:
    """HTML etiketlerini temizler ve dil adlarını standart İngilizceye çevirir."""
    if pd.isna(text):
        return ""
    clean = re.sub(r"<[^>]+>", "", str(text))
    clean = clean.replace("*", "").replace("languages with full audio support", "").strip()
    return normalize_multi_value_str(clean, LANGUAGE_PATTERNS, keep_clean_ascii_unmatched=True)


def clean_company_name(val) -> str:
    """Developer ve Publisher sütunlarındaki Mojibake bozukluklarını düzeltir."""
    if pd.isna(val):
        return ""
    s = fix_mojibake(str(val)).strip()
    if not s or s in ("\\N", "N", "nan", "None"):
        return "Unknown"
    return s


def compute_english_review_desc(total, pos_ratio) -> str:
    """İnceleme açıklamalarını %100 standart İngilizce Steam sınıflarına dönüştürür."""
    if pd.isna(total) or total <= 0 or pd.isna(pos_ratio):
        return "No Reviews"
    if pos_ratio >= 95 and total >= 500:
        return "Overwhelmingly Positive"
    if pos_ratio >= 80 and total >= 50:
        return "Very Positive"
    if pos_ratio >= 80:
        return "Positive"
    if pos_ratio >= 70:
        return "Mostly Positive"
    if pos_ratio >= 40:
        return "Mixed"
    if pos_ratio >= 20:
        return "Mostly Negative"
    if total >= 500:
        return "Overwhelmingly Negative"
    if total >= 50:
        return "Very Negative"
    return "Negative"


def parse_price(price_json_str, is_free):
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
    if pd.isna(val):
        return np.nan
    parts = str(val).replace(",", "").split("..")
    if len(parts) == 2:
        try:
            return (float(parts[0].strip()) + float(parts[1].strip())) / 2.0
        except ValueError:
            return np.nan
    return np.nan


def clean_app_id(df):
    df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce")
    df = df.dropna(subset=["app_id"]).copy()
    df["app_id"] = df["app_id"].astype("int64")
    return df


def main():
    print("Tam İngilizce Standardizasyonlu Preprocess başlatılıyor...")

    # 1. Çoklu Tablolar (genres, categories, tags)
    print("1/5: genres, categories, tags İngilizceye çevriliyor ve temizleniyor...")
    genres_df = clean_app_id(pd.read_csv(os.path.join(RAW_DATA_DIR, "genres.csv"), on_bad_lines="skip", low_memory=False))
    genres_grouped = genres_df.groupby("app_id")["genre"].apply(
        lambda x: ", ".join(x.dropna().astype(str).unique())
    ).rename("genres").reset_index()
    genres_grouped["genres"] = genres_grouped["genres"].apply(
        lambda v: normalize_multi_value_str(v, GENRE_PATTERNS, keep_clean_ascii_unmatched=False)
    )

    categories_df = clean_app_id(pd.read_csv(os.path.join(RAW_DATA_DIR, "categories.csv"), on_bad_lines="skip", low_memory=False))
    categories_grouped = categories_df.groupby("app_id")["category"].apply(
        lambda x: ", ".join(x.dropna().astype(str).unique())
    ).rename("categories").reset_index()
    categories_grouped["categories"] = categories_grouped["categories"].apply(
        lambda v: normalize_multi_value_str(v, CATEGORY_PATTERNS, keep_clean_ascii_unmatched=True)
    )

    tags_df = clean_app_id(pd.read_csv(os.path.join(RAW_DATA_DIR, "tags.csv"), on_bad_lines="skip", low_memory=False))
    tags_grouped = tags_df.groupby("app_id")["tag"].apply(
        lambda x: ", ".join(x.dropna().astype(str).unique())
    ).rename("tags").reset_index()
    tags_grouped["tags"] = tags_grouped["tags"].apply(
        lambda v: normalize_multi_value_str(v, GENRE_PATTERNS, keep_clean_ascii_unmatched=True)
    )

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
    games_df["clean_languages"] = games_df["languages"].apply(clean_languages_col)
    games_df = games_df.drop(columns=["price_overview", "languages", "type"], errors="ignore")

    # 3. reviews.csv
    print("3/5: reviews.csv işleniyor...")
    reviews_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "reviews.csv"), na_values=["\\N", r"\N", ""], escapechar="\\", on_bad_lines="skip", low_memory=False)
    reviews_df = clean_app_id(reviews_df)
    reviews_df = reviews_df.drop_duplicates(subset=["app_id"])

    cols_to_drop = ["reviews", "steamspy_user_score", "steamspy_score_rank", "steamspy_positive", "steamspy_negative"]
    reviews_df = reviews_df.drop(columns=cols_to_drop, errors="ignore")

    numeric_review_cols = ["review_score", "positive", "negative", "total", "metacritic_score", "recommendations"]
    for col in numeric_review_cols:
        if col in reviews_df.columns:
            reviews_df[col] = pd.to_numeric(reviews_df[col], errors="coerce")

    reviews_df["positive_ratio"] = np.where(
        reviews_df["total"] > 0,
        (reviews_df["positive"] / reviews_df["total"]) * 100,
        np.nan,
    )
    reviews_df["review_score_description"] = [
        compute_english_review_desc(tot, ratio)
        for tot, ratio in zip(reviews_df["total"], reviews_df["positive_ratio"])
    ]

    # 4. steamspy_insights.csv
    print("4/5: steamspy_insights.csv işleniyor...")
    steamspy_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "steamspy_insights.csv"), na_values=["\\N", r"\N", ""], escapechar="\\", on_bad_lines="skip", low_memory=False)
    steamspy_df = clean_app_id(steamspy_df)
    steamspy_df = steamspy_df.drop_duplicates(subset=["app_id"])

    steamspy_df["developer"] = steamspy_df["developer"].apply(clean_company_name)
    steamspy_df["publisher"] = steamspy_df["publisher"].apply(clean_company_name)
    steamspy_df["concurrent_users"] = pd.to_numeric(steamspy_df["concurrent_users_yesterday"], errors="coerce").fillna(0)
    steamspy_df["estimated_owners"] = steamspy_df["owners_range"].apply(parse_owners_midpoint)

    steamspy_df = steamspy_df[["app_id", "developer", "publisher", "estimated_owners", "concurrent_users"]]

    # 5. Birleştirme (Merge) ve Eksik Türleri Tags Üzerinden Kurtarma
    print("5/5: Tablolar birleştiriliyor ve eksik türler tamamlanıyor...")
    merged = games_df.merge(reviews_df, on="app_id", how="left")
    merged = merged.merge(steamspy_df, on="app_id", how="left")
    merged = merged.merge(genres_grouped, on="app_id", how="left")
    merged = merged.merge(categories_grouped, on="app_id", how="left")
    merged = merged.merge(tags_grouped, on="app_id", how="left")

    merged = merged[merged["total"].fillna(0) > 0].copy()
    merged = merged.dropna(subset=["name"])

    # Eğer bir oyunun genres alanı boş kaldıysa, tags içindeki standart türlerden otomatik doldur
    empty_genre_mask = merged["genres"].isna() | (merged["genres"].str.strip() == "")
    merged.loc[empty_genre_mask, "genres"] = merged.loc[empty_genre_mask, "tags"].apply(
        lambda v: normalize_multi_value_str(v, GENRE_PATTERNS, keep_clean_ascii_unmatched=False)
    )

    merged.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ Tam İngilizce veri seti hazır: {OUTPUT_FILE}")
    print(f"Toplam Oyun Sayısı: {len(merged)}")


if __name__ == "__main__":
    main()