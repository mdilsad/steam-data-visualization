# Steam Game Data Analysis

An interactive data visualization dashboard built with **Streamlit** analyzing ~88k Steam games with their genres, community tags, player counts, review scores, prices, developers, etc.

## Install and Run

Follow the steps below according to your operating system to set up your local environment.

### 1. Clone the Repository

```bash
git clone git@github.com:mdilsad/steam-data-visualization.git
cd steam-data-visualization
```

### 2. Create and Activate a Virtual Environment

#### Windows

*Using Command Prompt (CMD):*

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the verified cross-platform packages:

```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## About Dataset

`clean_steam.parquet` shows data of 88,005 Steam games from 2024, created by preprocessing the raw data found in `steam-insights` repository by `NewbieIndieGameDev`. Link to the repo is here: <https://github.com/NewbieIndieGameDev/steam-insights>

## Dashboard Tabs and Visualizations

### 1. Mehmet (`src/mehmet.py` & `src/mehmet_ml.py`)

- **Genre Distribution**: Interactive bar chart displaying the distribution and number of Steam games across all 12 genres, enhanced with K-Means Genre Profile Clustering and Random Forest Multi-label Genre Prediction.

- **Genre - Tags Relationship Network**: Interactive bipartite network graph illustrating co-occurrence relationships between game genres and tags, enhanced with K-Means Network Node Clustering and Random Forest Multi-label Tag Prediction.

- **Genre - Review Score Box Plot**: Interactive box plot comparing Steam review score distributions across genres, including medians, quartiles, and outliers, enhanced with K-Means Review Profile Clustering and Random Forest Review Score Prediction.

### 2. Eda (`src/eda.py`)

* **Positive Review Ratio by Game Genre**: Violin plot showing the distribution of positive review percentages across different game genres. It helps compare player satisfaction between genres.
* **Review Volume vs. Positive Rating**: Density contour plot showing the relationship between the total number of reviews and positive review percentages. It helps identify where most games are concentrated.
* **Steam Game Releases by Year**: Line chart showing the number of games released each year. It helps visualize changes in game releases over time.

### 3. Demir (`src/demir.py`)

* **Genre Hierarchy by Total Reviews**: Hierarchical treemap displaying the top 12 genres and their top 15 most-reviewed games.
* **Top Studios by Estimated Gross Revenue**: Stacked bar chart ranking the top 15 developers or publishers by estimated gross revenue and showing their highest earning games.
* **Game Success Profile**: Parallel coordinates graph about the top 50 most-reviewed games showing their release year, price, estimated owners count, total reviews and positive review ratio.

### 4. Dataset Preview

* Table showing the first 500 rows of clean_steam.parquet for quick lookup.

## File Descriptions

* **app.py**: Main program that creates Streamlit dashboard.
* **preprocess.py**: Used for processing the raw csv data and creating a single clean_steam.parquet table.
* **clean_steam.parquet**: Preprocessed dataset.
* **clean_steam.csv.zip**: Preprocessed and compressed dataset in csv format.
* **requirements.txt**: All dependencies.
* **src/**
  * **data_loader.py**: Loads parquet dataset and cache.
  * **demir.py**: Creates Demir's visualization methods: Treemap, Stacked Bar Chart and Parallel Coordinates.
  * **eda.py**: Creates Eda's three interactive visualizations: Violin Plot, Density Contour Plot and Line Chart. Includes filters for game genre, minimum review count and release year.
  * **mehmet.py**: Creates Mehmet's visualization methods: `doldurulacak`
  * **mehmet_ml.py**: `doldurulacak`
