# Netflix Catalogue Explorer — Dash + Plotly

A lightweight analytics dashboard to explore a catalogue of streaming titles (e.g., Netflix). It cleans the dataset, provides rich filtering, and renders a set of decision‑oriented visualizations plus shareable pages for **Executive Summary** and **Top 200 Titles**.

---

## ✨ Highlights

- **Unified filters:** Year, Country, Genre, Type, Theme (Light/Dim/Dark)
- **Robust cleaning:** genre parsing, certification normalization, runtime & IMDb imputations
- **Popularity proxy:** uses `tmdb_popularity` when present, else √(IMDb votes)
- **Two shareable pages:** `/execsum` (executive summary) and `/top200` (top titles table)
- **Colab-ready:** auto-proxies the Dash app when running in Google Colab

---

## 🧩 Visualizations (and why)

| Plot | Where | Why it’s used |
|---|---|---|
| **Pie (Top 10 Genres + Other)** | Explore | Quick sense of genre mix and dominance; “Other” captures long tail. |
| **Word Cloud (Titles)** | Explore | Fast “vibe check” of the slate after applying filters. |
| **Choropleth Map (by titles)** | Explore & Decisions | Shows production footprint breadth and hubs. |
| **Sunburst (Type → Rating → Top-3 Genres)** | Explore | Validates age-brand fit and where content clusters. |
| **Bubble Scatter – “Genre whitespace”** | Decisions | X = #titles, Y = avg IMDb, size = volume, color = popularity. Quadrants ⇒ invest/incubate/fix/experiment. |
| **Stacked Bar (100%) – Ratings mix over time** | Decisions | Trend of maturity (TV-MA/R) vs family (PG/TV-PG). |
| **Lorenz Curve + Gini** | Decisions | Measures hit concentration; informs tentpole vs mid‑tier balance. |
| **Area – Top‑genre capture** | Drilldowns | How many genres cover 50–80% of demand. |
| **Line – Top‑K curve** | Drilldowns | Coverage if we license top K titles; marks 50/70/80/90%. |
| **Bar – Top countries by popularity** | Drilldowns | Prioritize export/localization by demand weight. |
| **Line – Titles per year** | Drilldowns | Catalogue volume trend. |

**KPIs shown:** Total titles, Avg IMDb, Share Mature, Gini; plus Top‑genre share, Top‑10 titles share, Countries covered, Genres represented, Median runtime.

---

## 📊 Stakeholder questions this answers

- *What genres and ratings dominate this filtered slate?*
- *Are we trending more mature or family‑friendly over time?*
- *How hit‑driven is demand (Gini), and how many titles/genres cover 50–80%?*
- *Where is content produced, and which markets carry the most popularity?*
- *What should we **invest**, **incubate**, **fix**, or **de‑prioritize** across genres?*

---

## 🚀 Quickstart

### 1) Clone & install
```bash
git clone https://github.com/yourname/netflix-catalogue-explorer.git
cd netflix-catalogue-explorer
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Minimal `requirements.txt`:**
```
dash==2.17.1
plotly
pandas
numpy
pycountry
wordcloud
pillow
```

### 2) Put your data
Place `titles.csv` in the project root. (Columns used include: `title, type, release_year, genres, production_countries, imdb_score, imdb_votes, tmdb_popularity` – extra columns are ignored.)

### 3) Run
```bash
python app.py
```
Open `http://127.0.0.1:8050`

### Google Colab
Just run the single script cell; it auto‑proxies the port and shows an iframe + “Open in new tab” link.

---

## 🗂️ Project structure
```
.
├── app.py                # Dash app (cleaning, figures, callbacks, routes)
├── titles.csv            # Input dataset (not included)
├── requirements.txt
└── README.md
```

---

## 🔧 Notable implementation details

### Data cleaning
- Robust list parsing for `genres` & `production_countries` (handles JSON-like strings and delimited text).
- `genres_norm` canonicalization; fallback to `"Unknown"` to prevent empty visuals.
- `age_certification` normalized to uppercase values like `PG-13`, `TV-MA`, etc.
- `runtime` imputed by **type median** then **global median**.
- `imdb_score` imputed by **(type × primary_genre) median** then global median.
- Popularity `pop` uses `tmdb_popularity` if present; otherwise `sqrt(imdb_votes)`.

### Routing for shareable pages
- `/top200` and `/execsum` are real Flask routes. The callback writes HTML into
  `server.config["LAST_TOP200_HTML"]` and `server.config["LAST_EXECSUM_HTML"]`; the routes return that HTML.
- Links use a cache‑buster: `/top200?ts=...` and `/execsum?ts=...` so refresh shows latest data.

### Resilience
- All figures are wrapped in safe builders so empty filters won’t crash the app—empty states render a titled blank figure instead.

---

## 🧭 Using the dashboard

1. Choose **Theme** (Light/Dim/Dark).
2. Filter by **Year / Country / Genre / Type** – *all KPI tiles and charts update together*.
3. Use the **Drilldowns** tab for Top‑K and Top‑genre coverage with a slider and a link to **Top 200**.
4. Click **Open executive summary →** to view a printable one‑pager of insights for the current filters.

---

## 🧪 Troubleshooting

- **Executive Summary is blank**: Ensure the app has both the `/execsum` route and that the callback sets `server.config["LAST_EXECSUM_HTML"]` and returns `href="/execsum?ts=..."`. (This repo’s `app.py` already does this.)
- **No data appears**: Verify `titles.csv` is present and columns are named as above; check console for parsing errors.
- **Colab doesn’t show the app**: Make sure the Colab cell has run to completion; the proxy iframe URL appears just under the output cell.

---

## 📄 License
MIT — use freely with attribution.

---

## 🙏 Acknowledgments
- Dataset format inspired by community Netflix title datasets (Kaggle, etc.).
- Built with [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/python/).
