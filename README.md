# Netflix Catalogue Explorer

Interactive Plotly Dash app to explore a catalogue of titles (movies & shows): filter by **Year / Country / Genre / Type**, track KPIs, and visualize insights (genres, geography, ratings mix, hit concentration, Top‑K coverage, etc.). Built to run smoothly in **Google Colab** or locally with plain Python.

---

## ✨ Features

- **Unified filters** (Year, Country, Genre, Type) — all KPIs & charts update together.
- **KPIs**: total titles, avg IMDb, share of mature ratings (R/NC‑17/TV‑MA), Gini (hit concentration).
- **Explore tab**
  - **Top 10 Genres (donut)** with *Other* grouping.
  - **Word cloud** of titles.
  - **Production choropleth** by country (alpha‑2 → alpha‑3 conversion).
  - **Sunburst** (Type → Rating → Top‑5 Genres).
- **Decision Insights tab**
  - **Genre whitespace** (bubble: size=#titles, color=popularity) with median lines.
  - **Ratings mix over time** (100% stacked share).
  - **Lorenz curve & Gini** (hit concentration).
- **Drilldowns tab**
  - **Top‑genre capture** (how many genres cover 50% / 80% of demand).
  - **Top‑K curve** (coverage vs. number of titles) + **Top‑200 page** link.
  - **Top countries by popularity**, **Titles per year** trend.
- **Executive summary page** (`/execsum`) — printable one‑pager.
- **Colab‑friendly**: auto‑proxies the Dash server in an iframe.

> All figures are themed (Light/Dim/Dark) via a small token map and `apply_theme` helper.

---

## 📦 Data

Place a CSV called **`titles.csv`** in the working directory with at least these columns (extra columns are fine):

```
id, title, type, description, release_year, age_certification, runtime,
genres, production_countries, imdb_id, imdb_score, imdb_votes,
tmdb_popularity, tmdb_score
```

**Parsing helpers**:  
- `to_list` converts mixed representations to lists (handles real lists, JSON‑style strings, comma/pipe/semicolon slugs).  
- `normalize_genres` standardizes capitalization and removes unknowns.  
- Country helpers (`a2_to_a3`, `a2_name`) map ISO‑2 to ISO‑3 and human names.  
- `prepare_for_ratings` coerces years and normalizes rating labels.

Derived columns created by the app:
- `genres_list`, `production_countries_list`, `genres_norm`, `pop` (fallback from √imdb_votes when tmdb_popularity is missing).

---

## 🧰 How to run (Colab or local)

1. **Install once (Colab)**
   ```python
   !pip -q install dash==2.17.1 plotly pycountry wordcloud pillow
   ```

2. **Run the app cell**  
   The provided script creates the Dash server, adds `/top200` and `/execsum` routes, and embeds the app (Colab) or serves on `http://0.0.0.0:8050` (local).

3. **Open helper pages**  
   - Top‑200 table: the “Open top 200 titles →” link on the Drilldowns tab.
   - Executive summary: “Open executive summary →” at the top of the app.

> If running locally, just execute the file with `python app.py` and visit `http://localhost:8050`.

---

## 🧪 Mini single‑chart app (progress demos)

Sometimes you just want to show **one** chart (e.g., the donut genres) while you build. Use this minimal harness:

```python
# 1) Build a filtered frame (or use df directly)
dff = filtered(df, year_v="ALL", country_v="ALL", genre_v="ALL", type_v="ALL")

# 2) Create the figure using any of your builders, e.g.:
fig = fig_pie_genres(dff, theme="Light")

# 3) Serve one chart
from dash import Dash, dcc, html
app = Dash(__name__)
app.layout = html.Div([dcc.Graph(id="pie_genres", figure=fig)], style={"padding":"12px"})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
```

### Optional: single‑chart switcher
Want to switch **live** between charts without changing code? Add a dropdown to select a chart key and map it to a function; reuse the same `filtered(...)` inputs so your single‑chart app still respects filters.

---

## 🎛️ Filters

- **Year**: all or a specific year.  
- **Country**: ISO‑2 code behind the scenes; dropdown shows names.  
- **Genre**: normalized, title‑cased strings.  
- **Type**: MOVIE / SHOW / ALL.

Filtering is **AND‑logic**: each selected control further narrows the dataset. If any selection yields no rows, figures gracefully show “No data”.

---

## 🔍 Notable implementation details

- Robust list parsing (`to_list`) so you can feed raw CSVs containing Python‑like lists (`"['Drama','Comedy']"`), JSON‑like arrays, or delimited strings like `"Drama|Comedy"`.
- Defensive figure helpers (`_safe_fig`, `_empty_fig`) so a bad input never crashes the page.
- Reusable route registrar (`register_html_config_route`) to avoid duplicating Flask endpoints.
- Reusable utilities: `explode_nonempty`, `compute_topk`, `k_at`, `genre_table`, `country_top_rows`.
- Ratings normalization via `prepare_for_ratings`, and stable `RATING_ORDER` so stacked bars never flip.

---

## 🧭 Navigation

- **Explore** → overview KPIs and quick context (genres, word cloud, map, sunburst).  
- **Decision Insights** → whitespace, ratings trend, geography, concentration.  
- **Drilldowns** → Top‑genre capture, Top‑K, country bars, titles‑per‑year.  
- **Links** → *Open executive summary* (printable) and *Top‑200* (table).

---

## 🧑‍🍳 Troubleshooting

- **Executive Summary is blank** → ensure your app set `server.config["LAST_EXECSUM_HTML"]` *and* the `/execsum` route exists (this repo already wires both).  
- **No data appears** → verify `titles.csv` exists and column names match.  
- **Colab shows blank iframe** → make sure the cell ran to completion; the proxy URL is printed by the cell.

---

## 📁 Suggested repository structure

```
.
├── app.py                # the Dash app (this file)
├── titles.csv            # your data (not included in repo)
├── README.md             # this document
└── requirements.txt      # optional pin: dash==2.17.1 etc.
```

> Tip: GitHub’s normal file limit is ~100 MB; keep large CSVs out of the repo or use LFS.

---

## 📜 License

MIT — use freely with attribution.
