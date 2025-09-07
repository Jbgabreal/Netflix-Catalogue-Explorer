# Netflix Catalogue Explorer

Interactive **Plotly Dash** app to explore a catalogue of titles (movies & shows). Filter by **Year / Country / Genre / Type**, track **KPIs**, and visualize insights (genres, geography, ratings mix, hit concentration, Top‑K coverage, etc.). Runs locally, in **Google Colab**, and can be deployed on **Render**.

---

## ✨ Features

- **Unified filters** — Year, Country, Genre, Type (all charts & KPIs react together).
- **KPIs**: total titles, average IMDb score, share of mature ratings (R/NC‑17/TV‑MA), and **Gini** (hit concentration).
- **Explore tab**
  - **Top 10 Genres (donut)** with *Other* grouping.
  - **Word cloud** of titles.
  - **Production choropleth** by country (ISO‑2 → human names, ISO‑3 for map).
  - **Sunburst** (Type → Rating → Top‑5 Genres, others grouped).
- **Decision Insights tab**
  - **Genre whitespace** (bubble): size=#titles, color=popularity; dotted median lines.
  - **Ratings mix over time** (100% stacked share).
  - **Lorenz curve & Gini** (hit concentration).
- **Drilldowns tab**
  - **Top‑genre capture** (how many genres cover 50% / 80% of demand).
  - **Top‑K curve** (coverage vs # titles) + **Top‑200 page** link.
  - **Top countries by popularity** (bar), **Titles per year** (trend).
- **Executive summary page** (`/execsum`) — printable one‑pager reflecting current filters.
- **Theming** — Light / Dim / Dark with a small token map.
- **Colab‑friendly** — auto‑proxies the Dash server in an iframe.

---

## 📦 Data

Put a CSV named **`titles.csv`** in the project root with (at least) these columns (extra columns are fine):

```
id, title, type, description, release_year, age_certification, runtime,
genres, production_countries, imdb_id, imdb_score, imdb_votes,
tmdb_popularity, tmdb_score
```

**Parsing & enrichment (handled by the app):**

- `to_list` converts mixed representations to lists (real lists, JSON‑like strings, or delimited `"Drama|Comedy"`/`"Drama, Comedy"` values).
- `normalize_genres` title‑cases and removes unknowns.
- Country helpers (`a2_name`, `a2_to_a3`) map codes to names and ISO‑3 for maps.
- `prepare_for_ratings` coerces year and normalizes rating labels.
- A derived **`pop`** column is created from `tmdb_popularity` (or falls back to `sqrt(imdb_votes)` when missing).

> Tip: keep big CSVs out of Git if they exceed GitHub’s size limits. You can mount storage or fetch from object storage if needed.

---

## 🚀 Quick start (local)

1. **Create a virtual environment** (recommended) and install deps:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. **Ensure your data file exists**: `titles.csv` in the same folder as `app.py`.

3. **Run the app (dev server)**:

   ```bash
   python app.py
   ```

   Then open http://localhost:8050

4. **Production-style run (optional)** — with Gunicorn:

   ```bash
   gunicorn --bind 0.0.0.0:8050 app:server
   ```

   The `app:server` entry points Gunicorn to the Flask server exposed by Dash inside `app.py`.

---

## 💻 Run in Google Colab

1. Upload `app.py` and `titles.csv` to your Colab workspace or mount Google Drive.  
2. Install once:

   ```python
   !pip -q install dash==2.17.1 plotly pandas numpy pycountry wordcloud pillow gunicorn
   ```

3. Run your `app.py` cell. The notebook will display the app in an iframe and also give a link that opens the proxied URL in a new tab.

---

## 🌐 Deploy to Render (free tier friendly)

1. **Push to GitHub**: repo with `app.py`, `requirements.txt`, and **(optionally)** `Procfile` (useful for Heroku/Railway; Render uses a Start Command instead).
2. **Create Web Service** (Render dashboard → *New* → *Web Service* → connect your repo).
3. **Environment**: leave defaults (Render automatically provides a `$PORT` env var).
4. **Build Command**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Start Command**:

   ```bash
   gunicorn --bind 0.0.0.0:$PORT app:server
   ```

   - `app` = your filename `app.py` (no extension)  
   - `server` = the Flask server object exposed by Dash (`server = app.server` in code)

6. **Deploy**. When it’s live, open the public URL Render gives you.

**Notes**

- The filesystem is ephemeral on deploys; keeping `titles.csv` in the repo is fine for read‑only access. For bigger/updated datasets, consider remote storage or a database.
- You do **not** need a `Procfile` on Render (the Start Command replaces it), but including one doesn’t hurt for cross‑PaaS portability.
- If you use health checks, `/` is a safe path.

---

## 🧭 App navigation

- **Explore** → overview KPIs and quick context (genres, word cloud, production map, sunburst).
- **Decision Insights** → whitespace, ratings trend, geography, hit concentration.
- **Drilldowns** → top‑genre capture, Top‑K curve (with slider), country bars, titles‑per‑year.
- **Links** → *Open executive summary* and *Top‑200* pages, reflecting the active filters.

---

## 🎛️ Filters (AND‑logic)

- **Year**: all or a specific year.  
- **Country**: dropdown labels are country names; the filter uses ISO‑2 codes internally.  
- **Genre**: normalized, title‑cased strings.  
- **Type**: MOVIE / SHOW / ALL.

All charts and KPIs react to the current filter set. When a selection yields no rows, figures gracefully show “No data”.

---

## 🧪 Mini “single‑chart” preview (optional)

For progress demos or to zoom into a single figure, the project includes a minimal **single‑chart switcher** that keeps the same filters and lets you pick any chart from a dropdown. It’s a small standalone Dash app using the same data/figure builders so behavior is identical.

---

## 🧰 Requirements

```
dash==2.17.1
plotly>=5.22
pandas
numpy
pycountry
wordcloud
pillow
gunicorn
```

Save this as **`requirements.txt`** and Render/Colab/local installs will work the same way.

---

## 📁 Suggested repository structure

```
.
├── app.py
├── titles.csv
├── requirements.txt
└── README.md
```

> Optional for other PaaS: a `Procfile` containing  
> `web: gunicorn --bind 0.0.0.0:$PORT app:server`

---

## 🧑‍⚕️ Troubleshooting

- **Blank charts or no data** → verify `titles.csv` exists and column names match those listed above.
- **“Failed to find attribute ‘server’ in ‘app’” when using Gunicorn** → ensure your file is `app.py` and that it exposes `server = app.server`. Start with `app:server`.
- **Executive summary shows old content** → interact with the dashboard first; the summary is generated from the *current* filters.
- **Country list doesn’t update after changing Year** → this app dynamically recomputes Country options per Year and keeps or resets the selection to a valid value.

---

## 📝 License

MIT — use freely with attribution.
