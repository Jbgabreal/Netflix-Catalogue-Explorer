import ast, io, base64, time, urllib.parse
from html import escape as esc

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycountry
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

from dash import Dash, dcc, html
from dash import Input, Output, no_update
from flask import make_response

# =========================================================
#                      DATA LOADING
# =========================================================
df_raw = pd.read_csv("titles.csv")
print("Data loaded successfully")
df_raw.head()

# =========================================================
#                      HELPERS & CLEAN
# =========================================================
def to_list(x):
    if pd.isna(x): return []
    if isinstance(x, list): return x
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    for sep in [",","|",";","/"]:
        if sep in s:
            return [t.strip() for t in s.split(sep) if t.strip()]
    return [s] if s else []

def normalize_genres(lst):
    out = []
    for g in lst:
        if isinstance(g, str):
            g2 = g.strip()
            if g2 and g2.lower() not in {"unknown","nan","none"}:
                out.append(g2.title())
    return out

def a2_to_a3(code):
    if not isinstance(code, str): return None
    code = code.upper().strip()
    if code == "XK": return "XKX"
    try:
        return pycountry.countries.get(alpha_2=code).alpha_3
    except Exception:
        return None

def a2_name(code):
    if not isinstance(code, str): return None
    code = code.upper().strip()
    if code == "XK": return "Kosovo"
    try:
        return pycountry.countries.get(alpha_2=code).name
    except Exception:
        return None

def prepare_for_ratings(df):
    out = df.copy()
    out["release_year"] = pd.to_numeric(out["release_year"], errors="coerce")
    s = out["age_certification"].astype(str)
    s = s.mask(s.isin(["nan","None","NaN"]), None)
    out["age_certification"] = s.fillna("UNKNOWN").str.upper().str.replace(" ", "", regex=False)
    return out

# ---------- clean & enrich (no rows dropped)
df = df_raw.copy()
df["type"] = df["type"].fillna("UNKNOWN").str.upper()
df.loc[~df["type"].isin(["MOVIE","SHOW"]), "type"] = "UNKNOWN"
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")

df["genres_list"] = df["genres"].apply(to_list)
df["production_countries_list"] = df["production_countries"].apply(to_list)
df["genres_norm"] = df["genres_list"].apply(normalize_genres).apply(lambda L: L if L else ["Unknown"])

df["age_certification"] = df["age_certification"].astype(str)
df.loc[df["age_certification"].isin(["nan","None","NaN"]), "age_certification"] = np.nan
df["age_certification"] = df["age_certification"].fillna("UNKNOWN").str.upper().str.replace(" ", "", regex=False)

df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
type_medians = df.groupby("type")["runtime"].transform("median")
df["runtime"] = df["runtime"].fillna(type_medians).fillna(df["runtime"].median())

df["imdb_score"] = pd.to_numeric(df.get("imdb_score"), errors="coerce")
df["primary_genre"] = df["genres_norm"].apply(lambda L: L[0] if L else "Unknown")
group_med = df.groupby(["type","primary_genre"])["imdb_score"].transform("median")
df["imdb_score"] = df["imdb_score"].fillna(group_med).fillna(df["imdb_score"].median())

if "tmdb_popularity" in df.columns:
    df["pop"] = pd.to_numeric(df["tmdb_popularity"], errors="coerce")
else:
    df["pop"] = np.nan
df["imdb_votes"] = pd.to_numeric(df.get("imdb_votes"), errors="coerce")
df.loc[df["pop"].isna(), "pop"] = np.sqrt(df["imdb_votes"].clip(lower=0)).fillna(0)
df["pop"] = df["pop"].fillna(0)

years = sorted(df["release_year"].dropna().astype(int).unique().tolist())
all_countries_alpha2 = (
    pd.Series([c for row in df["production_countries_list"] for c in row if isinstance(c,str) and c.strip()])
    .unique()
)
country_opts = [{"label":"All countries","value":"ALL"}]
for c2 in sorted(all_countries_alpha2):
    nm = a2_name(c2)
    if nm:
        country_opts.append({"label": nm, "value": c2})

all_genres = sorted({g for row in df["genres_norm"] for g in row})
genre_opts = [{"label":"All genres","value":"ALL"}] + [{"label":g,"value":g} for g in all_genres]
type_opts = [{"label":"ALL","value":"ALL"}] + [{"label":t,"value":t} for t in ["MOVIE","SHOW"]]

# =========================================================
#                       THEME TOKENS
# =========================================================
THEMES = {
    "Light": {"bg":"#ffffff","fg":"#111827","panel":"#f8fafc","template":"plotly_white","wc_bg":"#ffffff"},
    "Dim":   {"bg":"#0b1220","fg":"#e5e7eb","panel":"#111827","template":"plotly_dark","wc_bg":"#111827"},
    "Dark":  {"bg":"#000000","fg":"#ffffff","panel":"#0b0f19","template":"plotly_dark","wc_bg":"#0b0f19"},
}
def apply_theme(fig, theme):
    t = THEMES.get(theme or "Light", THEMES["Light"])
    fig.update_layout(template=t["template"], paper_bgcolor=t["panel"], plot_bgcolor=t["panel"],
                      font_color=t["fg"], margin=dict(l=10, r=10, t=45, b=10))
    return fig

# =========================================================
#                    FILTER & UTILS
# =========================================================
MATURE = {"R","NC-17","TV-MA"}

def filtered(dfin, year_v, country_v, genre_v, type_v):
    dff = dfin.copy()
    if year_v != "ALL":
        dff = dff[dff["release_year"].astype("Int64") == int(year_v)]
    if country_v != "ALL":
        dff = dff[dff["production_countries_list"].apply(lambda L: isinstance(L,list) and country_v in L)]
    if genre_v != "ALL":
        dff = dff[dff["genres_norm"].apply(lambda L: isinstance(L,list) and genre_v in L)]
    if type_v != "ALL":
        dff = dff[dff["type"] == type_v]
    return dff

def wc_src_from(dff, theme):
    text = " ".join(dff["title"].dropna().astype(str).tolist())
    if not text.strip():
        buf = io.BytesIO()
        Image.new("RGBA",(10,10),(0,0,0,0)).save(buf, format="PNG"); buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
    sw = set(STOPWORDS); sw.update(["The","A","Of","And","Part","Series","De"])
    wc = WordCloud(width=1200, height=520, background_color=THEMES[theme]["wc_bg"],
                   stopwords=sw, collocations=False)
    img = wc.generate(text).to_image()
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def gini_from(series):
    x = np.sort(np.asarray(series, dtype=float))
    if x.size == 0 or x.sum() == 0: 
        return 0.0, ([0,100],[0,100])
    cum = np.cumsum(x); cum_rel = cum / cum[-1]
    p = np.linspace(0,1,len(x))
    B = np.trapz(cum_rel, p)
    G = 1 - 2*B
    return float(G), (p*100, cum_rel*100)

def explode_nonempty(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.explode(col).dropna(subset=[col])

def compute_topk(pop_series: pd.Series):
    s = pop_series.clip(lower=0).sort_values(ascending=False).values
    tot = float(s.sum()) if s.size else 0.0
    cum = np.cumsum(s) / (tot if tot > 0 else 1.0) if s.size else np.array([])
    return s, tot, cum

def k_at(cum: np.ndarray, p: float) -> int:
    return int(np.searchsorted(cum, p, side="left") + 1) if cum.size else 0

def country_top_rows(dff: pd.DataFrame):
    rows_titles, rows_pop = [], []
    xc = explode_nonempty(dff, "production_countries_list")
    if not xc.empty:
        ct_titles = (xc.groupby("production_countries_list").size()
                      .sort_values(ascending=False).head(5).reset_index(name="n"))
        ct_titles["name"] = ct_titles["production_countries_list"].map(a2_name)
        rows_titles = [f"<li>{esc(r['name'])} — {int(r['n'])} titles</li>" for _, r in ct_titles.iterrows()]
        ct_pop = (xc.groupby("production_countries_list")["pop"].sum()
                    .sort_values(ascending=False).head(5).reset_index())
        ct_pop["name"] = ct_pop["production_countries_list"].map(a2_name)
        rows_pop = [f"<li>{esc(r['name'])} — {r['pop']:.0f} pop</li>" for _, r in ct_pop.iterrows()]
    return rows_titles, rows_pop

def genre_table(dff: pd.DataFrame):
    xg = explode_nonempty(dff, "genres_norm")
    if xg.empty:
        return xg, None
    gtab = (xg.groupby("genres_norm")
              .agg(n=("genres_norm","size"), imdb=("imdb_score","mean"), pop=("pop","sum"))
              .sort_values("pop", ascending=False)
              .reset_index())
    return xg, gtab

# Serve HTML snapshots (/top200, /execsum)
def register_html_config_route(app, path, config_key, default_html, *, endpoint=None):
    if endpoint is None:
        endpoint = ("serve__" + path.strip("/").replace("/", "_")) or "serve__root"
    def _view():
        html_doc = app.config.get(config_key, default_html)
        resp = make_response(html_doc)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        return resp
    app.add_url_rule(path, endpoint=endpoint, view_func=_view, methods=["GET"])
    return _view

# =========================================================
#                    BUILD FIGURES
# =========================================================
def fig_pie_genres(dff, theme):
    joined = dff["genres_norm"].apply(lambda L: "|".join(sorted(set(L))))
    if joined.empty:
        fig = px.pie(values=[1], names=["No data"], hole=0.35, title="Top 10 Genres (with 'Other')")
        return apply_theme(fig, theme)
    dummies = joined.str.get_dummies(sep="|").astype("int64")
    counts = dummies.sum().sort_values(ascending=False)
    if counts.empty:
        fig = px.pie(values=[1], names=["No data"], hole=0.35, title="Top 10 Genres (with 'Other')")
        return apply_theme(fig, theme)
    top = counts.head(10).copy()
    other = counts.iloc[10:].sum()
    if other > 0: top.loc["Other"] = other
    fig = px.pie(values=top.values, names=top.index, hole=0.35, title="Top 10 Genres (with 'Other')")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return apply_theme(fig, theme)

def fig_choropleth(dff, theme):
    tmp = dff.explode("production_countries_list").dropna(subset=["production_countries_list"])
    if tmp.empty:
        fig = go.Figure(); fig.update_layout(title="Titles by production country")
        return apply_theme(fig, theme)
    tmp["iso_a3"] = tmp["production_countries_list"].map(a2_to_a3)
    counts = tmp.dropna(subset=["iso_a3"]).groupby("iso_a3").size().reset_index(name="titles")
    fig = px.choropleth(counts, locations="iso_a3", color="titles",
                        color_continuous_scale="YlOrRd", projection="natural earth",
                        title="Titles by production country")
    return apply_theme(fig, theme)

def fig_sunburst(dff, theme):
    g = dff.copy()
    if g.empty:
        fig = go.Figure(); fig.update_layout(title="Type → Rating → Top-3 Genres (others grouped)")
        return apply_theme(fig, theme)
    g["type"] = g["type"].fillna("Unknown")
    g["age_certification"] = g["age_certification"].fillna("UNKNOWN")
    g = g.explode("genres_norm").rename(columns={"genres_norm":"genre"})
    g["genre"] = g["genre"].fillna("Unknown")
    top8 = g["genre"].value_counts().head(8).index
    g["genre_group"] = g["genre"].where(g["genre"].isin(top8), "Other")
    agg = g.groupby(["type","age_certification","genre_group"], as_index=False).size().rename(columns={"size":"titles"})
    agg["rank_in_parent"] = agg.groupby(["type","age_certification"])["titles"].rank(method="first", ascending=False)
    K=5
    pruned = (agg.assign(genre_pruned=np.where(agg["rank_in_parent"]<=K, agg["genre_group"], "Other"))
                 .groupby(["type","age_certification","genre_pruned"], as_index=False)["titles"].sum())
    fig = px.sunburst(pruned, path=["type","age_certification","genre_pruned"], values="titles",
                      color="type", color_discrete_map={"MOVIE":"#636EFA","SHOW":"#EF553B"},
                      title="Type → Rating → Top-5 Genres (others grouped)")
    fig.update_traces(textinfo="label+percent parent", maxdepth=2)
    return apply_theme(fig, theme)

def fig_genre_whitespace(dff, theme):
    x = dff.explode("genres_norm").dropna(subset=["genres_norm"])
    if x.empty:
        fig = go.Figure(); fig.update_layout(title="Genre whitespace — size=#titles, color=popularity")
        return apply_theme(fig, theme)
    g = (x.groupby("genres_norm")
           .agg(n=("genres_norm","size"), imdb=("imdb_score","mean"), pop=("pop","mean"))
           .reset_index().rename(columns={"genres_norm":"genre"}))
    fig = px.scatter(g, x="n", y="imdb", size="n", color="pop", hover_name="genre",
                     title="Genre whitespace — size=#titles, color=popularity",
                     labels={"n":"# Titles","imdb":"Avg IMDb"})
    if not g.empty:
        fig.add_hline(y=dff["imdb_score"].mean(), line_width=1, line_dash="dot")
        fig.add_vline(x=g["n"].median(), line_width=1, line_dash="dot")
    return apply_theme(fig, theme)

RATING_ORDER = ["TV-Y","TV-Y7","TV-Y7-FV","TV-G","TV-PG","PG","PG-13","G","R","NC-17","TV-14","TV-MA","UNKNOWN"]

def fig_ratings_share(dff, theme):
    tmp = prepare_for_ratings(dff)
    tmp = tmp.loc[tmp["release_year"].notna(), ["release_year", "age_certification"]].copy()
    if tmp.empty:
        fig = go.Figure(); fig.update_layout(title="Ratings mix over time (share)")
        return apply_theme(fig, theme)

    wide = (tmp.groupby(["release_year","age_certification"]).size().unstack(fill_value=0))
    for r in RATING_ORDER:
        if r not in wide.columns:
            wide[r] = 0
    wide = wide[RATING_ORDER]

    row_sums = wide.sum(axis=1).replace(0, 1)
    share = wide.div(row_sums, axis=0).reset_index()
    long_share = share.melt(id_vars="release_year", var_name="age_certification", value_name="share")

    fig = px.bar(long_share, x="release_year", y="share", color="age_certification",
                 category_orders={"age_certification": RATING_ORDER},
                 title="Ratings mix over time (share)", barmode="stack",
                 labels={"release_year":"Year","share":"Share of titles","age_certification":"Rating"})
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return apply_theme(fig, theme)

def fig_lorenz(dff, theme):
    G, (pxs, pys) = gini_from(dff["pop"].clip(lower=0))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pxs, y=pys, mode="lines", name="Lorenz"))
    fig.add_trace(go.Scatter(x=[0,100], y=[0,100], mode="lines", name="Equality", line=dict(dash="dot")))
    fig.update_layout(title=f"Hit concentration (Lorenz curve) — Gini={G:.2f}",
                      xaxis_title="Cumulative % of titles", yaxis_title="Cumulative % of popularity")
    return apply_theme(fig, theme)

def fig_top_genre_capture(dff, theme):
    x = dff.explode("genres_norm").dropna(subset=["genres_norm"])
    if x.empty:
        fig = go.Figure(); fig.update_layout(title="Top-genre capture of popularity")
        return apply_theme(fig, theme)
    g = x.groupby("genres_norm")["pop"].sum().sort_values(ascending=False)
    tot = float(g.sum()) or 1.0
    share = g.rename("share").div(tot).reset_index().rename(columns={"genres_norm":"genre"})
    share["rank"] = np.arange(1, len(share) + 1)
    share["cum_share"] = share["share"].cumsum()

    fig = px.area(share, x="rank", y="cum_share", title="Top-genre capture of popularity",
                  labels={"rank":"Top N genres","cum_share":"Cumulative share"})
    fig.update_traces(
        mode="lines+markers",
        customdata=np.stack([share["genre"].values, share["share"].values], axis=-1),
        hovertemplate="Rank %{x}: %{customdata[0]}<br>Genre share=%{customdata[1]:.1%}<br>Cumulative=%{y:.1%}<extra></extra>",
    )
    for thr in (0.50, 0.80):
        fig.add_hline(y=thr, line_dash="dot")
        hit = share.loc[share["cum_share"].ge(thr), "rank"]
        if not hit.empty:
            n = int(hit.iloc[0])
            fig.add_vline(x=n, line_dash="dot")
            fig.add_annotation(x=n, y=thr, xanchor="left", yanchor="bottom", text=f"N={n} @ {int(thr*100)}%",
                               showarrow=True, arrowhead=0)
    legend_rows = [f"{r}. {g} — {s:.0%}" for r,g,s in zip(share["rank"].head(12), share["genre"].head(12), share["share"].head(12))]
    legend_text = "<b>Top genres</b><br>" + "<br>".join(legend_rows)
    fig.add_annotation(x=1.02, y=0.5, xref="paper", yref="paper", text=legend_text, showarrow=False, align="left",
                       bgcolor="rgba(255,255,255,0.65)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1, font=dict(size=12))
    fig.update_layout(margin=dict(r=180))
    return apply_theme(fig, theme)

def fig_topK_curve(dff, K, theme):
    s = dff["pop"].clip(lower=0).sort_values(ascending=False).values
    if s.size == 0:
        s = np.array([0.0])
    tot = s.sum() if s.sum() > 0 else 1.0
    top = np.cumsum(s) / tot
    x = np.arange(1, len(top) + 1)

    fig = px.line(x=x, y=top, labels={"x":"Top K titles","y":"Cumulative share"},
                  title=f"How quickly hits cover demand (Top-K) — K={K}")
    fig.update_traces(mode="lines+markers",
                      hovertemplate="Top K titles=%{x}<br>Cumulative share=%{y:.1%}<extra></extra>")
    fig.add_vline(x=min(K, len(top)), line_dash="dot")
    for pct in (0.5, 0.7, 0.8, 0.9):
        k_star = int(np.searchsorted(top, pct, side="left") + 1)
        k_star = min(max(k_star, 1), len(top))
        fig.add_hline(y=pct, line_dash="dot", opacity=0.35)
        fig.add_vline(x=k_star, line_dash="dot", opacity=0.35)
        fig.add_annotation(x=k_star, y=pct, text=f"{int(pct*100)}% @ K={k_star}",
                           showarrow=True, arrowhead=2, ax=18, ay=-18,
                           bgcolor="rgba(255,255,255,.65)", bordercolor="rgba(0,0,0,.25)", borderwidth=1)
    if len(dff):
        tot_pop = float(dff["pop"].clip(lower=0).sum()) or 1.0
        tt = (dff.sort_values("pop", ascending=False)[["title","pop"]]
                .assign(share=lambda x: x["pop"]/tot_pop).head(15).reset_index(drop=True))
        legend_rows = [f"{i}. {t} — {s:.0%}" for i,(t,s) in enumerate(zip(tt["title"], tt["share"]), start=1)]
        legend_text = "<b>Top titles</b><br>" + "<br>".join(legend_rows)
        fig.add_annotation(x=1.02, y=0.5, xref="paper", yref="paper", text=legend_text, showarrow=False, align="left",
                           bgcolor="rgba(255,255,255,0.70)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1, font=dict(size=12))
        fig.update_layout(margin=dict(r=260))
    return apply_theme(fig, theme)

def fig_top_country_pop(dff, theme):
    tmp = dff.explode("production_countries_list").dropna(subset=["production_countries_list"])
    if tmp.empty:
        fig = go.Figure(); fig.update_layout(title="Top countries by popularity")
        return apply_theme(fig, theme)
    g = (tmp.groupby("production_countries_list")["pop"].sum()
           .sort_values(ascending=False).head(15).reset_index())
    g["country"] = g["production_countries_list"].map(a2_name)
    fig = px.bar(g, x="country", y="pop", title="Top countries by popularity")
    fig.update_layout(xaxis_tickangle=-30)
    return apply_theme(fig, theme)

def fig_titles_per_year(dff, theme):
    g = dff.dropna(subset=["release_year"]).groupby("release_year").size().reset_index(name="titles")
    if g.empty:
        fig = go.Figure(); fig.update_layout(title="Titles per year")
        return apply_theme(fig, theme)
    fig = px.line(g, x="release_year", y="titles", title="Titles per year")
    return apply_theme(fig, theme)

def fig_wordcloud(dff, theme):
    src = wc_src_from(dff, theme)
    fig = go.Figure()
    fig.add_layout_image(dict(source=src, xref="paper", yref="paper", x=0, y=1, sizex=1, sizey=1,
                              sizing="contain", layer="below"))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(title="Word Cloud (Titles)", margin=dict(l=0, r=0, t=40, b=0))
    return apply_theme(fig, theme)

# =========================================================
#                       DASH APP (MAIN)
# =========================================================
app = Dash(__name__)
app.title = "Netflix Explorer"
server = app.server  # <- WSGI entrypoint for Gunicorn

# --- Flask routes (for HTML snapshots) ---
server.config["LAST_TOP200_HTML"]  = "<!doctype html><title>Top 200</title><p>No data yet.</p>"
server.config["LAST_EXECSUM_HTML"] = "<!doctype html><title>Executive summary</title><p>No data yet.</p>"
register_html_config_route(server, "/top200",  "LAST_TOP200_HTML",
                           "<!doctype html><title>Top 200</title><p>No data yet.</p>",
                           endpoint="serve_top200")
register_html_config_route(server, "/execsum", "LAST_EXECSUM_HTML",
                           "<!doctype html><title>Executive summary</title><p>No data yet.</p>",
                           endpoint="serve_execsum")

DROPDOWN_CSS = """
.Select-menu-outer { background-color: #1e1e1e !important; color: #ffffff !important; }
.Select-menu-outer .VirtualizedSelectOption { color: #ffffff !important; }
.Select-control, .Select-placeholder, .Select--single > .Select-control .Select-value { color: #ffffff !important; }
"""
HOWTO_CSS = """
.howto-card{background:rgba(245,158,11,.10);border:1px solid rgba(245,158,11,.35);
  border-left:8px solid #f59e0b;border-radius:12px;padding:12px 14px;margin:10px 0 6px 0;
  box-shadow:0 2px 8px rgba(0,0,0,.08)}
.howto-card[open]{ box-shadow:0 6px 16px rgba(0,0,0,.12) }
.howto-sticky{ position:sticky; top:8px; z-index:40 }
.howto-summary{ cursor:pointer; font-weight:800; letter-spacing:.2px; display:flex; align-items:center; gap:8px }
.howto-summary::before{ content:"💡" }
.howto-card ul{ margin:8px 0 0 0 } .howto-card li{ line-height:1.35; margin:.15rem 0 }
"""
app.index_string = f"""
<!DOCTYPE html><html><head>
  {{%metas%}}<title>Netflix Explorer</title>{{%favicon%}}{{%css%}}
  <style>{DROPDOWN_CSS}{HOWTO_CSS}</style></head>
  <body style="margin:0">{{%app_entry%}}
  <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body></html>
"""

def kpi_card(id_, label):
    return html.Div([
        html.Div(id=id_, style={"fontSize":"28px","fontWeight":"700","marginBottom":"4px"}),
        html.Div(label, style={"fontSize":"12px","opacity":0.8})
    ], style={"background":"rgba(0,0,0,0.04)","padding":"14px","borderRadius":"10px"})

def _bullet_list(items):
    li_nodes = []
    for it in items:
        if isinstance(it, (list, tuple)):
            text, children = it[0], it[1] if len(it) > 1 else []
            li_nodes.append(html.Li([text, _bullet_list(children)]))
        else:
            li_nodes.append(html.Li(it))
    return html.Ul(li_nodes, style={"marginTop": "8px"})

def make_howto_section(section_id, summary_text, bullets, *, open=False,
                       className="howto-card howto-sticky"):
    return html.Details(id=section_id, open=open, className=className,
                        children=[html.Summary(summary_text, className="howto-summary"), _bullet_list(bullets)])

EXPLORE_BULLETS = [
    "Use the filters (Year, Country, Genre, Type). All KPIs & charts update together.",
    ["KPIs:", ["Total titles", "Avg IMDb", "Share mature (R/NC-17/TV-MA)", "Gini (hit concentration)"]],
    ["Top-left: Top 10 Genres (pie).", ["‘Other’ groups the long tail."]],
    ["Top-right: Word cloud.", ["Quick vibe check; filter first for signal."]],
    ["Bottom-left: Production map.", ["Co-productions count for each country."]],
    ["Bottom-right: Sunburst.", ["Type → Rating → Top genres."]],
]
DECISIONS_BULLETS = [
    ["KPIs:", ["Top genre share", "Top-10 share", "Countries covered", "Genres represented"]],
    ["Top-left: Genre whitespace.", ["Invest / Incubate / Fix / Experiment using medians."]],
    ["Top-right: Ratings mix over time.", []],
    ["Bottom-left: Production map (titles).", []],
    ["Bottom-right: Lorenz & Gini.", []],
]
DRILL_BULLETS = [
    "KPI: Median runtime.",
    ["Top-left: Top-genre capture (area).", []],
    ["Top-right: Top-K curve (slider).", []],
    ["Bottom-left: Top countries by popularity.", []],
    ["Bottom-right: Titles per year.", []],
]

def howto_section_explore():  return make_howto_section("howto1","Quick tour & What to look for (Explore)",EXPLORE_BULLETS)
def howto_section_decisions():return make_howto_section("howto2","Quick tour & What to look for (Decision Insights)",DECISIONS_BULLETS)
def howto_section_drill():    return make_howto_section("howto3","Quick tour & What to look for (Drilldowns)",DRILL_BULLETS)

controls = html.Div(
    style={"display":"grid","gridTemplateColumns":"auto 1fr 1fr 1fr 1fr","gap":"10px","alignItems":"center","padding":"10px 12px"},
    children=[
        html.Div([
            html.Label("Theme", style={"fontWeight": 700}),
            dcc.RadioItems(
                id="theme",
                options=[{"label":"Light","value":"Light"},{"label":"Dim","value":"Dim"},{"label":"Dark","value":"Dark"}],
                value="Light",
                labelStyle={"display":"inline-block","marginRight":"10px"},
            ),
        ]),
        html.Div([html.Label("Year", style={"fontWeight":700}),
                  dcc.Dropdown(id="year", options=[{"label":"All years","value":"ALL"}]+[{"label":int(y),"value":int(y)} for y in years],
                               value="ALL", clearable=False, className="themed-dd", style={"width":"100%"})]),
        html.Div([html.Label("Country", style={"fontWeight":700}),
                  dcc.Dropdown(id="country", options=country_opts, value="ALL", clearable=False,
                               className="themed-dd", style={"width":"100%"})]),
        html.Div([html.Label("Genre", style={"fontWeight":700}),
                  dcc.Dropdown(id="genre", options=genre_opts, value="ALL", clearable=False,
                               className="themed-dd", style={"width":"100%"})]),
        html.Div([html.Label("Type", style={"fontWeight":700}),
                  dcc.Dropdown(id="ctype", options=type_opts, value="ALL", clearable=False,
                               className="themed-dd", style={"width":"100%"})]),
    ],
)

tab_explore = html.Div([
    html.Div(style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"12px"},
             children=[kpi_card("kpi_count","Total titles (current filters)"),
                       kpi_card("kpi_imdb","Avg IMDb score"),
                       kpi_card("kpi_mature","Share mature (R/NC-17/TV-MA)"),
                       kpi_card("kpi_gini","Gini (hit concentration)")]),
    howto_section_explore(),
    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginTop":"8px"},
             children=[dcc.Graph(id="pie_genres"),
                       html.Div([html.H3("Word Cloud (Titles)", style={"marginTop":"0.25rem"}),
                                 html.Img(id="wc", style={"width":"100%","height":"520px","objectFit":"cover","borderRadius":"8px"})])]),
    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginTop":"16px"},
             children=[dcc.Graph(id="map_countries"), dcc.Graph(id="sunburst")])
])

tab_decisions = html.Div([
    html.Div(style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"12px"},
             children=[kpi_card("kpi_topgenre","Top genre share of popularity"),
                       kpi_card("kpi_top10","Top-10 titles share of popularity"),
                       kpi_card("kpi_countries","Countries covered"),
                       kpi_card("kpi_genres","Genres represented")]),
    howto_section_decisions(),
    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginTop":"8px"},
             children=[dcc.Graph(id="genre_whitespace"), dcc.Graph(id="ratings_share")]),
    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginTop":"16px"},
             children=[dcc.Graph(id="dec_map"), dcc.Graph(id="lorenz")])
])

tab_drill = html.Div([
    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr 1fr","gap":"12px"},
             children=[
                kpi_card("kpi_runtime","Median runtime (min)"),
                html.Div([html.Div("Top-K slider (for Top-K curve)", style={"fontSize":"12px","opacity":0.8,"marginBottom":"6px"}),
                          dcc.Slider(10, 200, 10, value=50, marks=None, tooltip={"always_visible":False}, id="k_slider")],
                         style={"background":"rgba(0,0,0,0.04)","padding":"10px","borderRadius":"10px"}),
                html.Div(), html.Div()
             ]),
    howto_section_drill(),
    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginTop":"8px"},
             children=[
                dcc.Graph(id="top_genre_capture"),
                html.Div(style={"display":"grid","gridTemplateRows":"auto auto auto","gap":"8px"},
                         children=[dcc.Graph(id="topK_curve"),
                                   html.Div(id="topk_summary", style={"fontSize":"12px","opacity":0.9,"marginTop":"-6px"}),
                                   html.Div([html.A("Open top 200 titles →", id="topk_open_link", href="#", target="_blank",
                                                    style={"textDecoration":"underline"})],
                                            style={"fontSize":"12px","textAlign":"right","marginTop":"-2px"})])
             ]),
    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginTop":"16px"},
             children=[dcc.Graph(id="top_country_pop"), dcc.Graph(id="titles_per_year")])
])

app.layout = html.Div(
    id="page",
    style={"backgroundColor": THEMES["Light"]["bg"], "color": THEMES["Light"]["fg"],
           "minHeight":"100vh", "padding":"12px"},
    children=[
        html.H1("Netflix Catalogue Explorer",
                style={"textAlign":"center","fontFamily":"Georgia, 'Times New Roman', serif",
                       "fontWeight":800,"letterSpacing":"1px"}),
        html.Div([
            html.A("Open executive summary →", id="execsum_link", href="#", target="_blank",
                   style={"textDecoration":"underline","fontWeight":600, "marginRight":"12px"}),
            html.A("Open mini view →", id="open_mini_link", href="/mini/", target="_blank",
                   style={"textDecoration":"underline","fontWeight":600}),
        ], style={"textAlign":"right","margin":"-6px 2px 8px"}),
        controls,
        dcc.Tabs(id="tabs", value="tab1",
                 children=[dcc.Tab(label="Explore", value="tab1", children=tab_explore),
                           dcc.Tab(label="Decision Insights", value="tab2", children=tab_decisions),
                           dcc.Tab(label="Drilldowns", value="tab3", children=tab_drill)])
    ]
)
app.validation_layout = app.layout

# ===== Safe helpers for callback =====
def _empty_fig(title, theme):
    f = go.Figure(); f.update_layout(title=title)
    return apply_theme(f, theme)

def _safe_fig(builder, theme, title, *args, **kwargs):
    try:
        return builder(*args, **kwargs)
    except Exception:
        return _empty_fig(title, theme)

def _safe_wc(dff, theme):
    try:
        return wc_src_from(dff, theme)
    except Exception:
        buf = io.BytesIO(); Image.new("RGBA",(1,1),(0,0,0,0)).save(buf, format="PNG"); buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def country_options_for_year(df, year):
    base = df
    if year != "ALL":
        base = base[base["release_year"].astype("Int64") == int(year)]
    xc = base.explode("production_countries_list").dropna(subset=["production_countries_list"])
    opts = [{"label": "All countries", "value": "ALL"}]
    if xc.empty:
        return opts
    counts = xc["production_countries_list"].value_counts().sort_values(ascending=False)
    for c2, n in counts.items():
        nm = a2_name(c2) or c2
        opts.append({"label": f"{nm} ({int(n)})", "value": c2})
    return opts

# =========================================================
#                  SINGLE UNIFIED CALLBACK
# =========================================================
@app.callback(
    Output("page","style"),
    Output("country","options"),
    Output("kpi_count","children"), Output("kpi_imdb","children"),
    Output("kpi_mature","children"), Output("kpi_gini","children"),
    Output("pie_genres","figure"), Output("wc","src"),
    Output("map_countries","figure"), Output("sunburst","figure"),
    Output("kpi_topgenre","children"), Output("kpi_top10","children"),
    Output("kpi_countries","children"), Output("kpi_genres","children"),
    Output("genre_whitespace","figure"), Output("ratings_share","figure"),
    Output("dec_map","figure"), Output("lorenz","figure"),
    Output("kpi_runtime","children"),
    Output("top_genre_capture","figure"), Output("topK_curve","figure"),
    Output("topk_summary","children"), Output("topk_open_link","href"),
    Output("top_country_pop","figure"), Output("titles_per_year","figure"),
    Output("execsum_link","href"),
    Output("open_mini_link","href"),
    Input("theme","value"),
    Input("year","value"), Input("country","value"), Input("genre","value"), Input("ctype","value"),
    Input("k_slider","value"),
)
def update_all(theme, year, country, genre, ctype, k_value):
    theme = theme or "Light"
    t = THEMES.get(theme, THEMES["Light"])
    page_style = {"backgroundColor": t["bg"], "color": t["fg"], "minHeight": "100vh", "padding": "16px"}

    country_opts_dyn = country_options_for_year(df, year)
    dff = filtered(df, year, country, genre, ctype)
    is_empty = dff.empty

    kpi_count  = f"{len(dff):,}"
    kpi_imdb   = f"{pd.to_numeric(dff['imdb_score'], errors='coerce').mean():.2f}" if not is_empty else "—"
    share_mat  = dff["age_certification"].fillna("UNKNOWN").isin(MATURE).mean() if not is_empty else 0.0
    kpi_mature = f"{round(share_mat*100)}%"
    gini_val, _ = gini_from(dff["pop"].clip(lower=0))
    kpi_gini   = f"{gini_val:.2f}"

    pie_fig      = _safe_fig(fig_pie_genres, theme, "Top 10 Genres", dff, theme)
    wc_img_src   = _safe_wc(dff, theme)
    map_explore  = _safe_fig(fig_choropleth, theme, "Titles by production country", dff, theme)
    sunburst_fig = _safe_fig(fig_sunburst, theme, "Type → Rating → Top-3 Genres", dff, theme)

    if not is_empty:
        xx = explode_nonempty(dff, "genres_norm")
        if not xx.empty:
            genre_pop   = xx.groupby("genres_norm")["pop"].sum()
            tot_pop     = float(genre_pop.sum()) or 0.0
            top_genre   = float(genre_pop.max()/tot_pop) if tot_pop > 0 else 0.0
            top10_share = float(dff["pop"].nlargest(10).sum() / (dff["pop"].sum() or 1.0))
            n_countries = int(xx["production_countries_list"].explode().dropna().nunique())
            n_genres    = int(xx["genres_norm"].nunique())
        else:
            top_genre = top10_share = 0.0; n_countries = n_genres = 0
    else:
        top_genre = top10_share = 0.0; n_countries = n_genres = 0

    kpi_topgenre  = f"{round(top_genre*100)}%"
    kpi_top10     = f"{round(top10_share*100)}%"
    kpi_countries = str(n_countries)
    kpi_genres    = str(n_genres)

    gw_fig            = _safe_fig(fig_genre_whitespace, theme, "Genre whitespace", dff, theme)
    ratings_share_fig = _safe_fig(fig_ratings_share,   theme, "Ratings mix over time (share)", dff, theme)
    dec_map_fig       = _safe_fig(fig_choropleth,      theme, "Titles by production country", dff, theme)
    lorenz_fig        = _safe_fig(fig_lorenz,          theme, "Hit concentration (Lorenz)", dff, theme)

    med_runtime = pd.to_numeric(dff["runtime"], errors="coerce").median()
    kpi_runtime = f"{int(med_runtime)} min" if pd.notna(med_runtime) else "—"

    tgc_fig = _safe_fig(fig_top_genre_capture, theme, "Top-genre capture", dff, theme)
    topk_fig= _safe_fig(fig_topK_curve,       theme, "Top-K curve", dff, k_value, theme)

    s, tot, cum = compute_topk(dff["pop"])
    k50, k80, k90 = k_at(cum, 0.5), k_at(cum, 0.8), k_at(cum, 0.9)
    k_clamped     = max(1, min(int(k_value or 1), len(cum))) if cum.size else 1
    cur_cov       = float(cum[k_clamped - 1]) if cum.size else 0.0
    topk_summary = html.Div([
        html.Span(f"Current K={k_clamped} → {cur_cov:.0%} coverage.  "),
        html.Span(f"Milestones: 50% @ K={k50} · 80% @ K={k80} · 90% @ K={k90}")
    ])

    # ----- EXEC SUMMARY HTML
    xg, gtab = genre_table(dff)
    top_gen_rows = []
    invest_rows = incubate_rows = fix_rows = experiment_rows = []
    if gtab is not None:
        top_gen = gtab.head(5)
        top_gen_rows = [f"<li><b>{esc(r['genres_norm'])}</b> — {int(r['n'])} titles, IMDb {r['imdb']:.1f}</li>"
                        for _, r in top_gen.iterrows()]
        n_med, imdb_med = gtab["n"].median(), gtab["imdb"].median()
        def _fmt_rows(df, k=4):
            return [f"<li><b>{esc(r['genres_norm'])}</b> — {int(r['n'])} titles, IMDb {r['imdb']:.1f}</li>"
                    for _, r in df.head(k).iterrows()]
        invest   = gtab[(gtab["imdb"]>=imdb_med) & (gtab["n"]>=n_med)].sort_values(["imdb","n","pop"], ascending=[False,False,False])
        incubate = gtab[(gtab["imdb"]>=imdb_med) & (gtab["n"]< n_med)].sort_values(["imdb","pop"], ascending=[False,False])
        fixq     = gtab[(gtab["imdb"]< imdb_med) & (gtab["n"]>=n_med)].sort_values(["n","imdb"], ascending=[False,True])
        exper    = gtab[(gtab["imdb"]< imdb_med) & (gtag["n"]< n_med)] if False else gtab[(gtab["imdb"]< imdb_med) & (gtab["n"]< n_med)]
        experiment_rows = _fmt_rows(exper); invest_rows = _fmt_rows(invest); incubate_rows = _fmt_rows(incubate); fix_rows = _fmt_rows(fixq)

    top_ct_rows, top_cp_rows = country_top_rows(dff)
    trend_text = "Not enough years to compute trend."
    yr = pd.to_numeric(dff["release_year"], errors="coerce").dropna().astype(int)
    if not yr.empty:
        y_max = int(yr.max())
        y_all = sorted(yr.unique())
        W = 5 if len(y_all) >= 10 else (3 if len(y_all) >= 6 else None)
        if W:
            recent = list(range(y_max - W + 1, y_max + 1))
            prev   = list(range(y_max - 2*W + 1, y_max - W + 1))
            def mature_share_in(years):
                sl = dff[dff["release_year"].isin(years)]
                return sl["age_certification"].isin(MATURE).mean() if len(sl) else np.nan
            r_sh, p_sh = mature_share_in(recent), mature_share_in(prev)
            if pd.notna(r_sh) and pd.notna(p_sh):
                delta = (r_sh - p_sh) * 100
                trend_text = (f"Mature share changed from <b>{p_sh*100:.0f}%</b> "
                              f"(Y{min(prev)}–Y{max(prev)}) to <b>{r_sh*100:.0f}%</b> "
                              f"(Y{min(recent)}–Y{max(recent)}) "
                              f"(<b>{delta:+.0f} pp</b>).")

    exec_html = f"""<!doctype html><html><head><meta charset='utf-8'>
    <title>Executive summary — current filters</title>
    <style>
      body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:22px;line-height:1.45}}
      h1{{margin:0 0 10px 0}} h2{{margin:18px 0 8px}} h3{{margin:14px 0 6px}}
      .kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:10px 0 6px}}
      .kpi{{background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;padding:10px}}
      .kpi b{{font-size:20px;display:block}}
      .pill{{display:inline-block;padding:2px 8px;border-radius:999px;background:#eef2ff;border:1px solid #c7d2fe;margin-left:6px;font-size:12px}}
      ul{{margin:6px 0 12px 18px}} .note{{opacity:.75}}
    </style></head><body>
    <h1>Executive summary <span class="pill">current filters</span></h1>
    <div class="kpis">
      <div class="kpi"><b>{len(dff):,}</b><div>Total titles</div></div>
      <div class="kpi"><b>{pd.to_numeric(dff['imdb_score'], errors='coerce').mean():.2f}</b><div>Avg IMDb</div></div>
      <div class="kpi"><b>{(dff['age_certification'].isin(MATURE).mean()*100 if len(dff) else 0):.0f}%</b><div>Share mature</div></div>
      <div class="kpi"><b>{gini_val:.2f}</b><div>Gini (hit concentration)</div></div>
    </div>
    <h2>What the data says</h2>
    <ul>
      <li>Top genre captures <b>{kpi_topgenre}</b> of total popularity; top-10 titles account for <b>{kpi_top10}</b>.</li>
      <li>To cover 50% of demand you need ~<b>{k50}</b> titles; 80% needs ~<b>{k80}</b>.</li>
      <li>{trend_text}</li>
      <li>Catalogue breadth: <b>{kpi_countries}</b> countries, <b>{kpi_genres}</b> genres.</li>
    </ul>
    <h2>Genre mix</h2><h3>Top genres (by popularity)</h3>
    <ul>{''.join(top_gen_rows) or '<li class="note">No genre data.</li>'}</ul>
    <h3>Actionable buckets</h3>
    <h4>Invest (quality &amp; scale)</h4><ul>{''.join(invest_rows) or '<li class="note">No candidates.</li>'}</ul>
    <h4>Incubate (quality, low scale)</h4><ul>{''.join(incubate_rows) or '<li class="note">No candidates.</li>'}</ul>
    <h4>Fix quality (scale, weak quality)</h4><ul>{''.join(fix_rows) or '<li class="note">No candidates.</li>'}</ul>
    <h4>Experiment / De-prioritize (low both)</h4><ul>{''.join(experiment_rows) or '<li class="note">No candidates.</li>'}</ul>
    <h2>Geography</h2><h3>Top producers (by titles)</h3>
    <ul>{''.join(top_ct_rows) or '<li class="note">No country data.</li>'}</ul>
    <h3>Top markets (by popularity)</h3>
    <ul>{''.join(top_cp_rows) or '<li class="note">No country data.</li>'}</ul>
    <p class="note">Generated from the active filters in your dashboard.</p></body></html>"""
    server.config["LAST_EXECSUM_HTML"] = exec_html
    execsum_href = f"/execsum?ts={int(time.time())}"

    tot_pop_all = float(dff["pop"].clip(lower=0).sum()) or 1.0
    top200 = dff.sort_values("pop", ascending=False).head(200).copy()
    if not top200.empty:
        top200["share"] = top200["pop"] / tot_pop_all
        top200["primary_genre"] = top200["genres_norm"].apply(lambda L: L[0] if isinstance(L, list) and L else "—"
        )
        top200["year"] = top200["release_year"].apply(lambda y: int(y) if pd.notna(y) else "—")
        rows_html = []
        for i, r in enumerate(top200.to_dict(orient="records"), start=1):
            imdb = "" if pd.isna(r.get("imdb_score")) else f"{float(r['imdb_score']):.1f}"
            rows_html.append("<tr>"
                             f"<td>{i}</td><td>{esc(r['title'])}</td><td>{esc(r['type'])}</td>"
                             f"<td>{r['year']}</td><td>{esc(r['primary_genre'])}</td>"
                             f"<td>{imdb}</td><td>{r['share']:.1%}</td></tr>")
        rows = "\n".join(rows_html)
        html_doc = f"""<!doctype html><html><head><meta charset='utf-8'>
        <title>Top 200 titles — current filters</title>
        <style>body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:18px}}
        h2{{margin:0 0 12px 0}}table{{border-collapse:collapse;width:100%}}
        th,td{{padding:6px 8px;border-bottom:1px solid #e5e7eb;font-size:13px;text-align:left}}
        th{{background:#f8fafc;position:sticky;top:0}} small{{opacity:.7}}</style></head><body>
        <h2>Top 200 titles <small>(by popularity; current filters)</small></h2>
        <table><thead><tr><th>#</th><th>Title</th><th>Type</th><th>Year</th>
        <th>Primary genre</th><th>IMDb</th><th>Popularity share</th></tr></thead>
        <tbody>{rows}</tbody></table></body></html>"""
        server.config["LAST_TOP200_HTML"] = html_doc
        topk_link_href = f"/top200?ts={int(time.time())}"
    else:
        server.config["LAST_TOP200_HTML"] = "<!doctype html><title>Top 200</title><p>No titles in current filter.</p>"
        topk_link_href = "/top200"

    # Build a link to /mini/ with current selections
    q = urllib.parse.urlencode({
        "chart": "Top-K curve",
        "year": year, "country": country, "genre": genre, "type": ctype,
        "theme": theme, "k": k_value
    })
    mini_href = f"/mini/?{q}"

    tcp_fig = _safe_fig(fig_top_country_pop, theme, "Top countries by popularity", dff, theme)
    tpy_fig = _safe_fig(fig_titles_per_year, theme, "Titles per year", dff, theme)

    return (page_style, country_opts_dyn, kpi_count, kpi_imdb, kpi_mature, kpi_gini,
            pie_fig, wc_img_src, map_explore, sunburst_fig, kpi_topgenre, kpi_top10,
            kpi_countries, kpi_genres, gw_fig, ratings_share_fig, dec_map_fig, lorenz_fig,
            kpi_runtime, tgc_fig, topk_fig, topk_summary, topk_link_href, tcp_fig, tpy_fig,
            execsum_href, mini_href)

# Keep country value valid when options change
@app.callback(
    Output("country", "value"),
    Input("country", "value"),
    Input("country", "options"),
)
def ensure_valid_country(selected, options):
    values = {opt["value"] for opt in (options or [])}
    if selected in values:
        return no_update
    return "ALL"

# =========================================================
#                    MINI APP (zoom page)
# =========================================================
MINI_CHARTS = {
    "Genres pie":            lambda dff, theme, K=None: fig_pie_genres(dff, theme),
    "Word cloud":            lambda dff, theme, K=None: fig_wordcloud(dff, theme),
    "Choropleth":            lambda dff, theme, K=None: fig_choropleth(dff, theme),
    "Sunburst":              lambda dff, theme, K=None: fig_sunburst(dff, theme),
    "Genre whitespace":      lambda dff, theme, K=None: fig_genre_whitespace(dff, theme),
    "Ratings mix":           lambda dff, theme, K=None: fig_ratings_share(dff, theme),
    "Lorenz (Gini)":         lambda dff, theme, K=None: fig_lorenz(dff, theme),
    "Top-genre capture":     lambda dff, theme, K=None: fig_top_genre_capture(dff, theme),
    "Top-K curve":           lambda dff, theme, K:     fig_topK_curve(dff, K or 50, theme),
    "Top country pop":       lambda dff, theme, K=None: fig_top_country_pop(dff, theme),
    "Titles per year":       lambda dff, theme, K=None: fig_titles_per_year(dff, theme),
}
mini = Dash(__name__ + "_mini", server=server, url_base_pathname="/mini/")
mini.title = "Single Chart Preview"

mini.layout = html.Div(
    style={"backgroundColor": THEMES["Light"]["bg"], "color": THEMES["Light"]["fg"], "padding": "12px"},
    children=[
        dcc.Location(id="mini_url"),
        html.H3("Single-chart preview", style={"margin":"6px 0 10px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"1.2fr 1fr 1fr 1fr 1fr 0.8fr","gap":"10px"}, children=[
            dcc.Dropdown(id="mini_chart", options=[{"label":k,"value":k} for k in MINI_CHARTS.keys()],
                         value="Genres pie", clearable=False),
            dcc.Dropdown(id="mini_year",    options=[{"label":"All years","value":"ALL"}]+[{"label":int(y),"value":int(y)} for y in years],
                         value="ALL", clearable=False),
            dcc.Dropdown(id="mini_country", options=country_opts, value="ALL", clearable=False),
            dcc.Dropdown(id="mini_genre",   options=genre_opts,   value="ALL", clearable=False),
            dcc.Dropdown(id="mini_type",    options=type_opts,    value="ALL", clearable=False),
            dcc.Dropdown(id="mini_theme",   options=[{"label":k,"value":k} for k in THEMES.keys()],
                         value="Light", clearable=False),
        ]),
        html.Div(style={"margin":"8px 0 4px"}, children=[
            html.Label("K (Top-K curve only)"),
            dcc.Slider(10, 200, 10, value=50, id="mini_k", tooltip={"always_visible":False}),
        ]),
        dcc.Graph(id="mini_figure", style={"height":"78vh"}, config={"scrollZoom": True}),
    ]
)

# Keep country options in mini synced to year
@mini.callback(Output("mini_country","options"), Input("mini_year","value"))
def _mini_country_opts(y):
    return country_options_for_year(df, y)

@mini.callback(Output("mini_country","value"), Input("mini_country","value"), Input("mini_country","options"))
def _mini_country_valid(val, opts):
    values = {o["value"] for o in (opts or [])}
    return val if val in values else "ALL"

# Parse query string to prefill mini controls
@mini.callback(
    Output("mini_chart","value"), Output("mini_year","value"), Output("mini_country","value"),
    Output("mini_genre","value"), Output("mini_type","value"), Output("mini_theme","value"),
    Output("mini_k","value"),
    Input("mini_url","search")
)
def _mini_prefill(search):
    q = urllib.parse.parse_qs((search or "").lstrip("?"))
    def g(key, default):
        v = q.get(key, [default])[0]
        return v
    chart = g("chart", "Genres pie")
    year  = g("year", "ALL")
    try: year = int(year) if year != "ALL" else "ALL"
    except Exception: year = "ALL"
    country = g("country","ALL")
    genre   = g("genre","ALL")
    typ     = g("type","ALL")
    theme   = g("theme","Light") if g("theme","Light") in THEMES else "Light"
    try: k  = int(g("k", 50))
    except Exception: k = 50
    return chart, year, country, genre, typ, theme, k

# Render selected figure in mini
@mini.callback(
    Output("mini_figure","figure"),
    Input("mini_chart","value"),
    Input("mini_year","value"), Input("mini_country","value"),
    Input("mini_genre","value"), Input("mini_type","value"),
    Input("mini_theme","value"), Input("mini_k","value")
)
def _mini_render(chart_key, year_v, country_v, genre_v, type_v, theme_v, k):
    dff = filtered(df, year_v, country_v, genre_v, type_v)
    builder = MINI_CHARTS.get(chart_key)
    try:
        return builder(dff, theme_v, k)
    except TypeError:
        return builder(dff, theme_v)

# =========================================================
#                    LOCAL DEV ONLY
# =========================================================
# if __name__ == "__main__":
#     # Local debug run (Render/Gunicorn will import app:server)
#     app.run(host="0.0.0.0", port=8050, debug=False)

