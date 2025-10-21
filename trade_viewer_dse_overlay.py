# trade_viewer_dse_overlay.py
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash.exceptions
import plotly.graph_objects as go
from collections import OrderedDict
import time as _time

# ---------------- CONFIG ----------------
JSON_FOLDER = "json_input_files"
BD_TZ = pytz.timezone("Asia/Dhaka")
UTC = pytz.UTC
INTERVALS = ["15min", "30min", "1H", "2H", "1D", "1W"]
TRADING_START = datetime.strptime("10:00:00", "%H:%M:%S").time()
TRADING_END = datetime.strptime("14:29:59", "%H:%M:%S").time()
BD_WEEKEND = {4, 5}  # Fri/Sat

# global holidays placeholder (populated during load)
HOLIDAYS = pd.to_datetime([])

# ---------------- Simple in-memory LRU cache ----------------
class SimpleLRUCache:
    def __init__(self, max_items=30):
        self.max_items = max_items
        self._od = OrderedDict()
    def get(self, key):
        if key in self._od:
            self._od.move_to_end(key)
            return self._od[key]
        return None
    def set(self, key, value):
        # store copy where possible
        self._od[key] = value.copy() if hasattr(value, "copy") else value
        self._od.move_to_end(key)
        while len(self._od) > self.max_items:
            evicted_key, _ = self._od.popitem(last=False)
            print(f"[Cache] Evicted: {evicted_key}")
    def clear(self):
        self._od.clear()
        print("[Cache] Cleared")

_resample_cache = SimpleLRUCache(max_items=30)

# ---------------- HOLIDAY DETECTION (from filenames) ----------------
def detect_holidays_from_files():
    """Detect missing dates between min and max date parts found in JSON filenames.
    Very cheap: reads filenames only (no JSON parsing) and returns a DatetimeIndex of missing days.
    """
    if not os.path.isdir(JSON_FOLDER):
        print("⚠️ JSON folder missing for holiday detection.")
        return pd.to_datetime([])

    files = sorted(f for f in os.listdir(JSON_FOLDER) if f.endswith(".json"))
    if not files:
        print("⚠️ No JSON files found for holiday detection.")
        return pd.to_datetime([])

    date_parts = sorted({f.split("_")[0] for f in files})
    dates = pd.to_datetime(date_parts, errors="coerce").dropna().normalize()
    if dates.empty:
        print("⚠️ No valid date parts detected in filenames.")
        return pd.to_datetime([])

    full_range = pd.date_range(dates.min(), dates.max(), freq="D")
    missing = full_range.difference(dates)
    print(f"📅 Detected {len(missing)} holidays / missing trading days between {dates.min().date()} and {dates.max().date()}.")
    if len(missing) > 0:
        print("   Sample holidays:", [d.strftime("%Y-%m-%d") for d in missing[:8]])
    return missing

# ---------------- LOAD JSON FOLDER (vectorized, assumes HHMMSS time_str) ----------------
def load_json_folder():
    """Parse all JSONs → single DataFrame with tz-aware DateTime (UTC→BD).
    Optimized for time_str always in HHMMSS format (e.g. 040918).
    Also fills the global HOLIDAYS variable using file names (fast).
    """
    global HOLIDAYS
    print(f"🔍 Scanning JSON folder: {JSON_FOLDER}")
    if not os.path.isdir(JSON_FOLDER):
        print("❌ JSON_FOLDER missing:", JSON_FOLDER)
        return pd.DataFrame(columns=["Symbol", "DateTime", "Close", "Volume"])

    files = sorted(f for f in os.listdir(JSON_FOLDER) if f.endswith(".json"))
    if not files:
        print("⚠️ No JSON files in", JSON_FOLDER)
        return pd.DataFrame(columns=["Symbol", "DateTime", "Close", "Volume"])

    # detect holidays first (cheap)
    HOLIDAYS = detect_holidays_from_files()

    all_records = []
    for fname in files:
        path = os.path.join(JSON_FOLDER, fname)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"⚠️ Failed to read {fname}: {e}")
            continue

        date_part = fname.split("_")[0]
        for sym in data.get("DAT", {}).get("SYM", []):
            symbol = sym.get("S", "").split("`")[0]
            ts_rows = sym.get("TS", [])
            if not ts_rows:
                continue
            for row in ts_rows:
                parts = row.split("|")
                if len(parts) < 4:
                    continue
                time_str = parts[0].strip()  # assumed HHMMSS
                # numeric parsing with small guard
                try:
                    ltp = float(parts[2])
                except Exception:
                    ltp = np.nan
                try:
                    vol = float(parts[3])
                except Exception:
                    vol = 0.0
                all_records.append((symbol, date_part, time_str, ltp, vol))

    if not all_records:
        print("❌ No rows parsed from JSON files.")
        return pd.DataFrame(columns=["Symbol", "DateTime", "Close", "Volume"])

    # vectorized dataframe build
    df = pd.DataFrame(all_records, columns=["Symbol", "DatePart", "TimeStr", "Close", "Volume"])

    # build 'YYYY-MM-DD HH:MM:SS' vectorized (fast)
    # transform HHMMSS -> HH:MM:SS
    df["FullDT"] = (
        df["DatePart"]
        + " "
        + df["TimeStr"].str.replace(r"(\d{2})(\d{2})(\d{2})", r"\1:\2:\3", regex=True)
    )

    # parse and localize to UTC (we assume the timestrings are UTC) then convert to BD_TZ
    # using utc=True for vectorized speed
    dt_utc = pd.to_datetime(df["FullDT"], errors="coerce", utc=True)
    df["DateTime"] = dt_utc.dt.tz_convert(BD_TZ)

    df = df.dropna(subset=["DateTime"])
    df = (
        df[["Symbol", "DateTime", "Close", "Volume"]]
        .sort_values(["Symbol", "DateTime"])
        .reset_index(drop=True)
        .set_index("DateTime")
    )

    print(f"✅ Loaded rows: {len(df):,} | symbols: {df['Symbol'].nunique()}")
    print("Sample symbols:", df["Symbol"].unique()[:8].tolist())
    return df

# ---------------- RESAMPLE ----------------
def resample_data(df, rule):
    """Given a tick DataFrame for one symbol (or pre-filtered df_input),
    produce OHLC + Volume / Delta / VWAP aggregated by 'rule' (pandas offset string)."""
    if df.empty: return df
    df = df.copy()

    # Delta direction & up/down volume estimation
    df["Delta"] = df.groupby("Symbol")["Close"].diff()

    def compute_dir(s):
        sgn = np.sign(s)
        sgn = pd.Series(sgn).replace(0, np.nan).ffill().fillna(0).values
        return sgn

    df["Dir"] = df.groupby("Symbol")["Delta"].transform(lambda x: compute_dir(x))
    df["UpVol"] = np.where(df["Dir"] == 1, df["Volume"], np.where(df["Dir"] == 0, df["Volume"]/2.0, 0))
    df["DnVol"] = np.where(df["Dir"] == -1, df["Volume"], np.where(df["Dir"] == 0, df["Volume"]/2.0, 0))
    df["UpValue"] = df["Close"] * df["UpVol"]
    df["DnValue"] = df["Close"] * df["DnVol"]
    df["Value"] = df["Close"] * df["Volume"]

    agg_dict = {
        "Close": ["first", "max", "min", "last"],
        "Volume": "sum", "UpVol": "sum", "DnVol": "sum",
        "UpValue": "sum", "DnValue": "sum", "Value": "sum"
    }

    # ensure DateTime index exists for resampling
    if "DateTime" in df.columns:
        df = df.set_index("DateTime")

    df_r = (df.groupby("Symbol")
            .resample(rule, label="left", closed="left")
            .agg(agg_dict)
            .reset_index())

    df_r.columns = ["Symbol","DateTime","Open","High","Low","Close","Volume",
                    "UpVol","DnVol","UpValue","DnValue","Value"]

    df_r = df_r[df_r["Close"].notna()].copy()

    # VWAPs and delta metrics
    df_r["vwap_TMP"] = np.where(df_r["Volume"]>0, df_r["Value"]/df_r["Volume"], np.nan)
    df_r["vwap_BUY"] = np.where(df_r["UpVol"]>0, df_r["UpValue"]/df_r["UpVol"], np.nan)
    df_r["vwap_SELL"] = np.where(df_r["DnVol"]>0, df_r["DnValue"]/df_r["DnVol"], np.nan)

    df_r["DnVol_signed"] = -df_r["DnVol"]
    df_r["DeltaVol"] = df_r["UpVol"] - df_r["DnVol"]
    df_r["DeltaVolPct"] = np.where(df_r["Volume"]>0, df_r["DeltaVol"]/df_r["Volume"]*100, 0.0)

    # ensure numerics (avoid object dtype surprises)
    for c in ["Open","High","Low","Close","Volume","UpVol","DnVol","DeltaVol","DeltaVolPct"]:
        if c in df_r.columns:
            df_r[c] = pd.to_numeric(df_r[c], errors="coerce").fillna(0)
    return df_r

# ---------------- Indicators ----------------
def compute_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
    if close_series is None or len(close_series) == 0:
        return pd.Series([], dtype=float)
    close_series = pd.to_numeric(close_series, errors="coerce")
    delta = close_series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill").fillna(50.0)

# ---------------- Cache helper (with optional df_input + window for virtual scroll) ----------------
def get_resampled(symbol, interval, df_input=None, window=None):
    """
    Parameters:
      symbol: symbol string
      interval: resample rule like '30min', '1D' etc.
      df_input: optional pre-filtered DataFrame (already only rows for that symbol),
                if provided we skip df_base filtering (faster for virtual scroll).
      window: optional tuple (start_ts, end_ts) - tz-aware timestamps to pre-filter df_input before resample.
              if None → use full df_input (or full df_base if df_input None)
    Returns:
      DataFrame resampled for the requested symbol + interval.
    """
    # cache key should include window to avoid collisions between different windows
    wkey = None
    if window is not None:
        start, end = window
        if hasattr(start, "isoformat") and hasattr(end, "isoformat"):
            wkey = (start.isoformat(), end.isoformat())
        else:
            wkey = (str(start), str(end))

    key = (symbol, interval, wkey)
    cached = _resample_cache.get(key)
    if cached is not None:
        print(f"[Cache] HIT {key} (rows={len(cached)})")
        return cached.copy()

    t0 = _time.time()
    if df_input is not None:
        df_symbol = df_input.copy()
    else:
        # df_base is global (loaded once)
        df_symbol = df_base[df_base["Symbol"] == symbol].copy()

    # apply window filter early (small slices → much faster resample)
    if window is not None and len(df_symbol) > 0:
        start, end = window
        df_symbol = df_symbol.loc[(df_symbol.index >= start) & (df_symbol.index <= end)].copy()

    df_res = resample_data(df_symbol, interval)
    dt = _time.time() - t0
    print(f"[Cache] MISS {key} -> computed {len(df_res):,} rows in {dt:.2f}s")
    _resample_cache.set(key, df_res)
    return df_res.copy()

# ---------------- DASH APP ----------------
app = dash.Dash(__name__)
app.title = "Pine-Style Volume Pane (virtual scroll)"

# load base ticks once (vectorized loader)
df_base = load_json_folder()
# deterministic mapping because loader already did UTC->BD conversion
tz_choice = "utc->bd"
symbols = sorted(df_base["Symbol"].unique()) if not df_base.empty else ["NO_DATA"]

# ---------------- UI LAYOUT ----------------
app.layout = html.Div([
    # ==== Top Toolbar ====
    html.Div([
        html.Label("Symbol:", style={"marginRight": "6px", "fontWeight": "600"}),
        dcc.Dropdown(
            options=[{"label": s, "value": s} for s in symbols],
            value=symbols[0],
            id="symbol-dropdown",
            style={"width": "160px", "display": "inline-block", "marginRight": "12px", "verticalAlign": "middle"}
        ),

        html.Label("Interval:", style={"marginRight": "6px", "fontWeight": "600"}),
        dcc.Dropdown(
            options=[{"label": i, "value": i} for i in INTERVALS],
            value="30min",
            id="interval-dropdown",
            style={"width": "120px", "display": "inline-block", "marginRight": "12px", "verticalAlign": "middle"}
        ),

        html.Label("VWAPs:", style={"marginRight": "6px", "fontWeight": "600"}),
        dcc.Checklist(
            id="vwap-toggle",
            options=[{"label": "All", "value": "all"},{"label": "Buy", "value": "buy"},{"label": "Sell", "value": "sell"}],
            value=["all", "buy", "sell"],
            inline=True,
            style={"display": "inline-flex", "alignItems": "center", "marginRight": "14px"}
        ),

        dcc.Checklist(
            id="compress-toggle",
            options=[{"label": "Compress", "value": "compress"}],
            value=["compress"],
            inline=True,
            style={"display": "inline-flex", "alignItems": "center", "marginRight": "14px"}
        ),

        html.Div(f"TZ: {tz_choice}", style={"display": "inline-block", "marginRight": "14px", "color": "#bbbbbb", "fontSize": "13px"}),

        dcc.Checklist(
            id="full-history-toggle",
            options=[{"label": "History", "value": "full"}],
            value=[],
            inline=True,
            style={"display": "inline-flex", "alignItems": "center"}
        ),

        # --- WATCHLIST controls (inline) ---
        dcc.Store(id="watchlist-store", data=[], storage_type="local"),
        html.Div([
            dcc.Input(id="watch-input", placeholder="Symbol", type="text", style={"width": "100px", "marginLeft":"12px"}),
            html.Button("Add", id="add-watch-btn", n_clicks=0, style={"marginLeft":"6px"}),
            dcc.Dropdown(id="remove-dropdown", options=[], placeholder="Remove", style={"width":"120px","display":"inline-block","marginLeft":"8px"}),
            html.Button("Remove", id="remove-watch-btn", n_clicks=0, style={"marginLeft":"6px"}),
            html.Button("Run Scanner", id="run-scanner-btn", n_clicks=0, style={"marginLeft":"12px","background":"#2b7a2b","color":"white"}),
            html.Span(id="scanner-status", style={"marginLeft":"10px","color":"lightgray"})
        ], style={"display":"inline-flex","alignItems":"center","marginLeft":"8px"})
    ],
    style={"display": "flex","alignItems": "center","justifyContent": "center","gap": "8px","flexWrap": "wrap","marginBottom": "8px","textAlign": "center","width": "95%"}
    ),

    # ---- Watchlist results (scanner) ----
    html.Div(id="watchlist-table", style={"margin":"6px auto","width":"95%"}),
    dcc.Interval(id="wl-interval", interval=60*1000, n_intervals=0),

    # ==== Graph placeholder ====
    dcc.Graph(id="chart", style={"height": "120vh"}),
])

# ---------------- WATCHLIST CALLBACK (merged add + remove) ----------------
@app.callback(
    Output("watchlist-table", "children"),
    Input("watchlist-store", "data"),
    Input("interval-dropdown", "value"),
    Input("wl-interval", "n_intervals"),
)
def render_watchlist_table(store, interval, _n):
    # normalize store list
    if not store:
        return html.Div("Watchlist is empty.", style={"color":"#bbb"})
    if isinstance(store, str):
        symbols_list = [store]
    else:
        try:
            symbols_list = list(store)
        except Exception:
            symbols_list = []
    symbols_list = [s.strip().upper() for s in symbols_list if isinstance(s, str) and s.strip()]
    symbols_list = [s for s in symbols_list if s in symbols]
    if not symbols_list:
        return html.Div("No valid symbols in watchlist.", style={"color":"#bbb"})

    rows = []
    for sym in symbols_list:
        # use last 120 days for intraday, else 2 years
        is_intraday = str(interval).upper() not in ("1D", "1W")
        end = df_base.index.max()
        lookback_days = 120 if is_intraday else 730
        start = end - pd.Timedelta(days=lookback_days)
        df_ticks = df_base[df_base["Symbol"] == sym]
        if df_ticks.empty:
            continue
        df_rs = get_resampled(sym, interval, df_input=df_ticks, window=(start, end))
        if df_rs.empty:
            continue
        # last row metrics
        last = df_rs.iloc[-1]
        close = float(last.get("Close", np.nan))
        open_ = float(last.get("Open", np.nan))
        high = float(last.get("High", np.nan))
        low = float(last.get("Low", np.nan))
        vol = float(last.get("Volume", 0.0))
        vwap_all = float(last.get("vwap_TMP", np.nan)) if "vwap_TMP" in df_rs.columns else np.nan
        delta_vol_pct = float(last.get("DeltaVolPct", 0.0))

        # compute change from previous close
        prev_close = float(df_rs["Close"].iloc[-2]) if len(df_rs) > 1 else np.nan
        chg = (close - prev_close) if not np.isnan(prev_close) else np.nan
        chg_pct = (chg / prev_close * 100.0) if prev_close and not np.isnan(prev_close) and prev_close != 0 else np.nan

        # compute RSI over closes
        rsi = float(compute_rsi(df_rs["Close"], period=14).iloc[-1]) if len(df_rs) >= 14 else np.nan

        rows.append({
            "Symbol": sym,
            "Price": round(close, 2) if np.isfinite(close) else None,
            "Chg%": round(chg_pct, 2) if np.isfinite(chg_pct) else None,
            "Vol": int(vol),
            "ΔVol%": round(delta_vol_pct, 1) if np.isfinite(delta_vol_pct) else None,
            "VWAP": round(vwap_all, 2) if np.isfinite(vwap_all) else None,
            "RSI14": round(rsi, 1) if np.isfinite(rsi) else None,
        })

    if not rows:
        return html.Div("No data for watchlist.", style={"color":"#bbb"})

    columns = [
        {"name": "Symbol", "id": "Symbol"},
        {"name": "Price", "id": "Price", "type": "numeric", "format": {"specifier": ",.2f"}},
        {"name": "Chg%", "id": "Chg%", "type": "numeric", "format": {"specifier": ",.2f"}},
        {"name": "Vol", "id": "Vol", "type": "numeric", "format": {"specifier": ","}},
        {"name": "ΔVol%", "id": "ΔVol%", "type": "numeric", "format": {"specifier": ",.1f"}},
        {"name": "VWAP", "id": "VWAP", "type": "numeric", "format": {"specifier": ",.2f"}},
        {"name": "RSI14", "id": "RSI14", "type": "numeric", "format": {"specifier": ",.1f"}},
    ]

    style_data_conditional = [
        {
            "if": {"filter_query": '{Chg%} > 0', "column_id": "Chg%"},
            "color": "#1ecb1e"
        },
        {
            "if": {"filter_query": '{Chg%} < 0', "column_id": "Chg%"},
            "color": "#ff5c5c"
        },
        {
            "if": {"filter_query": '{ΔVol%} > 0', "column_id": "ΔVol%"},
            "color": "#e6d200"
        },
        {
            "if": {"filter_query": '{RSI14} >= 70', "column_id": "RSI14"},
            "backgroundColor": "#3d1f1f"
        },
        {
            "if": {"filter_query": '{RSI14} <= 30', "column_id": "RSI14"},
            "backgroundColor": "#1f3d1f"
        },
    ]

    return dash_table.DataTable(
        id="watch-datatable",
        columns=columns,
        data=rows,
        sort_action="native",
        filter_action="native",
        page_action="none",
        style_table={"overflowX": "auto"},
        style_as_list_view=True,
        style_header={"backgroundColor": "#222", "fontWeight": "600"},
        style_cell={"backgroundColor": "#111", "color": "#ddd", "padding": "6px", "fontFamily": "Arial", "fontSize": 13},
        style_data_conditional=style_data_conditional,
        row_selectable="single",
    )

@app.callback(
    Output("watchlist-store", "data"),
    Input("add-watch-btn", "n_clicks"),
    Input("remove-watch-btn", "n_clicks"),
    State("watch-input", "value"),
    State("remove-dropdown", "value"),
    State("watchlist-store", "data"),
    prevent_initial_call=True
)
def modify_watch(add_n, remove_n, add_val, remove_val, store):
    """
    Handles Add / Remove symbol actions for watchlist.
    - Fully permissive for Add: allows manual symbols (not limited to symbols list)
    - Still sanitizes list and prevents overwriting.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate()
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    # --- Normalize the store ---
    if store is None:
        store = []
    if isinstance(store, str):
        store = [store]
    try:
        store = list(store)
    except Exception:
        store = []

    # --- Clean up values (upper + unique) ---
    clean_list = []
    for s in store:
        if not isinstance(s, str):
            continue
        s = s.strip().upper()
        if s and s not in clean_list:
            clean_list.append(s)
    store = clean_list

    # --- Add logic ---
    if trigger == "add-watch-btn":
        if not add_val:
            return store
        val = str(add_val).strip().upper()
        if val == "":
            return store
        if val not in store:
            store.append(val)
        return store

    # --- Remove logic ---
    if trigger == "remove-watch-btn":
        if remove_val is None:
            return store
        if isinstance(remove_val, str):
            remove_val = remove_val.strip().upper()
        store = [s for s in store if s != remove_val]
        return store

    return store


# ---------------- Scanner status wiring ----------------
@app.callback(
    Output("scanner-status", "children"),
    Input("run-scanner-btn", "n_clicks"),
    State("watchlist-store", "data"),
    prevent_initial_call=True,
)
def run_scanner(n_clicks, store):
    # Placeholder scanner: just echo count and timestamp
    try:
        count = len(store) if isinstance(store, list) else (1 if isinstance(store, str) and store else 0)
    except Exception:
        count = 0
    ts = datetime.now(BD_TZ).strftime("%Y-%m-%d %H:%M:%S")
    return f"Scanned {count} symbols at {ts}"


# ---------------- Row click -> load symbol ----------------
@app.callback(
    Output("symbol-dropdown", "value"),
    Input("watch-datatable", "active_cell"),
    State("watch-datatable", "data"),
    prevent_initial_call=True,
)
def load_symbol_from_watch(active_cell, table_data):
    if not active_cell or not table_data:
        raise dash.exceptions.PreventUpdate()
    row = active_cell.get("row")
    if row is None or row >= len(table_data):
        raise dash.exceptions.PreventUpdate()
    sym = table_data[row].get("Symbol")
    if not sym:
        raise dash.exceptions.PreventUpdate()
    return sym


# ---------------- update remove-dropdown options (defensive) ----------------
@app.callback(
    Output("remove-dropdown", "options"),
    Input("watchlist-store", "data")
)
def update_remove_options(store):
    """
    Build remove-dropdown options safely from store.
    Accepts None, string, list.
    """
    if store is None:
        return []
    # if store is a string, convert to single-item list
    if isinstance(store, str):
        store = [store]
    # ensure list
    try:
        items = list(store)
    except Exception:
        items = []

    # normalize items to uppercase known symbols and remove duplicates
    opts = []
    seen = set()
    for s in items:
        if not isinstance(s, str):
            continue
        su = s.strip().upper()
        if not su:
            continue
        if su in symbols and su not in seen:
            seen.add(su)
            opts.append({"label": su, "value": su})
    return opts

# ---------------- CHART CALLBACK ----------------
@app.callback(
    Output("chart", "figure"),
    Input("symbol-dropdown", "value"),
    Input("interval-dropdown", "value"),
    Input("vwap-toggle", "value"),
    Input("compress-toggle", "value"),
    Input("full-history-toggle", "value"),
)
def update_chart(symbol, interval, vwap_toggle, compress_vals, full_history_vals):
    # basic guards
    if symbol == "NO_DATA" or df_base.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="No data available")
        return fig

    # pre-filter ticks for the chosen symbol once (we'll pass this df_input into get_resampled)
    df_symbol_ticks = df_base[df_base["Symbol"] == symbol].copy()
    if df_symbol_ticks.empty:
        fig = go.Figure(); fig.update_layout(template="plotly_dark", title=f"No tick rows for {symbol}")
        return fig

    # Virtual-scrolling heuristic:
    load_full = ("full" in (full_history_vals or []))
    is_intraday = str(interval).upper() not in ("1D", "1W")

    # choose default window size:
    if load_full:
        window = None
    else:
        if is_intraday:
            days = 90
        else:
            days = 730
        end = df_symbol_ticks.index.max()
        start = end - pd.Timedelta(days=days)
        window = (start, end)

    # get resampled data using df_input pre-filter + optional window (fast)
    df = get_resampled(symbol, interval, df_input=df_symbol_ticks, window=window)
    if df.empty:
        fig = go.Figure(); fig.update_layout(template="plotly_dark", title="No resampled bars")
        return fig

    # Ensure we have OHLC columns; if missing, build synthetic from Close
    if df[["Open","High","Low","Close"]].dropna().empty and "Close" in df.columns and df["Close"].notna().any():
        df["Open"] = df["Close"]; df["High"] = df["Close"]; df["Low"] = df["Close"]

    # numeric safety
    for c in ["Open", "High", "Low", "Close", "Volume", "UpVol", "DnVol", "DeltaVol", "DeltaVolPct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # build traces
    traces = [
        go.Candlestick(
            x=df["DateTime"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="limegreen",
            decreasing_line_color="red",
            showlegend=False,
        )
    ]

    if "all" in vwap_toggle and "vwap_TMP" in df.columns:
        traces.append(go.Scatter(x=df["DateTime"], y=df["vwap_TMP"], mode="lines", name="VWAP (All)", line=dict(color="orange", width=2)))
    if "buy" in vwap_toggle and "vwap_BUY" in df.columns:
        traces.append(go.Scatter(x=df["DateTime"], y=df["vwap_BUY"], mode="lines", name="VWAP (Buy)", line=dict(color="lime", width=1.5)))
    if "sell" in vwap_toggle and "vwap_SELL" in df.columns:
        traces.append(go.Scatter(x=df["DateTime"], y=df["vwap_SELL"], mode="lines", name="VWAP (Sell)", line=dict(color="red", width=1.5)))

    traces += [
        go.Bar(x=df["DateTime"], y=df["Volume"], name="Total Vol", marker_color="#33849b", opacity=0.5, yaxis="y2"),
        go.Bar(x=df["DateTime"], y=df["UpVol"], name="UpVol", marker_color="#26a69a", opacity=0.6, yaxis="y2"),
        go.Bar(x=df["DateTime"], y=-df["DnVol"], name="DnVol", marker_color="#FC2B31", opacity=0.6, yaxis="y2"),
        go.Scatter(x=df["DateTime"], y=df["DeltaVolPct"], mode="lines", name="ΔVol%", line=dict(color="yellow", width=1.5, dash="dot"), yaxis="y3"),
        go.Scatter(x=df["DateTime"], y=df["DeltaVol"], mode="text", text=["—"]*len(df), textfont=dict(color=np.where(df["DeltaVol"]>=0,"lime","red"), size=10), name="ΔVol Char", yaxis="y2", showlegend=False)
    ]

    fig = go.Figure(data=traces)

    # Rangebreak / compression logic (Fri+Sat + off-hours + holidays when requested)
    compress = ("compress" in (compress_vals or []))
    is_daily = str(interval).upper() in ("1D", "1W")
    rb = []
    if compress and not is_daily:
        # hide Fri + Sat
        rb.append(dict(bounds=["fri","sun"]))
        # hide hours outside trading window
        e_h = TRADING_END.hour + TRADING_END.minute/60.0 + TRADING_END.second/3600.0 + 1e-6
        s_h = TRADING_START.hour + TRADING_START.minute/60.0 + TRADING_START.second/3600.0
        rb.append(dict(bounds=[e_h, s_h], pattern="hour"))
        # include holidays by date strings
        if len(HOLIDAYS) > 0:
            rb.append(dict(values=[d.strftime("%Y-%m-%d") for d in HOLIDAYS]))
    elif compress and is_daily:
        rb.append(dict(bounds=["fri","sun"]))
        if len(HOLIDAYS) > 0:
            rb.append(dict(values=[d.strftime("%Y-%m-%d") for d in HOLIDAYS]))

    # build sensible tick values: intraday show hour markers inside session; daily show date ticks
    tickvals = None
    ticktext = None
    if is_daily:
        tickvals = list(df["DateTime"])
        ticktext = [ts.strftime("%Y-%m-%d") for ts in tickvals]
    else:
        tickvals_list = []
        ticktext_list = []
        for ts in df["DateTime"]:
            lt = ts.tz_convert(BD_TZ).time()
            # only label whole hours inside trading hours
            if lt.minute == 0 and (TRADING_START <= lt <= TRADING_END):
                tickvals_list.append(ts)
                ticktext_list.append(ts.strftime("%H:%M"))
        if tickvals_list:
            tickvals = tickvals_list
            ticktext = ticktext_list

    # Add rangeslider + rangeselector (TradingView-like quick zoom). Rangeslider enables "scroll".
    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        xaxis=dict(
            showgrid=False,
            type="date",
            tickangle=-45,
            rangeslider=dict(visible=True, thickness=0.06),
            rangeselector=dict(
                buttons=[
                    dict(count=5, label="5D", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all")
                ],
                bgcolor="#818181",
                activecolor="#558D55",
                font=dict(color='white', family='Arial', size=12),
            ),
            rangeslider_visible=True,
            rangebreaks=rb,
            tickvals=tickvals,
            ticktext=ticktext,
            tickformat="%H:%M\n%d-%b" if not is_daily else "%Y-%m-%d",
        ),
        yaxis=dict(showgrid=False, title="Price + VWAPs", domain=[0.33, 1]),
        yaxis2=dict(title="Volume", domain=[0, 0.3], side="right", showgrid=True),
        yaxis3=dict(title="ΔVol%", overlaying="y2", side="left", showgrid=False),
        barmode="overlay",
        margin=dict(l=60, r=60, t=60, b=60)
    )

    return fig

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=False)
