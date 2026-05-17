from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "session_logs"


def exercise_from_file(path):
    parts = path.stem.split("_")
    if len(parts) > 2 and parts[-1].isdigit() and parts[-2].isdigit():
        parts = parts[:-2]
    return " ".join(parts).title()


def load_session(path):
    df = pd.read_csv(path)
    unit_column = "rep" if "rep" in df.columns else "second"

    df = df.rename(columns={unit_column: "unit"})
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["exercise"] = exercise_from_file(path)
    df["session"] = path.stem
    session_time = df["timestamp"].dropna().min()
    if pd.isna(session_time):
        session_label = f"{exercise_from_file(path)} - {path.stem}"
    else:
        session_label = f"{exercise_from_file(path)} - {session_time:%d %b %H:%M}"
    df["session_label"] = session_label
    df["unit_label"] = unit_column.title()
    return df.dropna(subset=["score"])


@st.cache_data
def load_all_sessions():
    files = sorted(LOG_DIR.glob("*.csv"), key=lambda path: path.stat().st_mtime)
    frames = []

    for path in files:
        try:
            frames.append(load_session(path))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def split_errors(errors):
    if pd.isna(errors) or errors in ("", "None"):
        return []
    return [error.strip() for error in str(errors).split(";") if error.strip()]


def format_score(value):
    return f"{value:.1f}%" if pd.notna(value) else "-"


def show_empty_state():
    st.info(
        "No session CSVs found yet. Complete an exercise session first, then refresh this dashboard."
    )
    st.caption(f"Expected log folder: {LOG_DIR}")


st.set_page_config(
    page_title="Exercise Form Dashboard",
    layout="wide",
)

st.title("Exercise Form Dashboard")
st.caption("Session scores, rep trends, and the form errors that show up most often.")

data = load_all_sessions()

if data.empty:
    show_empty_state()
    st.stop()

with st.sidebar:
    st.header("Filters")
    exercises = sorted(data["exercise"].dropna().unique())
    selected_exercises = st.multiselect("Exercise", exercises, default=exercises)

    filtered = data[data["exercise"].isin(selected_exercises)]
    sessions = ["All sessions"] + sorted(filtered["session_label"].dropna().unique())
    selected_session = st.selectbox("Session", sessions)

    if selected_session != "All sessions":
        filtered = filtered[filtered["session_label"] == selected_session]

    st.divider()
    st.caption("Place new CSV files in `session_logs`, then use Refresh.")
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

filtered = filtered.sort_values(["timestamp", "session", "unit"])

if filtered.empty:
    show_empty_state()
    st.stop()

avg_score = filtered["score"].mean()
best_score = filtered["score"].max()
worst_score = filtered["score"].min()
score_change = filtered["score"].iloc[-1] - filtered["score"].iloc[0]
session_count = filtered["session"].nunique()
unit_total = len(filtered)

metric_cols = st.columns(5)
metric_cols[0].metric("Logged points", unit_total)
metric_cols[1].metric("Sessions", session_count)
metric_cols[2].metric("Average score", format_score(avg_score))
metric_cols[3].metric("Best score", format_score(best_score))
metric_cols[4].metric("Score change", format_score(score_change))

st.subheader("Score Trend")
trend_chart = (
    alt.Chart(filtered)
    .mark_line(point=True)
    .encode(
        x=alt.X("unit:Q", title="Rep / Second"),
        y=alt.Y("score:Q", title="Form Score (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("session_label:N", title="Session"),
        tooltip=[
            alt.Tooltip("exercise:N", title="Exercise"),
            alt.Tooltip("unit:Q", title="Rep / Second"),
            alt.Tooltip("score:Q", title="Score", format=".1f"),
            alt.Tooltip("errors:N", title="Errors"),
            alt.Tooltip("timestamp:T", title="Time"),
        ],
    )
    .properties(height=360)
)
st.altair_chart(trend_chart, width="stretch")

left, right = st.columns([2, 1])

with left:
    st.subheader("Session Summary")
    summary = (
        filtered.groupby(["session_label", "exercise"], as_index=False)
        .agg(
            logged_points=("score", "count"),
            average_score=("score", "mean"),
            best_score=("score", "max"),
            worst_score=("score", "min"),
        )
        .sort_values("session_label", ascending=False)
    )
    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
        column_config={
            "session_label": "Session",
            "exercise": "Exercise",
            "logged_points": "Logged Points",
            "average_score": st.column_config.NumberColumn(
                "Average Score", format="%.1f%%"
            ),
            "best_score": st.column_config.NumberColumn("Best Score", format="%.1f%%"),
            "worst_score": st.column_config.NumberColumn(
                "Worst Score", format="%.1f%%"
            ),
        },
    )

with right:
    st.subheader("Common Errors")
    error_counts = Counter()
    for error_text in filtered["errors"]:
        error_counts.update(split_errors(error_text))

    if error_counts:
        error_df = pd.DataFrame(
            error_counts.most_common(), columns=["error", "count"]
        )
        error_chart = (
            alt.Chart(error_df)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("error:N", title=None, sort="-x"),
                tooltip=["error", "count"],
            )
            .properties(height=280)
        )
        st.altair_chart(error_chart, width="stretch")
    else:
        st.success("No errors logged in the selected data.")

st.subheader("Rep Log")
display_df = filtered[
    ["timestamp", "exercise", "unit_label", "unit", "score", "errors", "session"]
].sort_values("timestamp", ascending=False)
st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
    column_config={
        "timestamp": "Time",
        "exercise": "Exercise",
        "unit_label": "Type",
        "unit": "Rep / Second",
        "score": st.column_config.NumberColumn("Score", format="%.1f%%"),
        "errors": "Errors",
        "session": "CSV Session",
    },
)
