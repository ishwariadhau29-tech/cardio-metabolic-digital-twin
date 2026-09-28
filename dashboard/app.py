"""Streamlit clinician/patient demo dashboard for the synthetic health twin."""

from pathlib import Path
from html import escape
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.simulator import (
    generate_synthetic_history,
    get_synthetic_alerts,
    load_observations,
    load_profiles,
    replay_observation,
)
from models.risk_engine import (
    assess_risk,
    RISK_FACTOR_GUIDANCE,
    calculate_simulated_state,
    calculate_twin_state,
    generate_scenario_explanations,
    get_risk_contributors,
)


st.set_page_config(
    page_title="Twin Health | Metabolic Monitor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --ink: #172923;
        --muted: #4f655a;
        --forest: #173d32;
        --forest-light: #275b49;
        --mint: #e4f0e9;
        --paper: #f3f6f3;
        --line: #dce5df;
        --white: #ffffff;
        --coral: #c95743;
        --blue: #2b7380;
    }

    html, body, [class*="st-"] {
        font-family: 'DM Sans', 'Segoe UI', sans-serif;
        color: var(--ink);
    }
    .stApp { background: var(--paper); }
    [data-testid="stAppViewContainer"] { background: var(--paper); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 1360px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    h1 { font-size: 1.65rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.05rem !important; font-weight: 700 !important; }
    h3 { color: var(--forest); font-size: 0.95rem !important; font-weight: 700 !important; }
    p, label { color: var(--ink); }
    [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }

    .twin-masthead {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        min-height: 86px;
        margin: 0 0 1.35rem;
        padding: 1.15rem 1.4rem;
        color: #f3f7f2;
        background: var(--forest);
        border-radius: 8px;
        border-bottom: 3px solid #81b99d;
    }
    .twin-brand { display: flex; align-items: center; gap: 0.8rem; }
    .twin-brand-mark {
        display: grid;
        width: 38px;
        height: 38px;
        place-items: center;
        border: 1px solid #89b9a1;
        border-radius: 50%;
        color: #c8e6d5;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.76rem;
        font-weight: 600;
    }
    .twin-brand-name { font-size: 1.08rem; font-weight: 700; line-height: 1.15; }
    .twin-brand-subtitle {
        margin-top: 0.2rem;
        color: #bad0c5;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        text-transform: uppercase;
    }
    .twin-live-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.55rem;
        border: 1px solid #668b78;
        border-radius: 4px;
        color: #d2e5d9;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.66rem;
        white-space: nowrap;
    }
    .twin-header-meta {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0.3rem;
    }
    .twin-status-dot {
        width: 7px;
        height: 7px;
        background: #a9d6b9;
        border-radius: 50%;
    }
    .twin-header-detail {
        color: #c6d8cd;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.58rem;
        text-align: right;
    }
    .st-key-stream-controls {
        padding: 0.25rem 0.75rem;
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 6px;
    }
    .st-key-stream-controls [data-testid="stCaptionContainer"] p { margin-bottom: 0; }
    .stream-state {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin: 0.15rem 0 0.8rem;
        padding: 0.65rem 0.8rem;
        color: var(--forest);
        background: var(--mint);
        border: 1px solid #bbd3c4;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
    }
    .stream-state strong { font-size: 0.7rem; }
    .stream-state span:last-child { margin-left: auto; color: #42594f; }
    .stream-state-dot {
        width: 8px;
        height: 8px;
        flex: 0 0 8px;
        background: #32815d;
        border-radius: 50%;
    }
    .scenario-flow {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
        margin: 0.8rem 0;
        padding: 0.8rem 1rem;
        color: var(--forest);
        background: var(--mint);
        border: 1px solid #bbd3c4;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
    }
    .scenario-flow b { color: var(--coral); font-size: 1rem; }

    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background: var(--forest) !important;
        border-right: 1px solid #234c3f;
    }
    button[data-testid="stBaseButton-headerNoPadding"] {
        position: relative;
        width: 2.4rem;
        min-width: 2.4rem;
        height: 2.4rem;
        color: transparent !important;
        border-radius: 5px;
    }
    button[data-testid="stBaseButton-headerNoPadding"] [data-testid="stIconMaterial"] {
        color: transparent !important;
        font-size: 0 !important;
    }
    button[data-testid="stBaseButton-headerNoPadding"]::after {
        position: absolute;
        inset: 0;
        display: grid;
        place-items: center;
        color: var(--forest);
        content: "‹";
        font-family: 'DM Sans', 'Segoe UI', sans-serif;
        font-size: 1.8rem;
        line-height: 1;
    }
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"]::after {
        color: #ffffff;
    }
    button[data-testid="stBaseButton-headerNoPadding"]:hover {
        background: rgba(255, 255, 255, 0.12) !important;
    }
    button[data-testid="stBaseButton-headerNoPadding"]:focus-visible {
        outline: 2px solid #a9d6b9;
        outline-offset: 2px;
    }
    button[data-testid="stExpandSidebarButton"] {
        position: relative;
        width: 2.4rem;
        min-width: 2.4rem;
        height: 2.4rem;
        color: transparent !important;
        border-radius: 5px;
    }
    button[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {
        color: transparent !important;
        font-size: 0 !important;
    }
    button[data-testid="stExpandSidebarButton"]::after {
        position: absolute;
        inset: 0;
        display: grid;
        place-items: center;
        color: var(--forest);
        content: "›";
        font-family: 'DM Sans', 'Segoe UI', sans-serif;
        font-size: 1.8rem;
        line-height: 1;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] strong {
        color: #eff5f0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #d0ded5 !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #f9fbf9;
        border-color: #aac1b4;
    }
    section[data-testid="stSidebar"] hr { border-color: #426555; }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #adc6b8 !important;
    }
    .patient-profile-card {
        margin-top: 0.75rem;
        padding: 1rem;
        color: #eff5f0;
        background: #205142;
        border: 1px solid #477262;
        border-radius: 7px;
    }
    .patient-profile-head {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding-bottom: 0.9rem;
        border-bottom: 1px solid #527666;
    }
    .patient-monogram {
        display: grid;
        width: 38px;
        height: 38px;
        flex: 0 0 38px;
        place-items: center;
        color: var(--forest);
        background: #ffffff;
        border-radius: 50%;
        font-weight: 700;
    }
    .patient-name { color: #ffffff; font-size: 0.98rem; font-weight: 700; }
    .patient-id {
        margin-top: 0.15rem;
        color: #d0ded5;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.66rem;
    }
    .patient-profile-label {
        margin: 0.85rem 0 0.45rem;
        color: #d0ded5;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        text-transform: uppercase;
    }
    .patient-facts {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.7rem 0.5rem;
    }
    .patient-fact-label { color: #d0ded5; font-size: 0.68rem; }
    .patient-fact-value { margin-top: 0.1rem; color: #ffffff; font-size: 0.9rem; font-weight: 600; }
    .patient-history {
        margin-top: 0.85rem;
        padding-top: 0.75rem;
        border-top: 1px solid #527666;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] .patient-history p {
        margin: 0.2rem 0 0;
        color: #eff5f0 !important;
        font-size: 0.78rem;
        line-height: 1.4;
    }

    .safety-notice {
        margin: 1.35rem 0 0.6rem;
        padding: 0.95rem 1.1rem;
        background: #fff1ed;
        border: 1px solid #e8b2a5;
        border-left: 5px solid var(--coral);
        border-radius: 6px;
    }
    .safety-notice-label {
        margin-bottom: 0.25rem;
        color: #913b2d;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.67rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .safety-notice strong { color: #48251e; font-size: 0.94rem; }
    .safety-notice p { margin: 0.35rem 0 0; color: #583d36; font-size: 0.82rem; line-height: 1.5; }
    .twin-footer {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: center;
        gap: 0.75rem 1rem;
        margin-top: 1.25rem;
        padding: 1rem 1.15rem;
        color: #eff5f0;
        background: var(--forest);
        border-top: 3px solid #81b99d;
        border-radius: 6px;
    }
    .twin-footer-brand { color: #ffffff; font-size: 0.82rem; font-weight: 700; }
    .twin-footer-detail {
        margin-top: 0.18rem;
        color: #d0ded5;
        font-size: 0.7rem;
        line-height: 1.4;
    }
    .twin-footer-status {
        flex: 0 0 auto;
        padding: 0.35rem 0.55rem;
        color: #ffffff;
        background: #275b49;
        border: 1px solid #81b99d;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.63rem;
        font-weight: 500;
    }
    .twin-footer-note {
        grid-column: 1 / -1;
        padding-top: 0.65rem;
        color: #d0ded5;
        border-top: 1px solid #426555;
        font-size: 0.68rem;
        line-height: 1.45;
    }

    [data-testid="stMetric"] {
        min-height: 108px;
        padding: 0.95rem 1rem;
        background: var(--white);
        border: 1px solid var(--line);
        border-top: 3px solid #a7cbb5;
        border-radius: 7px;
    }
    [data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-size: 0.73rem;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.6rem; }
    [data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.69rem; }
    [data-testid="stAlert"] { border-radius: 6px; }
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        color: var(--ink) !important;
    }
    [data-testid="stSegmentedControl"] {
        width: 100%;
        margin: 0.15rem 0 0.8rem;
    }
    [data-testid="stSegmentedControl"] button {
        min-height: 2.8rem;
        color: var(--ink) !important;
        font-weight: 600;
    }
    [data-testid="stSegmentedControl"] button[aria-checked="true"] {
        color: #ffffff !important;
        background: var(--forest) !important;
        border-color: var(--forest) !important;
    }
    [data-testid="stSegmentedControl"] button[aria-checked="true"] p,
    [data-testid="stSegmentedControl"] button[aria-checked="true"] span {
        color: #ffffff !important;
    }
    [data-testid="stPlotlyChart"] {
        padding: 0.4rem 0.5rem 0.1rem;
        background: var(--white);
        border: 1px solid var(--line);
        border-radius: 7px;
    }
    [data-testid="stSlider"] label { color: var(--ink) !important; font-weight: 600; }
    [data-testid="stSliderThumbValue"] {
        padding: 0.12rem 0.3rem;
        color: var(--forest) !important;
        background: #ffffff !important;
        border: 1px solid #afc7b8;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(23, 41, 35, 0.16);
    }
    [data-testid="stSliderThumbValue"] p {
        margin: 0;
        color: var(--forest) !important;
    }
    [data-testid="stButton"] button {
        min-height: 2.55rem;
        background-color: var(--white) !important;
        border-radius: 5px;
        border-color: #afc7b8;
        color: var(--forest) !important;
        font-weight: 600;
    }
    [data-testid="stButton"] button p,
    [data-testid="stButton"] button span,
    [data-testid="stButton"] button svg {
        color: var(--forest) !important;
        fill: var(--forest) !important;
    }
    [data-testid="stButton"] button:hover {
        background-color: var(--mint) !important;
        border-color: var(--forest-light);
        color: var(--forest-light) !important;
    }
    @media (max-width: 700px) {
        .block-container { padding: 1rem 0.8rem 2rem; }
        .twin-masthead { align-items: flex-start; min-height: 0; padding: 0.85rem; }
        .twin-brand { gap: 0.55rem; }
        .twin-brand-mark { width: 34px; height: 34px; }
        .twin-live-tag { font-size: 0.55rem; }
        .twin-header-detail { max-width: 130px; font-size: 0.52rem; }
        .stream-controls { padding: 0.5rem 0.65rem; }
        .stream-state { align-items: flex-start; flex-wrap: wrap; }
        .stream-state span:last-child { margin-left: 0; }
        .scenario-flow { align-items: flex-start; flex-direction: column; }
        .twin-footer { align-items: flex-start; flex-direction: column; }
        .twin-footer { grid-template-columns: minmax(0, 1fr); }
        .twin-footer-status { white-space: normal; }
        .twin-footer-note { grid-column: 1; }
        [data-testid="stMetric"] { min-height: 94px; padding: 0.7rem; }
        [data-testid="stMetricValue"] { font-size: 1.25rem; }
        [data-testid="stSegmentedControl"] button { min-height: 2.6rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data() -> tuple[list[dict], list[dict]]:
    return load_profiles(), load_observations()


def style_chart(figure):
    figure.update_layout(
        font={"family": "DM Sans, Segoe UI, sans-serif", "color": "#172923"},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    figure.update_xaxes(
        title_font_color="#42594f",
        tickfont_color="#52665d",
        gridcolor="#e2e9e4",
        zerolinecolor="#cbd8d0",
    )
    figure.update_yaxes(
        title_font_color="#42594f",
        tickfont_color="#52665d",
        gridcolor="#e2e9e4",
        zerolinecolor="#cbd8d0",
    )
    return figure


def signal_delta(
    previous: dict | None,
    current: dict,
    key: str,
    unit: str = "",
    precision: int = 0,
) -> str | None:
    """Format a neutral direction cue for two adjacent synthetic readings."""
    if previous is None:
        return None
    difference = float(current[key]) - float(previous[key])
    if abs(difference) < 0.05:
        return "→ no change"
    direction = "↑" if difference > 0 else "↓"
    value = f"{abs(difference):,.{precision}f}" if precision else f"{abs(difference):,.0f}"
    return f"{direction} {value}{f' {unit}' if unit else ''}"


profiles, observations = get_data()
if not profiles or not observations:
    st.error("The synthetic demo datasets are empty. Check the files in data/.")
    st.stop()

st.markdown(
        """
        <header class="twin-masthead">
            <div class="twin-brand">
                <div class="twin-brand-mark">TH</div>
                <div>
                    <div class="twin-brand-name">Twin Health</div>
                    <div class="twin-brand-subtitle">Cardiometabolic monitor</div>
                </div>
            </div>
            <div class="twin-header-meta">
                <div class="twin-live-tag"><span class="twin-status-dot"></span>DEMO ENVIRONMENT</div>
                <div class="twin-header-detail">SYNTHETIC EHR + WEARABLE STREAM</div>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Patient profile")
    profile_by_id = {row["patient_id"]: row for row in profiles}
    selected_id = st.selectbox(
        "Patient profile",
        options=list(profile_by_id),
        format_func=lambda patient_id: (
            f"{profile_by_id[patient_id]['display_name']} · {patient_id}"
        ),
    )
    profile = profile_by_id[selected_id]
    initials = "".join(part[0] for part in profile["display_name"].split() if part)[:2].upper()
    summary_html = (
        '<section class="patient-profile-card" aria-label="Synthetic patient profile">'
        f'<div class="patient-profile-head"><div class="patient-monogram">{escape(initials)}</div>'
        f'<div><div class="patient-name">{escape(profile["display_name"])}</div>'
        f'<div class="patient-id">SYNTHETIC RECORD · {escape(profile["patient_id"])}</div></div></div>'
        '<div class="patient-profile-label">Baseline EHR</div><div class="patient-facts">'
        f'<div><div class="patient-fact-label">Age</div><div class="patient-fact-value">{profile["age"]} years</div></div>'
        f'<div><div class="patient-fact-label">Sex</div><div class="patient-fact-value">{escape(profile["sex"])}</div></div>'
        f'<div><div class="patient-fact-label">BMI</div><div class="patient-fact-value">{profile["bmi"]:.1f} kg/m²</div></div>'
        f'<div><div class="patient-fact-label">HbA1c</div><div class="patient-fact-value">{profile["hba1c_pct"]:.1f}%</div></div>'
        f'<div><div class="patient-fact-label">Total cholesterol</div><div class="patient-fact-value">{profile["total_cholesterol_mg_dl"]:.0f} mg/dL</div></div>'
        '</div><div class="patient-history"><div class="patient-fact-label">Medical history</div>'
        f'<p>{escape(profile["medical_history"])}</p></div></section>'
    )
    st.markdown(summary_html, unsafe_allow_html=True)

patient_rows = [row for row in observations if row["patient_id"] == selected_id]
patient_rows.sort(key=lambda row: row["timestamp"])
if selected_id not in st.session_state:
    st.session_state[selected_id] = -1

with st.container(key="stream-controls"):
    control_left, control_right = st.columns([1, 3], vertical_alignment="center")
    with control_left:
        auto_advance = st.toggle("Auto-advance stream", value=False)
    with control_right:
        st.caption("Replay bundled synthetic wearable readings manually or every 5 seconds.")


@st.fragment(run_every="5s" if auto_advance else None)
def live_dashboard() -> None:
    cursor = st.session_state[selected_id]
    if auto_advance or cursor < 0:
        previous = patient_rows[cursor] if cursor >= 0 else None
        current, cursor = replay_observation(observations, selected_id, cursor)
        st.session_state[selected_id] = cursor
    else:
        current = patient_rows[cursor]
        previous = patient_rows[cursor - 1] if cursor > 0 else None

    with st.container(border=True):
        navigation_column, advance_column = st.columns([3, 1], vertical_alignment="bottom")
        with navigation_column:
            selected_view = st.segmented_control(
                "Dashboard section",
                options=["Live overview", "Trends", "What-if"],
                default="Live overview",
                key="dashboard_section",
                label_visibility="visible",
                width="stretch",
                wrap=True,
            )
        with advance_column:
            if st.button("Next reading", icon="⏭️", help="Advance the synthetic wearable stream"):
                previous = current
                current, cursor = replay_observation(observations, selected_id, cursor)
                st.session_state[selected_id] = cursor

    assessment = assess_risk(profile, current)
    twin_state = calculate_twin_state(profile, current)

    if selected_view == "Live overview":
        stream_state = "STREAMING" if auto_advance else "PAUSED · MANUAL REPLAY"
        st.markdown(
            f'<div class="stream-state"><span class="stream-state-dot"></span>'
            f'<strong>LIVE WEARABLE STREAM</strong><span>{stream_state}</span></div>',
            unsafe_allow_html=True,
        )
        st.subheader("Digital Twin State")
        st.caption(
            "Synchronized from the selected synthetic EHR profile and current synthetic wearable reading."
        )
        with st.container(border=True):
            first_row = st.columns(4)
            previous_risk = assess_risk(profile, previous).score if previous else None
            risk_delta = None
            if previous_risk is not None:
                difference = twin_state.score - previous_risk
                risk_delta = "→ no change" if difference == 0 else f"{'↑' if difference > 0 else '↓'} {abs(difference)} points"
            first_row[0].metric(
                "Metabolic risk indicator",
                f"{twin_state.score} / 100",
                risk_delta,
                delta_color="off",
                help="An unvalidated, additive demonstration score. See Trends for the exact illustrative factor thresholds.",
            )
            first_row[1].metric(
                "Glucose",
                f"{twin_state.glucose_mg_dl:.0f} mg/dL",
                signal_delta(previous, current, "glucose_mg_dl", "mg/dL"),
                delta_color="off",
                help="Current synthetic wearable/CGM-style sample; not a real device reading.",
            )
            first_row[2].metric(
                "Heart rate",
                f"{twin_state.heart_rate_bpm:.0f} bpm",
                signal_delta(previous, current, "heart_rate_bpm", "bpm"),
                delta_color="off",
                help="Current synthetic wearable sample.",
            )
            first_row[3].metric(
                "Systolic BP",
                f"{twin_state.systolic_bp_mmhg:.0f} mmHg",
                signal_delta(previous, current, "systolic_bp_mmhg", "mmHg"),
                delta_color="off",
                help="Current synthetic home-reading sample; this is not a measurement or alert for a real person.",
            )

            second_row = st.columns(4)
            second_row[0].metric(
                "Daily steps",
                f"{twin_state.steps:,}",
                signal_delta(previous, current, "steps", "steps"),
                delta_color="off",
                help="Synthetic cumulative step count for this replayed day.",
            )
            second_row[1].metric(
                "Sleep duration",
                f"{twin_state.sleep_hours:.1f} h",
                signal_delta(previous, current, "sleep_hours", "h", precision=1),
                delta_color="off",
                help="Synthetic sleep-duration value bundled with this profile's readings.",
            )
            second_row[2].metric(
                "BMI · baseline EHR",
                f"{twin_state.bmi:.1f}",
                help="Illustrative score bands: <25 = 0, 25-29.9 = 6, >=30 kg/m² = 12 points.",
            )
            second_row[3].metric(
                "HbA1c · baseline EHR",
                f"{twin_state.hba1c_pct:.1f}%",
                help="Illustrative score bands: <5.7 = 0, 5.7-5.9 = 12, 6.0-6.4 = 18, >=6.5% = 25 points.",
            )
            st.caption(f"Current sample: {twin_state.timestamp} · {assessment.category}")

        report = {
            "patient_id": profile["patient_id"],
            "display_name": profile["display_name"],
            "age_years": profile["age"],
            "sex": profile["sex"],
            "baseline_bmi_kg_m2": profile["bmi"],
            "baseline_hba1c_percent": profile["hba1c_pct"],
            "baseline_total_cholesterol_mg_dl": profile["total_cholesterol_mg_dl"],
            "baseline_systolic_bp_mmhg": profile["systolic_bp_mmhg"],
            "family_history_t2d": profile["family_history_t2d"],
            "medical_history": profile["medical_history"],
            "latest_sample_timestamp": current["timestamp"],
            "latest_glucose_mg_dl": current["glucose_mg_dl"],
            "latest_heart_rate_bpm": current["heart_rate_bpm"],
            "latest_steps": current["steps"],
            "latest_sleep_hours": current["sleep_hours"],
            "latest_systolic_bp_mmhg": current["systolic_bp_mmhg"],
            "risk_indicator_score": assessment.score,
            "risk_indicator_category": assessment.category,
            "data_provenance": "Synthetic demonstration data; not for clinical use",
        }
        export_column, export_note = st.columns([1, 2], vertical_alignment="center")
        with export_column:
            st.download_button(
                "Export Patient Summary",
                data=pd.DataFrame([report]).to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{profile['patient_id']}_synthetic_twin_summary.csv",
                mime="text/csv",
                key=f"export-summary-{selected_id}",
                help="Download the selected synthetic EHR profile, current synthetic reading, and rule-based indicator.",
                width="stretch",
            )
        with export_note:
            st.caption("CSV contains synthetic profile data and the currently displayed replay reading.")

        with st.container(border=True):
            st.subheader("Why is the score changing?")
            contributors = get_risk_contributors(assessment, limit=5)
            left, right = st.columns([1, 2])
            with left:
                st.markdown("**Primary contributors**")
                for index, (name, points) in enumerate(contributors, start=1):
                    st.write(f"{index}. {name} · {points} simulation points")
            with right:
                contributor_names = {name for name, _ in contributors}
                explanations = []
                if contributor_names.intersection(
                    {"Baseline HbA1c", "BMI", "Baseline total cholesterol", "Family history"}
                ):
                    explanations.append("Baseline metabolic markers contribute to the indicator.")
                if contributor_names.intersection(
                    {"Daily steps", "Sleep duration", "Current systolic BP"}
                ):
                    explanations.append("Current activity, sleep, or blood-pressure points also contribute.")
                st.info(" ".join(explanations) or "No major positive rule-based contributors for this reading.")
                st.caption("Rule-based demonstration indicator · not a medical prediction.")

        if current["glucose_mg_dl"] >= 180:
            st.warning(
                "Simulated glucose reading is elevated (180 mg/dL or above). "
                "This demo alert is not a diagnosis; follow clinical guidance."
            )
        elif assessment.score >= 50:
            st.info("Composite indicator is elevated. Review the contributing factors with a clinician.")
        else:
            st.success("No demo threshold alert for this reading.")

        alert_feed = get_synthetic_alerts(observations, selected_id, limit=5)
        with st.expander(f"Recent synthetic alerts ({len(alert_feed)})", expanded=bool(alert_feed)):
            if alert_feed:
                for alert in reversed(alert_feed):
                    st.warning(alert)
            else:
                st.caption(
                    "No synthetic glucose-spike or systolic-pressure threshold events in this patient's bundled readings."
                )

        trend = pd.DataFrame(patient_rows)
        trend["timestamp"] = pd.to_datetime(trend["timestamp"])
        st.subheader("Glucose stream")
        glucose_chart = px.line(
            trend,
            x="timestamp",
            y="glucose_mg_dl",
            markers=True,
            labels={"timestamp": "Sample time", "glucose_mg_dl": "Glucose (mg/dL)"},
        )
        glucose_chart.add_hline(y=180, line_dash="dash", line_color="#d04a3a")
        st.plotly_chart(style_chart(glucose_chart), width="stretch")
        st.caption(
            f"Current sample: {current['timestamp']} · "
            f"Home systolic BP {current['systolic_bp_mmhg']:.0f} mmHg"
        )

    elif selected_view == "Trends":
        st.subheader("Synthetic longitudinal trends")
        trend_window = st.radio(
            "History window",
            options=["24 hours", "7 days", "30 days"],
            horizontal=True,
            key=f"trend-window-{selected_id}",
        )
        if trend_window == "24 hours":
            trend_rows = patient_rows
            st.caption("Bundled synthetic wearable readings · approximately 24 hours.")
        else:
            days = 7 if trend_window == "7 days" else 30
            trend_rows = generate_synthetic_history(observations, selected_id, days)
            st.caption(
                f"{trend_window} view is a deterministic synthetic extension of the bundled sample pattern, "
                "not additional measured patient history."
            )

        trend = pd.DataFrame(trend_rows)
        trend["timestamp"] = pd.to_datetime(trend["timestamp"])
        trend["risk_indicator"] = [
            assess_risk(profile, row).score for row in trend_rows
        ]
        chart_row = st.columns(2)
        with chart_row[0]:
            risk_chart = px.area(
                trend,
                x="timestamp",
                y="risk_indicator",
                markers=True,
                labels={"timestamp": "Sample time", "risk_indicator": "Indicator (0-100)"},
                title="Composite indicator trend",
            )
            risk_chart.update_yaxes(range=[0, 100])
            st.plotly_chart(style_chart(risk_chart), width="stretch")
        with chart_row[1]:
            glucose_trend = px.line(
                trend,
                x="timestamp",
                y="glucose_mg_dl",
                markers=True,
                labels={"timestamp": "Sample time", "glucose_mg_dl": "Glucose (mg/dL)"},
                title="Synthetic glucose history",
            )
            glucose_trend.add_hline(y=180, line_dash="dash", line_color="#c95743")
            st.plotly_chart(style_chart(glucose_trend), width="stretch")

        factors = pd.DataFrame(assessment.factors, columns=["Factor", "Points"])
        factors = factors[factors["Points"] > 0].sort_values("Points")
        st.subheader("Current score contributors")
        st.caption("Point contributions for the current selected wearable reading.")
        if factors.empty:
            st.caption("No points contributed for this reading.")
        else:
            factor_chart = px.bar(
                factors,
                x="Points",
                y="Factor",
                orientation="h",
                labels={"Points": "Simulation points", "Factor": ""},
            )
            st.plotly_chart(style_chart(factor_chart), width="stretch")

        with st.expander("Why each simulation factor receives points"):
            st.dataframe(
                pd.DataFrame(RISK_FACTOR_GUIDANCE, columns=["Factor", "Illustrative scoring rule"]),
                hide_index=True,
                width="stretch",
            )
            st.caption("These hand-authored thresholds are for demonstration only and are not clinically validated.")

    elif selected_view == "What-if":
        st.subheader("Digital Twin Scenario Builder")
        st.caption(
            f"Build a hypothetical rule-based scenario from current sample {current['timestamp']}. "
            "Activity frequency, diet, carbohydrates, and medication adherence are scenario assumptions, not stored patient data."
        )
        scenario_key = f"scenario-result-{selected_id}"
        with st.form(key=f"scenario-form-{selected_id}", border=True):
            steps_column, sleep_column = st.columns(2)
            with steps_column:
                scenario_steps = st.number_input(
                    "Scenario daily steps",
                    min_value=0,
                    max_value=30000,
                    value=int(current["steps"]),
                    step=500,
                    help="Hypothetical daily step target; no real activity is recorded.",
                    key=f"scenario-steps-{selected_id}",
                )
            with sleep_column:
                scenario_sleep = st.number_input(
                    "Scenario sleep duration (hours)",
                    min_value=0.0,
                    max_value=24.0,
                    value=float(current["sleep_hours"]),
                    step=0.5,
                    help="Hypothetical sleep setting used by this transparent simulation.",
                    key=f"scenario-sleep-{selected_id}",
                )

            activity_column, diet_column = st.columns(2)
            with activity_column:
                scenario_activity_days = st.slider(
                    "Active days per week",
                    min_value=0,
                    max_value=7,
                    value=2,
                    help="Two days/week is the demo baseline; this is not patient-reported data.",
                    key=f"scenario-days-{selected_id}",
                )
            with diet_column:
                scenario_diet = st.selectbox(
                    "Diet adherence scenario",
                    options=["Low", "Moderate", "High"],
                    index=1,
                    help="A simple hypothetical setting, not a nutritional assessment.",
                    key=f"scenario-diet-{selected_id}",
                )

            carbohydrate_column, medication_column = st.columns(2)
            with carbohydrate_column:
                carbohydrate_reduction = st.slider(
                    "Carbohydrate intake reduction (hypothetical)",
                    min_value=0,
                    max_value=50,
                    value=0,
                    step=5,
                    format="%d%%",
                    help="Scenario-only modifier; no carbohydrate intake is present in the EHR data.",
                    key=f"scenario-carbs-{selected_id}",
                )
            with medication_column:
                medication_adherence = st.selectbox(
                    "Medication adherence scenario",
                    options=["Not applicable", "Consistent", "Intermittent"],
                    help="Scenario-only assumption. Do not change medication use based on this demo.",
                    key=f"scenario-medication-{selected_id}",
                )

            run_scenario = st.form_submit_button(
                "Run Scenario",
                type="primary",
                width="stretch",
            )

        if run_scenario:
            st.session_state[scenario_key] = calculate_simulated_state(
                profile,
                current,
                steps=int(scenario_steps),
                sleep_hours=float(scenario_sleep),
                activity_days_per_week=scenario_activity_days,
                diet_adherence=scenario_diet,
                carbohydrate_reduction_percent=carbohydrate_reduction,
                medication_adherence=medication_adherence,
            )

        scenario_result = st.session_state.get(scenario_key)
        if scenario_result is None:
            st.info("Choose scenario inputs above, then select Run Scenario to compare twin states.")
        else:
            baseline_state = scenario_result.current_state
            future_state = scenario_result.simulated_state
            score_change = future_state.score - baseline_state.score
            st.markdown(
                '<div class="scenario-flow"><span>CURRENT TWIN</span><b>→</b>'
                '<span>RULE-BASED SCENARIO</span><b>→</b><span>SIMULATED FUTURE TWIN</span></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Scenario baseline sample: {baseline_state.timestamp}")
            current_column, future_column = st.columns(2)
            with current_column:
                with st.container(border=True):
                    st.subheader("Current Twin")
                    st.metric("Risk indicator", f"{baseline_state.score} / 100", baseline_state.category, delta_color="off")
                    metric_columns = st.columns(2)
                    metric_columns[0].metric("Steps", f"{baseline_state.steps:,}")
                    metric_columns[1].metric("Sleep", f"{baseline_state.sleep_hours:.1f} h")
                    metric_columns[0].metric("Active days/week", str(baseline_state.activity_days_per_week))
                    metric_columns[1].metric("Diet adherence", baseline_state.diet_adherence)
                    metric_columns[0].metric("Carbohydrate reduction", f"{baseline_state.carbohydrate_reduction_percent}%")
                    metric_columns[1].metric("Medication scenario", baseline_state.medication_adherence)
                    st.caption(f"Glucose {baseline_state.glucose_mg_dl:.0f} mg/dL · BP {baseline_state.systolic_bp_mmhg:.0f} mmHg · BMI {baseline_state.bmi:.1f} · HbA1c {baseline_state.hba1c_pct:.1f}%")
            with future_column:
                with st.container(border=True):
                    st.subheader("Simulated Twin")
                    st.metric(
                        "Risk indicator",
                        f"{future_state.score} / 100",
                        f"{score_change:+d} points from baseline",
                        delta_color="inverse",
                    )
                    metric_columns = st.columns(2)
                    metric_columns[0].metric("Steps", f"{future_state.steps:,}")
                    metric_columns[1].metric("Sleep", f"{future_state.sleep_hours:.1f} h")
                    metric_columns[0].metric("Active days/week", str(future_state.activity_days_per_week))
                    metric_columns[1].metric("Diet adherence", future_state.diet_adherence)
                    metric_columns[0].metric("Carbohydrate reduction", f"{future_state.carbohydrate_reduction_percent}%")
                    metric_columns[1].metric("Medication scenario", future_state.medication_adherence)
                    st.caption(f"Glucose unchanged at {future_state.glucose_mg_dl:.0f} mg/dL · BP unchanged at {future_state.systolic_bp_mmhg:.0f} mmHg · BMI {future_state.bmi:.1f} · HbA1c {future_state.hba1c_pct:.1f}%")

            st.subheader("Why did the twin change?")
            explanations = generate_scenario_explanations(scenario_result)
            for explanation in explanations:
                st.markdown(f"- {explanation}")

            comparison = pd.DataFrame(
                [
                    {"State": "Current Twin", "Rule-based indicator": baseline_state.score},
                    {"State": "Simulated Future Twin", "Rule-based indicator": future_state.score},
                ]
            )
            comparison_chart = px.bar(
                comparison,
                x="State",
                y="Rule-based indicator",
                color="State",
                color_discrete_sequence=["#2b7380", "#c95743"],
                title="Current vs simulated indicator",
            )
            comparison_chart.update_yaxes(range=[0, 100])
            st.plotly_chart(style_chart(comparison_chart), width="stretch")

            with st.expander("How the demonstration scenario points work"):
                st.markdown(
                    "- Steps and sleep reuse the existing risk-engine score bands.\n"
                    "- Activity frequency is neutral at 2 days/week and ranges from +2 to -5 points.\n"
                    "- Diet adherence is Low +2, Moderate 0, High -2 points.\n"
                    "- Hypothetical carbohydrate reduction subtracts 1 point per 15%, capped at 3 points.\n"
                    "- Medication adherence is Not applicable 0, Consistent -1, Intermittent +2 points.\n"
                    "- Glucose, blood pressure, BMI, and HbA1c are held unchanged; this is not a clinical prediction."
                )


live_dashboard()

st.markdown(
    """
    <aside class="safety-notice" role="note" aria-label="Clinical safety notice">
      <div class="safety-notice-label">Clinical safety · Demonstration only</div>
      <strong>Not for diagnosis, treatment, triage, or care decisions.</strong>
      <p>The score and alert thresholds are hand-authored, unvalidated simulation rules, not an individual probability or clinical advice. All patient and sensor records are synthetic.</p>
    </aside>
    """,
    unsafe_allow_html=True,
)

st.markdown(
        """
        <footer class="twin-footer" aria-label="Prototype information">
            <div>
                <div class="twin-footer-brand">Twin Health · Digital Twin Challenge 2026</div>
                <div class="twin-footer-detail">Cardiometabolic monitoring proof of concept</div>
            </div>
            <div class="twin-footer-status">SYNTHETIC DATA · NO DEVICE CONNECTION</div>
            <div class="twin-footer-note">For demonstration only. The score is not a validated clinical prediction and must not guide care decisions.</div>
        </footer>
        """,
        unsafe_allow_html=True,
)