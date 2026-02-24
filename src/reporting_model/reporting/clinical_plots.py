"""
Clinical Plots - Matplotlib-based medical trend visualizations
Generates vitals trend charts, event timelines, and severity distributions
for embedding in PDF reports.
"""

import io
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for PDF embedding
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ──────────────────────────────────────────────────────────
# Color palette (matches PDF report theme)
# ──────────────────────────────────────────────────────────
COLORS = {
    "primary": "#1a365d",
    "secondary": "#2c5282",
    "accent": "#3182ce",
    "success": "#38a169",
    "warning": "#d69e2e",
    "danger": "#e53e3e",
    "critical": "#c53030",
    "light_bg": "#f7fafc",
    "grid": "#e2e8f0",
    "text": "#2d3748",
    "muted": "#718096",
    # Vitals-specific
    "spo2": "#3182ce",
    "heart_rate": "#e53e3e",
    "temperature": "#d69e2e",
    "bp_systolic": "#805ad5",
    "bp_diastolic": "#9f7aea",
}

# Clinical thresholds
THRESHOLDS = {
    "spo2": {"critical_low": 88, "low": 92, "normal": 95},
    "heart_rate": {"low": 50, "normal_low": 60, "normal_high": 100, "high": 120},
    "temperature": {"low": 35.5, "normal_low": 36.1, "normal_high": 37.5, "high": 38.0, "fever": 38.5},
}


def _apply_clinical_style(fig, ax, title: str) -> None:
    """Apply consistent clinical style to a plot."""
    ax.set_title(title, fontsize=13, fontweight='bold', color=COLORS["primary"], pad=12)
    ax.grid(True, alpha=0.3, color=COLORS["grid"], linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS["muted"])
    ax.spines['bottom'].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["text"], labelsize=8)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')


def _save_plot_to_bytes(fig) -> bytes:
    """Save matplotlib figure to bytes buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _save_plot_to_file(fig, filepath: str) -> str:
    """Save matplotlib figure to a file."""
    fig.savefig(filepath, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    return filepath


# ──────────────────────────────────────────────────────────
# 1. SpO2 Trend Plot
# ──────────────────────────────────────────────────────────
def plot_spo2_trend(
    vitals_timeline: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> Optional[bytes]:
    """
    Generate SpO2 trend plot with clinical thresholds.
    
    Args:
        vitals_timeline: List of vitals readings with 'timestamp' and 'spo2'
        output_path: If provided, save to file instead of returning bytes
        
    Returns:
        PNG bytes or None (if saved to file)
    """
    if not MATPLOTLIB_AVAILABLE or not vitals_timeline:
        return None

    times = []
    values = []
    for v in vitals_timeline:
        ts = v.get("timestamp")
        spo2 = v.get("spo2")
        if ts and spo2 is not None:
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    continue
            times.append(ts)
            values.append(spo2)

    if not times:
        return None

    fig, ax = plt.subplots(figsize=(8, 3))
    _apply_clinical_style(fig, ax, "Tendance SpO2 - Surveillance Nocturne")

    # Threshold zones
    ax.axhspan(0, THRESHOLDS["spo2"]["critical_low"], alpha=0.08, color=COLORS["danger"], label="Zone critique (<88%)")
    ax.axhspan(THRESHOLDS["spo2"]["critical_low"], THRESHOLDS["spo2"]["low"], alpha=0.06, color=COLORS["warning"], label="Zone basse (88-92%)")
    ax.axhspan(THRESHOLDS["spo2"]["low"], 100, alpha=0.03, color=COLORS["success"])

    # Threshold lines
    ax.axhline(y=THRESHOLDS["spo2"]["critical_low"], color=COLORS["danger"], linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axhline(y=THRESHOLDS["spo2"]["low"], color=COLORS["warning"], linestyle='--', alpha=0.5, linewidth=0.8)

    # Data
    ax.plot(times, values, color=COLORS["spo2"], linewidth=1.5, marker='o', markersize=3, alpha=0.9)

    # Mark anomalies
    for t, v in zip(times, values):
        if v < THRESHOLDS["spo2"]["critical_low"]:
            ax.plot(t, v, 'o', color=COLORS["danger"], markersize=7, zorder=5)
        elif v < THRESHOLDS["spo2"]["low"]:
            ax.plot(t, v, 'o', color=COLORS["warning"], markersize=6, zorder=5)

    ax.set_ylabel("SpO2 (%)", fontsize=9, color=COLORS["text"])
    ax.set_xlabel("Heure", fontsize=9, color=COLORS["text"])
    ax.set_ylim(max(min(values) - 5, 70), 102)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend(fontsize=7, loc='lower left', framealpha=0.8)
    fig.tight_layout()

    if output_path:
        return _save_plot_to_file(fig, output_path)
    return _save_plot_to_bytes(fig)


# ──────────────────────────────────────────────────────────
# 2. Heart Rate Trend Plot
# ──────────────────────────────────────────────────────────
def plot_heart_rate_trend(
    vitals_timeline: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> Optional[bytes]:
    """Generate heart rate trend plot with bradycardia/tachycardia zones."""
    if not MATPLOTLIB_AVAILABLE or not vitals_timeline:
        return None

    times = []
    values = []
    for v in vitals_timeline:
        ts = v.get("timestamp")
        hr = v.get("heart_rate")
        if ts and hr is not None:
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    continue
            times.append(ts)
            values.append(hr)

    if not times:
        return None

    fig, ax = plt.subplots(figsize=(8, 3))
    _apply_clinical_style(fig, ax, "Frequence Cardiaque - Surveillance Nocturne")

    # Threshold zones
    ax.axhspan(0, THRESHOLDS["heart_rate"]["normal_low"], alpha=0.06, color=COLORS["accent"], label="Bradycardie (<60 bpm)")
    ax.axhspan(THRESHOLDS["heart_rate"]["normal_high"], 200, alpha=0.06, color=COLORS["danger"], label="Tachycardie (>100 bpm)")

    ax.axhline(y=THRESHOLDS["heart_rate"]["normal_low"], color=COLORS["accent"], linestyle='--', alpha=0.4, linewidth=0.8)
    ax.axhline(y=THRESHOLDS["heart_rate"]["normal_high"], color=COLORS["danger"], linestyle='--', alpha=0.4, linewidth=0.8)

    # Data
    ax.plot(times, values, color=COLORS["heart_rate"], linewidth=1.5, marker='o', markersize=3, alpha=0.9)

    # Mark anomalies
    for t, v in zip(times, values):
        if v > THRESHOLDS["heart_rate"]["high"]:
            ax.plot(t, v, 'o', color=COLORS["critical"], markersize=7, zorder=5)
        elif v > THRESHOLDS["heart_rate"]["normal_high"]:
            ax.plot(t, v, 'o', color=COLORS["warning"], markersize=6, zorder=5)
        elif v < THRESHOLDS["heart_rate"]["low"]:
            ax.plot(t, v, 'o', color=COLORS["accent"], markersize=6, zorder=5)

    ax.set_ylabel("FC (bpm)", fontsize=9, color=COLORS["text"])
    ax.set_xlabel("Heure", fontsize=9, color=COLORS["text"])
    ax.set_ylim(max(min(values) - 10, 30), max(max(values) + 10, 130))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    fig.tight_layout()

    if output_path:
        return _save_plot_to_file(fig, output_path)
    return _save_plot_to_bytes(fig)


# ──────────────────────────────────────────────────────────
# 3. Temperature Trend Plot
# ──────────────────────────────────────────────────────────
def plot_temperature_trend(
    vitals_timeline: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> Optional[bytes]:
    """Generate temperature trend plot with fever thresholds."""
    if not MATPLOTLIB_AVAILABLE or not vitals_timeline:
        return None

    times = []
    values = []
    for v in vitals_timeline:
        ts = v.get("timestamp")
        temp = v.get("temperature")
        if ts and temp is not None:
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    continue
            times.append(ts)
            values.append(temp)

    if not times:
        return None

    fig, ax = plt.subplots(figsize=(8, 3))
    _apply_clinical_style(fig, ax, "Temperature - Surveillance Nocturne")

    # Threshold zones
    ax.axhspan(THRESHOLDS["temperature"]["high"], 42, alpha=0.06, color=COLORS["danger"], label="Fievre (>38.0)")
    ax.axhspan(THRESHOLDS["temperature"]["normal_high"], THRESHOLDS["temperature"]["high"], alpha=0.04, color=COLORS["warning"], label="Subfebril (37.5-38.0)")

    ax.axhline(y=THRESHOLDS["temperature"]["normal_high"], color=COLORS["warning"], linestyle='--', alpha=0.4, linewidth=0.8)
    ax.axhline(y=THRESHOLDS["temperature"]["high"], color=COLORS["danger"], linestyle='--', alpha=0.4, linewidth=0.8)

    # Data
    ax.plot(times, values, color=COLORS["temperature"], linewidth=1.5, marker='s', markersize=3, alpha=0.9)

    for t, v in zip(times, values):
        if v >= THRESHOLDS["temperature"]["fever"]:
            ax.plot(t, v, 's', color=COLORS["critical"], markersize=7, zorder=5)
        elif v >= THRESHOLDS["temperature"]["high"]:
            ax.plot(t, v, 's', color=COLORS["warning"], markersize=6, zorder=5)

    ax.set_ylabel("T (C)", fontsize=9, color=COLORS["text"])
    ax.set_xlabel("Heure", fontsize=9, color=COLORS["text"])
    ax.set_ylim(max(min(values) - 0.5, 34), max(max(values) + 0.5, 39))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    fig.tight_layout()

    if output_path:
        return _save_plot_to_file(fig, output_path)
    return _save_plot_to_bytes(fig)


# ──────────────────────────────────────────────────────────
# 4. Combined Vitals Dashboard
# ──────────────────────────────────────────────────────────
def plot_vitals_dashboard(
    vitals_timeline: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> Optional[bytes]:
    """
    Generate a combined 3-panel vitals dashboard:
    - SpO2, Heart Rate, Temperature on vertically stacked subplots
    """
    if not MATPLOTLIB_AVAILABLE or not vitals_timeline:
        return None

    times, spo2_vals, hr_vals, temp_vals = [], [], [], []
    for v in vitals_timeline:
        ts = v.get("timestamp")
        if ts and isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                continue
        if ts is None:
            continue
        times.append(ts)
        spo2_vals.append(v.get("spo2"))
        hr_vals.append(v.get("heart_rate"))
        temp_vals.append(v.get("temperature"))

    if not times:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    fig.suptitle("Tableau de Bord - Constantes Nocturnes", fontsize=14,
                 fontweight='bold', color=COLORS["primary"], y=0.98)

    # SpO2
    ax = axes[0]
    ax.set_facecolor('white')
    ax.axhspan(0, 88, alpha=0.06, color=COLORS["danger"])
    ax.axhspan(88, 92, alpha=0.04, color=COLORS["warning"])
    ax.axhline(y=92, color=COLORS["warning"], linestyle='--', alpha=0.4, linewidth=0.7)
    valid = [(t, v) for t, v in zip(times, spo2_vals) if v is not None]
    if valid:
        t_v, v_v = zip(*valid)
        ax.plot(t_v, v_v, color=COLORS["spo2"], linewidth=1.3, marker='o', markersize=2.5)
        for t, v in valid:
            if v < 88:
                ax.plot(t, v, 'o', color=COLORS["danger"], markersize=6, zorder=5)
    ax.set_ylabel("SpO2 (%)", fontsize=9, color=COLORS["text"])
    ax.set_ylim(max(min(v for _, v in valid) - 5, 70) if valid else 80, 102)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Heart Rate
    ax = axes[1]
    ax.set_facecolor('white')
    ax.axhspan(0, 60, alpha=0.04, color=COLORS["accent"])
    ax.axhspan(100, 200, alpha=0.04, color=COLORS["danger"])
    ax.axhline(y=60, color=COLORS["accent"], linestyle='--', alpha=0.3, linewidth=0.7)
    ax.axhline(y=100, color=COLORS["danger"], linestyle='--', alpha=0.3, linewidth=0.7)
    valid = [(t, v) for t, v in zip(times, hr_vals) if v is not None]
    if valid:
        t_v, v_v = zip(*valid)
        ax.plot(t_v, v_v, color=COLORS["heart_rate"], linewidth=1.3, marker='o', markersize=2.5)
    ax.set_ylabel("FC (bpm)", fontsize=9, color=COLORS["text"])
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Temperature
    ax = axes[2]
    ax.set_facecolor('white')
    ax.axhspan(37.5, 42, alpha=0.04, color=COLORS["warning"])
    ax.axhspan(38.0, 42, alpha=0.04, color=COLORS["danger"])
    ax.axhline(y=37.5, color=COLORS["warning"], linestyle='--', alpha=0.3, linewidth=0.7)
    ax.axhline(y=38.0, color=COLORS["danger"], linestyle='--', alpha=0.3, linewidth=0.7)
    valid = [(t, v) for t, v in zip(times, temp_vals) if v is not None]
    if valid:
        t_v, v_v = zip(*valid)
        ax.plot(t_v, v_v, color=COLORS["temperature"], linewidth=1.3, marker='s', markersize=2.5)
    ax.set_ylabel("T (C)", fontsize=9, color=COLORS["text"])
    ax.set_xlabel("Heure", fontsize=9, color=COLORS["text"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        return _save_plot_to_file(fig, output_path)
    return _save_plot_to_bytes(fig)


# ──────────────────────────────────────────────────────────
# 5. Events Timeline Bar Chart
# ──────────────────────────────────────────────────────────
def plot_events_timeline(
    events: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> Optional[bytes]:
    """
    Generate an events timeline showing event types, severity, and time.
    """
    if not MATPLOTLIB_AVAILABLE or not events:
        return None

    # Parse events
    parsed = []
    for e in events:
        ts = e.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                continue
        etype = e.get("type", "unknown")
        level = e.get("level", e.get("severity", "low"))
        parsed.append({"time": ts, "type": etype, "level": level})

    if not parsed:
        return None

    parsed.sort(key=lambda x: x["time"])

    level_colors = {
        "critical": COLORS["critical"],
        "high": COLORS["danger"],
        "medium": COLORS["warning"],
        "low": COLORS["success"],
    }

    fig, ax = plt.subplots(figsize=(8, max(2.5, len(parsed) * 0.5)))
    _apply_clinical_style(fig, ax, "Chronologie des Evenements Nocturnes")

    y_positions = range(len(parsed))
    bar_colors = [level_colors.get(p["level"], COLORS["muted"]) for p in parsed]
    labels = [f"{p['type']}" for p in parsed]
    time_labels = [p["time"].strftime("%H:%M") if p["time"] else "?" for p in parsed]

    bars = ax.barh(y_positions, [1] * len(parsed), color=bar_colors, height=0.6, alpha=0.85)

    for i, (bar, label, tl) in enumerate(zip(bars, labels, time_labels)):
        ax.text(0.05, i, f"  {tl}  |  {label}", va='center', ha='left',
                fontsize=9, color='white', fontweight='bold')

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["critical"], label='Critique'),
        Patch(facecolor=COLORS["danger"], label='Eleve'),
        Patch(facecolor=COLORS["warning"], label='Modere'),
        Patch(facecolor=COLORS["success"], label='Faible'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='lower right', framealpha=0.8)
    fig.tight_layout()

    if output_path:
        return _save_plot_to_file(fig, output_path)
    return _save_plot_to_bytes(fig)


# ──────────────────────────────────────────────────────────
# 6. Severity Distribution Pie Chart
# ──────────────────────────────────────────────────────────
def plot_severity_distribution(
    events: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> Optional[bytes]:
    """Generate a pie chart showing event severity distribution."""
    if not MATPLOTLIB_AVAILABLE or not events:
        return None

    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for e in events:
        level = e.get("level", e.get("severity", "low"))
        if level in counts:
            counts[level] += 1
        else:
            counts["low"] += 1

    # Filter zero counts
    labels = []
    sizes = []
    pie_colors = []
    level_labels = {"critical": "Critique", "high": "Eleve", "medium": "Modere", "low": "Faible"}
    level_clrs = {"critical": COLORS["critical"], "high": COLORS["danger"],
                  "medium": COLORS["warning"], "low": COLORS["success"]}

    for k in ["critical", "high", "medium", "low"]:
        if counts[k] > 0:
            labels.append(f"{level_labels[k]} ({counts[k]})")
            sizes.append(counts[k])
            pie_colors.append(level_clrs[k])

    if not sizes:
        return None

    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=pie_colors,
        autopct='%1.0f%%', startangle=90, pctdistance=0.75,
        textprops={'fontsize': 8, 'color': COLORS["text"]}
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight('bold')

    ax.set_title("Distribution des Alertes", fontsize=11,
                 fontweight='bold', color=COLORS["primary"], pad=10)
    fig.tight_layout()

    if output_path:
        return _save_plot_to_file(fig, output_path)
    return _save_plot_to_bytes(fig)


# ──────────────────────────────────────────────────────────
# 7. Generate All Night Report Plots
# ──────────────────────────────────────────────────────────
def generate_night_report_plots(
    vitals_timeline: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    output_dir: str = "./data/reports/plots",
    patient_id: str = "unknown"
) -> Dict[str, str]:
    """
    Generate all plots for a night surveillance report.
    
    Args:
        vitals_timeline: List of vitals readings
        events: List of clinical events
        output_dir: Directory to save plot images
        patient_id: Patient ID for filenames
    
    Returns:
        Dict mapping plot name to file path
    """
    if not MATPLOTLIB_AVAILABLE:
        return {}

    plots_dir = Path(output_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{patient_id}_{datetime.now().strftime('%Y%m%d')}"
    generated = {}

    # 1. Vitals dashboard (combined)
    path = str(plots_dir / f"{prefix}_vitals_dashboard.png")
    result = plot_vitals_dashboard(vitals_timeline, output_path=path)
    if result:
        generated["vitals_dashboard"] = path

    # 2. SpO2 trend
    path = str(plots_dir / f"{prefix}_spo2_trend.png")
    result = plot_spo2_trend(vitals_timeline, output_path=path)
    if result:
        generated["spo2_trend"] = path

    # 3. Heart rate trend
    path = str(plots_dir / f"{prefix}_heart_rate_trend.png")
    result = plot_heart_rate_trend(vitals_timeline, output_path=path)
    if result:
        generated["heart_rate_trend"] = path

    # 4. Temperature trend
    path = str(plots_dir / f"{prefix}_temperature_trend.png")
    result = plot_temperature_trend(vitals_timeline, output_path=path)
    if result:
        generated["temperature_trend"] = path

    # 5. Events timeline
    path = str(plots_dir / f"{prefix}_events_timeline.png")
    result = plot_events_timeline(events, output_path=path)
    if result:
        generated["events_timeline"] = path

    # 6. Severity distribution
    path = str(plots_dir / f"{prefix}_severity_distribution.png")
    result = plot_severity_distribution(events, output_path=path)
    if result:
        generated["severity_distribution"] = path

    return generated
