#!/usr/bin/env python3
"""
Visualize Empatica raw sensor data stored in Avro files.

The script produces overview figures for each Avro file, showing:
  • Accelerometer (x, y, z) in g
  • Electrodermal Activity (EDA) in µS
  • Blood Volume Pulse (BVP)
  • Temperature in °C
  • Optional systolic peaks overlaid on the BVP panel

Usage example
-------------
python extract_raw_avro_data.py \
    --input-dir /Users/awwu/Downloads/WAZEER1107-3YK9V1L15N/wazeer1107/v6 \
    --files 1-1-WAZEER1107_1762547001.avro 1-1-WAZEER1107_1762547050.avro \
    --output-dir /tmp/wazeer1107_plots

Requirements: pip install pandas matplotlib avro-python3 python-dateutil plotly
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

try:
    import pandas as pd
    from avro.datafile import DataFileReader
    from avro.io import DatumReader
    import matplotlib.pyplot as plt
except ImportError as exc:
    print(
        "Error: Missing required packages. "
        "Please install them first:\n"
        "    pip install pandas matplotlib avro-python3 python-dateutil",
        file=sys.stderr,
    )
    raise


def import_plotly():
    """
    Import Plotly modules on demand to keep the dependency optional.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for interactive output. "
            "Install it with: pip install plotly"
        ) from exc

    return go, make_subplots


def read_avro_file(avro_file_path: str) -> Optional[Dict]:
    """
    Read an Empatica Avro file and return a dictionary of raw sensor streams.
    """
    try:
        with DataFileReader(open(avro_file_path, "rb"), DatumReader()) as reader:
            data = next(reader)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"⚠️  Skipping {avro_file_path!r}: {exc}")
        return None

    avro_version = (
        data["schemaVersion"]["major"],
        data["schemaVersion"]["minor"],
        data["schemaVersion"]["patch"],
    )

    result = {"avro_version": avro_version, "raw_data": {}}
    raw_data = data.get("rawData", {})

    # EDA -----------------------------------------------------------------
    if "eda" in raw_data:
        eda = raw_data["eda"]
        timestamps = [
            round(eda["timestampStart"] + i * (1e6 / eda["samplingFrequency"]))
            for i in range(len(eda["values"]))
        ]
        result["raw_data"]["eda"] = {"timestamps": timestamps, "values": eda["values"]}

    # Temperature ---------------------------------------------------------
    if "temperature" in raw_data:
        tmp = raw_data["temperature"]
        timestamps = [
            round(tmp["timestampStart"] + i * (1e6 / tmp["samplingFrequency"]))
            for i in range(len(tmp["values"]))
        ]
        result["raw_data"]["temperature"] = {
            "timestamps": timestamps,
            "values": tmp["values"],
        }

    # Accelerometer -------------------------------------------------------
    if "accelerometer" in raw_data:
        acc = raw_data["accelerometer"]
        timestamps = [
            round(acc["timestampStart"] + i * (1e6 / acc["samplingFrequency"]))
            for i in range(len(acc["x"]))
        ]

        # Convert ADC counts to g
        if avro_version < (6, 5, 0):
            delta_physical = acc["imuParams"]["physicalMax"] - acc["imuParams"]["physicalMin"]
            delta_digital = acc["imuParams"]["digitalMax"] - acc["imuParams"]["digitalMin"]
            x_g = [val * delta_physical / delta_digital for val in acc["x"]]
            y_g = [val * delta_physical / delta_digital for val in acc["y"]]
            z_g = [val * delta_physical / delta_digital for val in acc["z"]]
        else:
            conversion_factor = acc["imuParams"]["conversionFactor"]
            x_g = [val * conversion_factor for val in acc["x"]]
            y_g = [val * conversion_factor for val in acc["y"]]
            z_g = [val * conversion_factor for val in acc["z"]]

        result["raw_data"]["accelerometer"] = {
            "timestamps": timestamps,
            "x": x_g,
            "y": y_g,
            "z": z_g,
        }

    # Blood Volume Pulse --------------------------------------------------
    if "bvp" in raw_data:
        bvp = raw_data["bvp"]
        timestamps = [
            round(bvp["timestampStart"] + i * (1e6 / bvp["samplingFrequency"]))
            for i in range(len(bvp["values"]))
        ]
        result["raw_data"]["bvp"] = {"timestamps": timestamps, "values": bvp["values"]}

    # Systolic Peaks ------------------------------------------------------
    if "systolicPeaks" in raw_data:
        sps = raw_data["systolicPeaks"]
        result["raw_data"]["systolic_peaks"] = {"timestamps": sps["peaksTimeNanos"]}
    print(avro_file_path, raw_data.keys())
    if "temperature" in raw_data:
        print(len(raw_data["temperature"]["values"]))

    return result


def to_datetime(
    timestamps: List[int],
    unit: str,
    target_tz: Union[str, datetime.tzinfo],
) -> pd.Series:
    """
    Convert integer timestamps to timezone-aware pandas Series.
    """
    if not timestamps:
        return pd.Series(dtype="datetime64[ns, UTC]")

    index = pd.to_datetime(pd.Series(timestamps), unit=unit, utc=True)
    try:
        return index.dt.tz_convert(target_tz)
    except Exception:
        print(f"⚠️  Unknown timezone {target_tz!r}, falling back to UTC.")
        return index


def build_sensor_frames(
    raw_data: Dict, target_tz: Union[str, datetime.tzinfo]
) -> Dict[str, pd.DataFrame]:
    """
    Build pandas DataFrames for each sensor stream.
    """
    frames: Dict[str, pd.DataFrame] = {}

    # Accelerometer -------------------------------------------------------
    acc = raw_data.get("accelerometer")
    if acc and acc.get("timestamps"):
        df_acc = pd.DataFrame(
            {
                "timestamp": to_datetime(acc["timestamps"], "us", target_tz),
                "acc_x_g": acc["x"],
                "acc_y_g": acc["y"],
                "acc_z_g": acc["z"],
            }
        )
        frames["accelerometer"] = df_acc

    # EDA -----------------------------------------------------------------
    eda = raw_data.get("eda")
    if eda and eda.get("timestamps"):
        df_eda = pd.DataFrame(
            {
                "timestamp": to_datetime(eda["timestamps"], "us", target_tz),
                "eda_uS": eda["values"],
            }
        )
        frames["eda"] = df_eda

    # Temperature ---------------------------------------------------------
    tmp = raw_data.get("temperature")
    if tmp and tmp.get("timestamps"):
        df_tmp = pd.DataFrame(
            {
                "timestamp": to_datetime(tmp["timestamps"], "us", target_tz),
                "temperature_c": tmp["values"],
            }
        )
        frames["temperature"] = df_tmp

    # BVP -----------------------------------------------------------------
    bvp = raw_data.get("bvp")
    if bvp and bvp.get("timestamps"):
        df_bvp = pd.DataFrame(
            {
                "timestamp": to_datetime(bvp["timestamps"], "us", target_tz),
                "bvp": bvp["values"],
            }
        )
        frames["bvp"] = df_bvp

    # Systolic Peaks ------------------------------------------------------
    peaks = raw_data.get("systolic_peaks")
    if peaks and peaks.get("timestamps"):
        frames["systolic_peaks"] = pd.DataFrame(
            {
                "timestamp": to_datetime(peaks["timestamps"], "ns", target_tz),
            }
        )

    return frames


def plot_sensor_panels(
    sensor_frames: Dict[str, pd.DataFrame],
    avro_name: str,
    output_path: Optional[str],
    show: bool,
    dpi: int,
) -> None:
    """
    Generate a multi-panel figure for the available sensor streams.
    """
    panels: List[str] = []
    for key in ("accelerometer", "eda", "temperature"):
        if key in sensor_frames and not sensor_frames[key].empty:
            panels.append(key)

    if not panels:
        print(f"⚠️  No plottable streams in {avro_name}")
        return

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(14, 3.2 * len(panels)),
        sharex=True,
        constrained_layout=True,
    )

    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        df = sensor_frames[panel]

        if panel == "accelerometer":
            ax.plot(df["timestamp"], df["acc_x_g"], label="x", linewidth=0.8)
            ax.plot(df["timestamp"], df["acc_y_g"], label="y", linewidth=0.8)
            ax.plot(df["timestamp"], df["acc_z_g"], label="z", linewidth=0.8)
            ax.set_ylabel("Acceleration [g]")
            ax.legend(loc="upper right", fontsize=8)

        elif panel == "eda":
            ax.plot(df["timestamp"], df["eda_uS"], color="#6C71C4", linewidth=0.8)
            ax.set_ylabel("EDA [µS]")

        elif panel == "temperature":
            ax.plot(df["timestamp"], df["temperature_c"], color="#268BD2", linewidth=0.8)
            ax.set_ylabel("Temp [°C]")

        ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)

    tz_info = sensor_frames[panels[-1]]["timestamp"].dt.tz
    if tz_info is None:
        tz_str = "UTC"
    else:
        tz_name = getattr(tz_info, "key", None) or tz_info.tzname(None)
        tz_str = tz_name or str(tz_info)
    axes[-1].set_xlabel(f"Time ({tz_str})")
    fig.suptitle(avro_name, fontsize=14, fontweight="bold")

    if output_path:
        fig.savefig(output_path, dpi=dpi)
        print(f"✅  Saved {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def get_time_bounds(sensor_frames: Dict[str, pd.DataFrame]) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Determine the global min/max timestamps across sensor frames.
    """
    min_ts: Optional[pd.Timestamp] = None
    max_ts: Optional[pd.Timestamp] = None
    for df in sensor_frames.values():
        if df.empty or "timestamp" not in df.columns:
            continue
        current_min = df["timestamp"].min()
        current_max = df["timestamp"].max()
        if min_ts is None or (current_min is not None and current_min < min_ts):
            min_ts = current_min
        if max_ts is None or (current_max is not None and current_max > max_ts):
            max_ts = current_max
    return min_ts, max_ts


def format_datetime_local(
    ts: Optional[pd.Timestamp],
    tz_info: Optional[Union[str, datetime.tzinfo]],
) -> str:
    """
    Convert a timezone-aware timestamp to an HTML datetime-local compatible string.
    """
    if ts is None or pd.isna(ts):
        return ""
    ts_local = ts
    try:
        if tz_info and ts.tzinfo:
            ts_local = ts.tz_convert(tz_info)
    except Exception:
        ts_local = ts
    if ts_local.tzinfo is not None:
        ts_local = ts_local.tz_localize(None)
    return ts_local.strftime("%Y-%m-%dT%H:%M:%S")


def build_interactive_html(
    figure_html: str,
    tz_label: str,
    start_value: str,
    end_value: str,
) -> str:
    """
    Wrap Plotly's HTML with custom controls for start/end range inputs.
    """
    controls = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Sensor Overview</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0;
    padding: 1rem;
    background: #f7f7f7;
}}
.controls {{
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1rem;
    align-items: flex-end;
}}
.controls label {{
    display: flex;
    flex-direction: column;
    font-size: 0.85rem;
    color: #333;
}}
.controls input {{
    padding: 0.3rem 0.4rem;
    font-size: 0.9rem;
}}
.controls button {{
    padding: 0.4rem 0.8rem;
    font-size: 0.9rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}}
#apply-range {{ background: #268BD2; color: white; }}
#reset-range {{ background: #EEE; }}
.timezone-note {{
    font-size: 0.8rem;
    color: #666;
}}
</style>
</head>
<body>
<div class="controls">
    <label>
        Start ({tz_label})
        <input type="datetime-local" id="start-input" value="{start_value}">
    </label>
    <label>
        End ({tz_label})
        <input type="datetime-local" id="end-input" value="{end_value}">
    </label>
    <button id="apply-range">Apply Range</button>
    <button id="reset-range">Reset</button>
    <div class="timezone-note">Times interpret in {tz_label}. Leave empty for auto range.</div>
</div>
{figure_html}
<script>
(function() {{
    const startInput = document.getElementById("start-input");
    const endInput = document.getElementById("end-input");
    const plotDiv = document.getElementsByClassName("plotly-graph-div")[0];

    function applyRange() {{
        if (!plotDiv) return;
        const update = {{}};
        if (startInput.value) {{
            update["xaxis.range[0]"] = new Date(startInput.value).toISOString();
        }}
        if (endInput.value) {{
            update["xaxis.range[1]"] = new Date(endInput.value).toISOString();
        }}
        Plotly.relayout(plotDiv, update);
    }}

    function resetRange() {{
        if (!plotDiv) return;
        startInput.value = "";
        endInput.value = "";
        Plotly.relayout(plotDiv, {{"xaxis.autorange": true}});
    }}

    document.getElementById("apply-range").addEventListener("click", applyRange);
    document.getElementById("reset-range").addEventListener("click", resetRange);
}})();
</script>
</body>
</html>
"""
    return controls


def plot_interactive_sensor_panels(
    sensor_frames: Dict[str, pd.DataFrame],
    avro_name: str,
    output_path: Optional[str],
) -> None:
    """
    Generate an interactive Plotly HTML figure for the available sensor streams.
    """
    panels: List[str] = []
    for key in ("accelerometer", "eda", "temperature"):
        if key in sensor_frames and not sensor_frames[key].empty:
            panels.append(key)

    if not panels:
        print(f"⚠️  No plottable streams in {avro_name} (interactive skipped)")
        return

    go, make_subplots = import_plotly()

    fig = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[panel.capitalize() for panel in panels],
    )

    for idx, panel in enumerate(panels, start=1):
        df = sensor_frames[panel]

        if panel == "accelerometer":
            for axis, color in zip(
                ("acc_x_g", "acc_y_g", "acc_z_g"),
                ("#268BD2", "#859900", "#DC322F"),
            ):
                component = axis.split("_")[1].upper()
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df[axis],
                        mode="lines",
                        name=f"Accel {component}",
                        hovertemplate=(
                            "Time=%{x|%Y-%m-%d %H:%M:%S}<br>"
                            f"{axis}=%{{y:.4f}}<extra></extra>"
                        ),
                        line=dict(color=color, width=1),
                    ),
                    row=idx,
                    col=1,
                )
            fig.update_yaxes(title_text="Acceleration [g]", row=idx, col=1)

        elif panel == "eda":
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["eda_uS"],
                    mode="lines",
                    name="EDA",
                    hovertemplate=(
                        "Time=%{x|%Y-%m-%d %H:%M:%S}<br>"
                        "EDA=%{y:.3f} µS<extra></extra>"
                    ),
                    line=dict(color="#6C71C4", width=1),
                ),
                row=idx,
                col=1,
            )
            fig.update_yaxes(title_text="EDA [µS]", row=idx, col=1)

        elif panel == "temperature":
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["temperature_c"],
                    mode="lines",
                    name="Temperature",
                    hovertemplate=(
                        "Time=%{x|%Y-%m-%d %H:%M:%S}<br>"
                        "Temp=%{y:.3f} °C<extra></extra>"
                    ),
                    line=dict(color="#268BD2", width=1),
                ),
                row=idx,
                col=1,
            )
            fig.update_yaxes(title_text="Temp [°C]", row=idx, col=1)

    tz_info = sensor_frames[panels[-1]]["timestamp"].dt.tz
    if tz_info is None:
        tz_str = "UTC"
    else:
        tz_name = getattr(tz_info, "key", None) or tz_info.tzname(None)
        tz_str = tz_name or str(tz_info)

    fig.update_layout(
        title=avro_name,
        height=320 * len(panels),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text=f"Time ({tz_str})", row=len(panels), col=1)

    if output_path:
        min_ts, max_ts = get_time_bounds(sensor_frames)
        start_value = format_datetime_local(min_ts, tz_info)
        end_value = format_datetime_local(max_ts, tz_info)
        figure_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
        html_doc = build_interactive_html(figure_html, tz_str, start_value, end_value)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(html_doc)
        print(f"✅  Saved interactive plot {output_path}")

def gather_avro_files(input_dir: str, filenames: Optional[List[str]]) -> List[str]:
    """
    Resolve the Avro files to process.
    """
    if filenames:
        resolved = []
        for name in filenames:
            candidate = os.path.join(input_dir, name)
            if os.path.isfile(candidate):
                resolved.append(candidate)
            else:
                print(f"⚠️  Requested file not found: {candidate}")
        return resolved

    return sorted(
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.lower().endswith(".avro")
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Empatica raw Avro recordings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Absolute path to the directory containing Avro files.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Optional list of Avro filenames to process (default: all .avro files).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.getcwd(), "visualizations"),
        help="Directory where figures will be written.",
    )
    parser.add_argument(
        "--tz",
        default=None,
        help=(
            "Timezone used for plotting (e.g. 'UTC', 'America/New_York'). "
            "Defaults to the computer's local time."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Also export each plot as an interactive Plotly HTML file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure resolution (dots per inch) when saving PNGs.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export the combined sensor streams to a CSV file for notebook analysis.",
    )
    parser.add_argument(
        "--csv-path",
        help=(
            "Optional path for the exported CSV file. "
            "Defaults to <output-dir>/<session-name>_raw.csv."
        ),
    )
    return parser.parse_args()


def merge_sensor_frames(
    combined: Dict[str, pd.DataFrame],
    new_frames: Dict[str, pd.DataFrame],
) -> None:
    """
    Append new sensor frames into the combined dictionary (in-place).
    """
    for key, df in new_frames.items():
        if key not in combined:
            combined[key] = df.copy()
            continue

        combined[key] = (
            pd.concat([combined[key], df], ignore_index=True)
            .sort_values("timestamp")
            .drop_duplicates(subset="timestamp")
            .reset_index(drop=True)
        )


def assemble_combined_dataframe(
    sensor_frames: Dict[str, pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """
    Merge all sensor DataFrames into a single timestamp-indexed table.
    """
    combined_df: Optional[pd.DataFrame] = None
    for df in sensor_frames.values():
        if df.empty or "timestamp" not in df.columns:
            continue
        frame = df.set_index("timestamp")
        combined_df = frame if combined_df is None else combined_df.join(frame, how="outer")

    if combined_df is None:
        return None

    return combined_df.sort_index().reset_index().rename(columns={"index": "timestamp"})


def main() -> None:
    args = parse_arguments()

    target_tz: Union[str, datetime.tzinfo]
    if args.tz:
        target_tz = args.tz
    else:
        target_tz = datetime.now().astimezone().tzinfo or "UTC"

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    avro_files = gather_avro_files(input_dir, args.files)
    if not avro_files:
        print("❌ No Avro files found to visualize.")
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    session_name = os.path.basename(os.path.normpath(input_dir))
    export_csv_requested = args.export_csv or bool(args.csv_path)
    print(f"Processing {len(avro_files)} Avro file(s)...")
    combined_frames: Dict[str, pd.DataFrame] = {}
    for avro_path in avro_files:
        basename = os.path.basename(avro_path)
        print(f"→ {basename}")
        data = read_avro_file(avro_path)
        if not data:
            continue

        frames = build_sensor_frames(data["raw_data"], target_tz)
        if not frames:
            print(f"⚠️  No sensor data extracted from {basename}")
            continue

        output_path = os.path.join(
            output_dir, f"{os.path.splitext(basename)[0]}_overview.png"
        )
        plot_sensor_panels(
            frames,
            avro_name=basename,
            output_path=output_path,
            show=args.show,
            dpi=args.dpi,
        )
        if args.interactive:
            interactive_path = os.path.join(
                output_dir, f"{os.path.splitext(basename)[0]}_overview.html"
            )
            plot_interactive_sensor_panels(frames, basename, interactive_path)
        merge_sensor_frames(combined_frames, frames)

    if combined_frames:
        if len(avro_files) > 1:
            combined_path = os.path.join(output_dir, f"{session_name}_combined_overview.png")
            plot_sensor_panels(
                combined_frames,
                avro_name=f"{session_name} (combined)",
                output_path=combined_path,
                show=args.show,
                dpi=args.dpi,
            )
            if args.interactive:
                combined_html = os.path.join(
                    output_dir, f"{session_name}_combined_overview.html"
                )
                plot_interactive_sensor_panels(
                    combined_frames, f"{session_name} (combined)", combined_html
                )
        if export_csv_requested:
            csv_path = args.csv_path or os.path.join(
                output_dir, f"{session_name}_raw.csv"
            )
            combined_df = assemble_combined_dataframe(combined_frames)
            if combined_df is None:
                print("⚠️  No sensor data available to export as CSV.")
            else:
                combined_df.to_csv(csv_path, index=False)
                print(f"✅  Saved CSV {csv_path}")

    print("Done.")


if __name__ == "__main__":
    main()
