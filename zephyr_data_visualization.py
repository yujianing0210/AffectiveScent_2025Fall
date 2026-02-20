import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import argparse
import sys
import os
import json

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize breathing rate data from CSV file')
parser.add_argument('csv_file', type=str, nargs='?', default=None,
                    help='Path to the CSV file to process (positional argument)')
parser.add_argument('-i', '--input', '--csv', dest='input_file', type=str, default=None,
                    help='Path to the CSV file to process (alternative to positional argument)')
parser.add_argument('-o', '--output', type=str, default=None,
                    help='Output HTML file name (default: auto-generated from input file)')

args = parser.parse_args()

# Use named argument if provided, otherwise use positional argument
if args.input_file:
    args.csv_file = args.input_file

# Validate that CSV file is provided
if args.csv_file is None:
    parser.error('CSV file is required. Please provide it as a positional argument or use -i/--input/--csv option.')

# Extract subject name from CSV filename (e.g., "Yixing_Zephyr_Summary.csv" -> "Yixing")
csv_basename = os.path.basename(args.csv_file)
subject_name = csv_basename.split('_')[0] if '_' in csv_basename else 'Subject'

# Auto-generate output filename if not specified
if args.output is None:
    args.output = f'{subject_name}_Breathing_Rate_Visualization.html'

# Read the CSV file
try:
    df = pd.read_csv(args.csv_file)
except FileNotFoundError:
    print(f"Error: File '{args.csv_file}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    sys.exit(1)

# Parse the timestamp column
df['Time'] = pd.to_datetime(df['Time'], format='%d/%m/%Y %H:%M:%S.%f')

# Format time as HH:MM:SS for display
def format_time(dt):
    return dt.strftime('%H:%M:%S')

df['Time_Formatted'] = df['Time'].apply(format_time)

# Extract columns for visualization
# Column C (index 2) is BR, Column J (index 9) is BRAmplitude
br_data = df.iloc[:, 2]  # BR column
bra_data = df.iloc[:, 9]  # BRAmplitude column

# Generate minute-based tick values and labels using absolute time
start_time = df['Time'].iloc[0]
end_time = df['Time'].iloc[-1]

# Round start time down to the nearest minute
start_minute = start_time.replace(second=0, microsecond=0)
# Round end time up to the nearest minute
end_minute = end_time.replace(second=0, microsecond=0) + pd.Timedelta(minutes=1)

# Generate minute ticks
minute_ticks = []
minute_labels = []
current_time = start_minute
while current_time <= end_minute:
    minute_ticks.append(current_time.isoformat())
    minute_labels.append(current_time.strftime('%H:%M:%S'))
    current_time += pd.Timedelta(minutes=1)

# Format datetime-local input values (YYYY-MM-DDTHH:MM)
start_datetime_local = start_time.strftime('%Y-%m-%dT%H:%M')
end_datetime_local = end_time.strftime('%Y-%m-%dT%H:%M')

# Read event log Excel file if it exists
# Try multiple possible paths
csv_dir = os.path.dirname(args.csv_file) if os.path.dirname(args.csv_file) else '.'
possible_paths = [
    os.path.join(csv_dir, f'Event_log_{subject_name}.xlsx'),
    os.path.join(csv_dir, 'Event_log_Yanning.xlsx'),  # Try with exact filename
    f'Event_log_{subject_name}.xlsx',
    'Event_log_Yanning.xlsx',
    'Yanning_Zephyr_data/Event_log_Yanning.xlsx'
]

event_log_path = None
for path in possible_paths:
    if os.path.exists(path):
        event_log_path = path
        print(f"Found event log at: {event_log_path}")
        break

phases_data = []
if event_log_path and os.path.exists(event_log_path):
    try:
        import openpyxl
        events_df = pd.read_excel(event_log_path)
        
        # Parse start_time and end_time, combine with date from CSV data
        data_date = start_time.date()
        
        for idx, row in events_df.iterrows():
            if pd.notna(row.get('start_time')) and pd.notna(row.get('end_time')):
                phase_name = str(row.get('phase', f'Phase_{idx}'))
                
                # Parse time strings (format: HH:MM:SS)
                start_time_str = str(row['start_time'])
                end_time_str = str(row['end_time'])
                
                try:
                    # Parse time and combine with date
                    if ':' in start_time_str:
                        time_parts = start_time_str.split(':')
                        start_h = int(time_parts[0])
                        start_m = int(time_parts[1])
                        start_s = int(time_parts[2]) if len(time_parts) > 2 else 0
                        from datetime import time as dt_time
                        phase_start = pd.Timestamp.combine(data_date, dt_time(start_h, start_m, start_s))
                    else:
                        continue
                    
                    if ':' in end_time_str:
                        time_parts = end_time_str.split(':')
                        end_h = int(time_parts[0])
                        end_m = int(time_parts[1])
                        end_s = int(time_parts[2]) if len(time_parts) > 2 else 0
                        from datetime import time as dt_time
                        phase_end = pd.Timestamp.combine(data_date, dt_time(end_h, end_m, end_s))
                        
                        # If end_time is earlier than start_time on the same day, 
                        # it's likely a data error - use the next phase's start or data end time
                        if phase_end < phase_start and phase_end.date() == phase_start.date():
                            # Check if there's a next phase
                            if idx + 1 < len(events_df):
                                next_row = events_df.iloc[idx + 1]
                                if pd.notna(next_row.get('start_time')):
                                    next_start_str = str(next_row['start_time'])
                                    if ':' in next_start_str:
                                        next_parts = next_start_str.split(':')
                                        next_h = int(next_parts[0])
                                        next_m = int(next_parts[1])
                                        next_s = int(next_parts[2]) if len(next_parts) > 2 else 0
                                        phase_end = pd.Timestamp.combine(data_date, dt_time(next_h, next_m, next_s))
                                    else:
                                        phase_end = end_time  # Use data end time as fallback
                                else:
                                    phase_end = end_time  # Use data end time as fallback
                            else:
                                phase_end = end_time  # Use data end time as fallback
                    else:
                        continue
                    
                    # If end equals start, add a small duration (1 second)
                    if phase_end == phase_start:
                        phase_end = phase_start + pd.Timedelta(seconds=1)
                    
                    # Only add if end is after or equal to start
                    if phase_end >= phase_start:
                        phases_data.append({
                            'phase': phase_name,
                            'start': phase_start,
                            'end': phase_end,
                            'label': str(row.get('label', phase_name))
                        })
                    else:
                        print(f"Warning: Skipping phase {phase_name} - end_time ({phase_end}) is not after start_time ({phase_start})")
                except Exception as e:
                    print(f"Warning: Could not parse phase {idx}: {e}")
                    continue
    except Exception as e:
        print(f"Warning: Could not read event log file: {e}")
        phases_data = []
else:
    print(f"Info: Event log file not found, proceeding without phase annotations")

# Create HTML with two charts
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>{subject_name} - Breathing Rate Analysis</title>
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
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
.chart-container {{
    margin-bottom: 2rem;
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
h2 {{
    margin-top: 0;
    color: #333;
    font-size: 1.2rem;
}}
.time-label {{
    text-align: center;
    color: #666;
    font-size: 0.85rem;
    margin-top: 0.5rem;
    font-style: italic;
}}
.info-header {{
    background: white;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    font-size: 0.9rem;
    color: #555;
}}
.info-header strong {{
    color: #268BD2;
}}
</style>
<script charset="utf-8" src="https://cdn.plot.ly/plotly-3.3.0.min.js" integrity="sha256-bO3dS6yCpk9aK4gUpNELtCiDeSYvGYnK7jFI58NQnHI=" crossorigin="anonymous"></script>
</head>
<body>
<h1>{subject_name} - Breathing Rate Data Visualization</h1>

<div class="info-header">
    <strong>Subject:</strong> {subject_name} | 
    <strong>Data Source:</strong> Zephyr Summary | 
    <strong>Date:</strong> {start_time.strftime('%Y-%m-%d')} | 
    <strong>Time Range:</strong> {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}
</div>

<div class="controls">
    <label>
        Start Time
        <input type="datetime-local" id="start-input" value="{start_datetime_local}">
    </label>
    <label>
        End Time
        <input type="datetime-local" id="end-input" value="{end_datetime_local}">
    </label>
    <button id="apply-range">Apply Range</button>
    <button id="reset-range">Reset</button>
</div>

<div class="chart-container">
    <h2>Breathing Rate (BR) - BPM</h2>
    <div id="br-chart" style="height:500px; width:100%;"></div>
    <div class="time-label">Time (local)</div>
</div>

<div class="chart-container">
    <h2>BR Amplitude</h2>
    <div id="bra-chart" style="height:500px; width:100%;"></div>
    <div class="time-label">Time (local)</div>
</div>

<script type="text/javascript">
    // Generate phase shapes and annotations
    var phaseShapes = [];
    var phaseAnnotations = [];
    var phaseColors = ['rgba(255, 200, 200, 0.3)', 'rgba(200, 255, 200, 0.3)', 'rgba(200, 200, 255, 0.3)', 
                       'rgba(255, 255, 200, 0.3)', 'rgba(255, 200, 255, 0.3)', 'rgba(200, 255, 255, 0.3)',
                       'rgba(255, 220, 180, 0.3)', 'rgba(220, 180, 255, 0.3)', 'rgba(180, 255, 220, 0.3)'];
    
    var phases = {json.dumps([{
        'phase': p['phase'],
        'start': p['start'].isoformat(),
        'end': p['end'].isoformat(),
        'label': p['label']
    } for p in phases_data]) if phases_data else '[]'};
    
    phases.forEach(function(phase, index) {{
        var color = phaseColors[index % phaseColors.length];
        phaseShapes.push({{
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: phase.start,
            x1: phase.end,
            y0: 0,
            y1: 1,
            fillcolor: color,
            line: {{width: 0}},
            layer: 'below'
        }});
        
        // Add annotation for phase name
        phaseAnnotations.push({{
            x: phase.start,
            y: 1.02,
            xref: 'x',
            yref: 'paper',
            text: phase.phase,
            showarrow: false,
            font: {{size: 10, color: '#666'}},
            xanchor: 'left',
            bgcolor: 'rgba(255, 255, 255, 0.8)',
            bordercolor: color.replace('0.3', '0.8'),
            borderwidth: 1
        }});
    }});

    // Prepare data for BR chart
    var brData = {{
        x: {json.dumps([t.isoformat() for t in df['Time'].tolist()])},
        y: {json.dumps([float(y) for y in br_data.tolist()])},
        type: 'scatter',
        mode: 'lines',
        name: 'Breathing Rate (BPM)',
        line: {{color: '#268BD2', width: 1}},
        hovertemplate: 'Time=%{{x|%H:%M:%S}}<br>BR=%{{y:.2f}} BPM<extra></extra>'
    }};

    var brLayout = {{
        title: {{text: 'Breathing Rate (BPM) Over Time', font: {{size: 16}}}},
        xaxis: {{
            title: 'Time (HH:MM:SS)',
            type: 'date',
            showgrid: true,
            gridcolor: '#e0e0e0',
            tickmode: 'array',
            tickvals: {minute_ticks},
            ticktext: {minute_labels},
            tickformat: '%H:%M:%S',
            tickangle: -45
        }},
        yaxis: {{
            title: '',
            showgrid: true,
            gridcolor: '#e0e0e0'
        }},
        shapes: phaseShapes,
        annotations: [
            {{
                text: 'Breathing Rate (BPM)',
                xref: 'paper',
                yref: 'paper',
                x: -0.08,
                y: 0.5,
                xanchor: 'center',
                yanchor: 'middle',
                textangle: -90,
                font: {{size: 12, color: '#333'}},
                showarrow: false
            }}
        ].concat(phaseAnnotations.length > 0 ? phaseAnnotations : []),
        hovermode: 'x unified',
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        margin: {{l: 80, r: 50, t: 80, b: 80}}
    }};

    // Wait for page and Plotly to be fully loaded
    window.addEventListener('load', function() {{
        var brPlotDiv = document.getElementById('br-chart');
        var braPlotDiv = document.getElementById('bra-chart');
        
        // Wait for Plotly to be fully loaded
        if (typeof Plotly === 'undefined') {{
            console.error('Plotly library not loaded');
            brPlotDiv.innerHTML = '<p style="color: red; padding: 2rem;">Error: Plotly library failed to load. Please check your internet connection.</p>';
            braPlotDiv.innerHTML = '<p style="color: red; padding: 2rem;">Error: Plotly library failed to load. Please check your internet connection.</p>';
        }} else {{
            // Ensure data is valid before plotting
            if (brData.x && brData.y && brData.x.length > 0 && brData.y.length > 0) {{
                console.log('Plotting BR data:', brData.x.length, 'points');
                try {{
                    Plotly.newPlot(brPlotDiv, [brData], brLayout);
                }} catch(err) {{
                    console.error('Plotly BR error:', err);
                    brPlotDiv.innerHTML = '<p style="color: red; padding: 2rem;">Error plotting data: ' + err.message + '</p>';
                }}
            }} else {{
                console.error('Error: Invalid BR data structure', brData);
                brPlotDiv.innerHTML = '<p style="color: red; padding: 2rem;">Error: No valid data available for visualization.</p>';
            }}
            
            if (braData.x && braData.y && braData.x.length > 0 && braData.y.length > 0) {{
                console.log('Plotting BRA data:', braData.x.length, 'points');
                try {{
                    Plotly.newPlot(braPlotDiv, [braData], braLayout);
                }} catch(err) {{
                    console.error('Plotly BRA error:', err);
                    braPlotDiv.innerHTML = '<p style="color: red; padding: 2rem;">Error plotting data: ' + err.message + '</p>';
                }}
            }} else {{
                console.error('Error: Invalid BRA data structure', braData);
                braPlotDiv.innerHTML = '<p style="color: red; padding: 2rem;">Error: No valid data available for visualization.</p>';
            }}
        }}
    }});

    // Prepare data for BRAmplitude chart
    var braData = {{
        x: {json.dumps([t.isoformat() for t in df['Time'].tolist()])},
        y: {json.dumps([float(y) for y in bra_data.tolist()])},
        type: 'scatter',
        mode: 'lines',
        name: 'BR Amplitude',
        line: {{color: '#DC143C', width: 1}},
        hovertemplate: 'Time=%{{x|%H:%M:%S}}<br>BR Amplitude=%{{y:.2f}}<extra></extra>'
    }};

    var braLayout = {{
        title: {{text: 'BR Amplitude Over Time', font: {{size: 16}}}},
        xaxis: {{
            title: 'Time (HH:MM:SS)',
            type: 'date',
            showgrid: true,
            gridcolor: '#e0e0e0',
            tickmode: 'array',
            tickvals: {minute_ticks},
            ticktext: {minute_labels},
            tickformat: '%H:%M:%S',
            tickangle: -45
        }},
        yaxis: {{
            title: '',
            showgrid: true,
            gridcolor: '#e0e0e0'
        }},
        shapes: phaseShapes,
        annotations: [
            {{
                text: 'BR Amplitude',
                xref: 'paper',
                yref: 'paper',
                x: -0.08,
                y: 0.5,
                xanchor: 'center',
                yanchor: 'middle',
                textangle: -90,
                font: {{size: 12, color: '#333'}},
                showarrow: false
            }}
        ].concat(phaseAnnotations.length > 0 ? phaseAnnotations : []),
        hovermode: 'x unified',
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        margin: {{l: 80, r: 50, t: 80, b: 80}}
    }};

    // Time range control functionality
    const startInput = document.getElementById("start-input");
    const endInput = document.getElementById("end-input");
    const applyBtn = document.getElementById("apply-range");
    const resetBtn = document.getElementById("reset-range");

    function applyRange() {{
        const startTime = new Date(startInput.value);
        const endTime = new Date(endInput.value);
        
        if (startTime >= endTime) {{
            alert("Start time must be before end time!");
            return;
        }}

        // Update both charts
        const update = {{
            "xaxis.range[0]": startTime.toISOString(),
            "xaxis.range[1]": endTime.toISOString()
        }};
        
        Plotly.relayout(brPlotDiv, update);
        Plotly.relayout(braPlotDiv, update);
    }}

    function resetRange() {{
        Plotly.relayout(brPlotDiv, {{"xaxis.autorange": true}});
        Plotly.relayout(braPlotDiv, {{"xaxis.autorange": true}});
        
        // Reset input values
        startInput.value = "{start_datetime_local}";
        endInput.value = "{end_datetime_local}";
    }}

    applyBtn.addEventListener("click", applyRange);
    resetBtn.addEventListener("click", resetRange);
</script>
</body>
</html>
"""

# Save to HTML file
with open(args.output, 'w', encoding='utf-8') as f:
    f.write(html_content)

# Create charts directory
charts_dir = 'charts'
os.makedirs(charts_dir, exist_ok=True)

# Create Plotly figure objects for PNG export
# Add phase shapes and annotations for PNG
phase_colors = ['rgba(255, 200, 200, 0.3)', 'rgba(200, 255, 200, 0.3)', 'rgba(200, 200, 255, 0.3)', 
                'rgba(255, 255, 200, 0.3)', 'rgba(255, 200, 255, 0.3)', 'rgba(200, 255, 255, 0.3)',
                'rgba(255, 220, 180, 0.3)', 'rgba(220, 180, 255, 0.3)', 'rgba(180, 255, 220, 0.3)']

png_shapes = []
png_annotations_br = [dict(
    text='Breathing Rate (BPM)',
    xref='paper',
    yref='paper',
    x=-0.08,
    y=0.5,
    xanchor='center',
    yanchor='middle',
    textangle=-90,
    font=dict(size=12, color='#333'),
    showarrow=False
)]

png_annotations_bra = [dict(
    text='BR Amplitude',
    xref='paper',
    yref='paper',
    x=-0.08,
    y=0.5,
    xanchor='center',
    yanchor='middle',
    textangle=-90,
    font=dict(size=12, color='#333'),
    showarrow=False
)]

for idx, phase in enumerate(phases_data):
    color = phase_colors[idx % len(phase_colors)]
    png_shapes.append(dict(
        type='rect',
        xref='x',
        yref='paper',
        x0=phase['start'],
        x1=phase['end'],
        y0=0,
        y1=1,
        fillcolor=color,
        line=dict(width=0),
        layer='below'
    ))
    
    # Add annotation for phase name
    png_annotations_br.append(dict(
        x=phase['start'],
        y=1.02,
        xref='x',
        yref='paper',
        text=phase['phase'],
        showarrow=False,
        font=dict(size=10, color='#666'),
        xanchor='left',
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor=color.replace('0.3', '0.8'),
        borderwidth=1
    ))
    
    png_annotations_bra.append(dict(
        x=phase['start'],
        y=1.02,
        xref='x',
        yref='paper',
        text=phase['phase'],
        showarrow=False,
        font=dict(size=10, color='#666'),
        xanchor='left',
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor=color.replace('0.3', '0.8'),
        borderwidth=1
    ))

# BR Chart
br_fig = go.Figure()
br_fig.add_trace(go.Scatter(
    x=df['Time'],
    y=br_data,
    mode='lines',
    name='Breathing Rate (BPM)',
    line=dict(color='#268BD2', width=1),
    hovertemplate='Time=%{x|%H:%M:%S}<br>BR=%{y:.2f} BPM<extra></extra>'
))
br_fig.update_layout(
    title=dict(text='Breathing Rate (BPM) Over Time', font=dict(size=16)),
    width=1200,
    height=500,
    xaxis=dict(
        title='Time (HH:MM:SS)',
        type='date',
        showgrid=True,
        gridcolor='#e0e0e0',
        tickmode='array',
        tickvals=[pd.Timestamp(t) for t in minute_ticks],
        ticktext=minute_labels,
        tickformat='%H:%M:%S',
        tickangle=-45
    ),
    yaxis=dict(
        title='',
        showgrid=True,
        gridcolor='#e0e0e0'
    ),
    shapes=png_shapes,
    annotations=png_annotations_br,
    hovermode='x unified',
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=80, r=50, t=80, b=80)
)

# BRAmplitude Chart
bra_fig = go.Figure()
bra_fig.add_trace(go.Scatter(
    x=df['Time'],
    y=bra_data,
    mode='lines',
    name='BR Amplitude',
    line=dict(color='#DC143C', width=1),
    hovertemplate='Time=%{x|%H:%M:%S}<br>BR Amplitude=%{y:.2f}<extra></extra>'
))
bra_fig.update_layout(
    title=dict(text='BR Amplitude Over Time', font=dict(size=16)),
    width=1200,
    height=500,
    xaxis=dict(
        title='Time (HH:MM:SS)',
        type='date',
        showgrid=True,
        gridcolor='#e0e0e0',
        tickmode='array',
        tickvals=[pd.Timestamp(t) for t in minute_ticks],
        ticktext=minute_labels,
        tickformat='%H:%M:%S',
        tickangle=-45
    ),
    yaxis=dict(
        title='',
        showgrid=True,
        gridcolor='#e0e0e0'
    ),
    shapes=png_shapes,
    annotations=png_annotations_bra,
    hovermode='x unified',
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=80, r=50, t=80, b=80)
)

# Save charts as PNG files
try:
    br_png_path = os.path.join(charts_dir, f'{subject_name}_Breathing_Rate_BPM.png')
    bra_png_path = os.path.join(charts_dir, f'{subject_name}_BR_Amplitude.png')
    
    br_fig.write_image(br_png_path, width=1200, height=500, scale=2)
    bra_fig.write_image(bra_png_path, width=1200, height=500, scale=2)
    
    print(f"\nCharts saved as PNG:")
    print(f"  - {br_png_path}")
    print(f"  - {bra_png_path}")
except Exception as e:
    print(f"\nWarning: Could not save PNG images. Error: {e}")
    print("Make sure kaleido is installed: pip install kaleido")

print("\nVisualization created successfully!")
print(f"Input file: {args.csv_file}")
print(f"Output file: {args.output}")
print(f"Total data points: {len(df)}")
print(f"Time range: {df['Time'].iloc[0].strftime('%H:%M:%S')} to {df['Time'].iloc[-1].strftime('%H:%M:%S')}")
print(f"BR range: {br_data.min():.2f} - {br_data.max():.2f} BPM")
print(f"BRAmplitude range: {bra_data.min():.2f} - {bra_data.max():.2f}")