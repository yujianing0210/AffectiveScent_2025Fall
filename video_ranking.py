# =========================================
# Cross-participant EDA → arousal box plots
# =========================================

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import os
from datetime import datetime
from matplotlib.patches import Patch


# -------------------------------------------------
# 1. 配置：每个 participant 的 EDA CSV + events.csv 路径
#    👉 请把下面这些路径改成你本机的实际路径
# -------------------------------------------------

participants = [
    {
        "name": "P1_G1",
        "eda_csv": "/Users/jianingyu/Downloads/Design_Analytics/data_analysis_final/csv_participants/1206ASH-G1/v6_raw_eda_with_variability_5s.csv",
        "events_csv": "/Users/jianingyu/Downloads/Design_Analytics/data_analysis_final/csv_participants/1206ASH-G1/events.csv",
    },
    {
        "name": "P2_G2",
        "eda_csv": "/Users/jianingyu/Downloads/Design_Analytics/data_analysis_final/csv_participants/1211ASH-G2/data_analysis_final_raw_eda_with_variability_5s.csv",
        "events_csv": "/Users/jianingyu/Downloads/Design_Analytics/data_analysis_final/csv_participants/1211ASH-G2/events.csv",
    },
    {
        "name": "P3_G2",
        "eda_csv": "/Users/jianingyu/Downloads/Design_Analytics/data_analysis_final/csv_participants/1206MELO-G2/v6_raw_eda_with_variability_5s.csv",
        "events_csv": "/Users/jianingyu/Downloads/Design_Analytics/data_analysis_final/csv_participants/1206MELO-G2/events.csv",
    },
    {
        "name": "P4_G1",
        "eda_csv": "/Users/jianingyu/Downloads/Design_Analytics/data_analysis_final/csv_participants/1211MELO-G1/data_analysis_final_raw_eda_with_variability_5s.csv",
        "events_csv": "/Users/jianingyu/Downloads/Design_Analytics/data_analysis_final/csv_participants/1211MELO-G1/events.csv",
    },
    {
        "name": "P5_G3",
        "eda_csv": "/Users/jianingyu/Downloads/Design_Analytics/data_analysis_final/csv_participants/1206TERESA-G3/data_analysis_final_raw_eda_with_variability_5s.csv",
        "events_csv": "/Users/jianingyu/Downloads/Design_Analytics/data_analysis_final/csv_participants/1206TERESA-G3/events.csv",
    },
    {
        "name": "P6_G3",
        "eda_csv": "/Users/jianingyu/Downloads/Design_Analytics/data_analysis_final/csv_participants/1211WENXIN-G3/data_analysis_final_raw_eda_with_variability_5s.csv",
        "events_csv": "/Users/jianingyu/Downloads/Design_Analytics/data_analysis_final/csv_participants/1211WENXIN-G3/events.csv",
    },
    {
        "name": "P7_NOMY1206",
        "eda_csv": "/Users/jianingyu/Downloads/Design_Analytics/data_analysis_final/csv_participants/1206NOMY-G4/data_analysis_final_raw_eda_with_variability_5s.csv",
        "events_csv": "/Users/jianingyu/Downloads/Design_Analytics/data_analysis_final/csv_participants/1206NOMY-G4/events.csv",
    },
    {
        "name": "P8_AWU1210",
        "eda_csv": "/Users/jianingyu/Downloads/Design_Analytics/data_analysis_final/csv_participants/1210AWU-G4/data_analysis_final_raw_eda_with_variability_5s.csv",
        "events_csv": "/Users/jianingyu/Downloads/Design_Analytics/data_analysis_final/csv_participants/1210AWU-G4/events.csv",
    },
]

# -------------------------------------------------
# 2. 统一解析时间：支持 ISO8601 + 时区
# -------------------------------------------------

def to_naive_datetime(series: pd.Series) -> pd.Series:
    """
    Parse timestamps that may include timezone offset, fractional seconds, etc.,
    and convert to naive datetime (no tz).
    """
    dt = pd.to_datetime(series, format="ISO8601", errors="coerce")

    # 如果带 tz → 转成 naive
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert(None)

    # 调试：如果有 NaT，打印出来（方便查错）
    if dt.isna().any():
        print("⚠️ Some timestamps were NaT after parsing:", dt.isna().sum())
        print(series[dt.isna()].head())

    return dt


# -------------------------------------------------
# 3. 核心函数：从单个 participant 提取每个视频的 arousal summary
#    这里加了 offset_hours=-5，把 EDA 往前平移 5 小时对齐 events
# -------------------------------------------------

def extract_video_arousal_from_participant(
    eda_csv_path: str | Path,
    events_csv_path: str | Path,
    participant_name: str | None = None,
    arousal_col: str = "EDA_Phasic_z",
    offset_hours: int = -5,   # 👈 关键：统一减去 5 小时
    verbose: bool = True,
) -> pd.DataFrame:
    eda_csv_path = Path(eda_csv_path)
    events_csv_path = Path(events_csv_path)

    if participant_name is None:
        participant_name = eda_csv_path.parent.name

    # ---- 读取 EDA ----
    df_eda = pd.read_csv(eda_csv_path)
    if "timestamp_local" not in df_eda.columns:
        raise ValueError(f"{eda_csv_path} 中没有 'timestamp_local' 列")

    df_eda["timestamp_local"] = to_naive_datetime(df_eda["timestamp_local"])

    # ⭐ 这里做时间平移：23:00 → 18:00
    if offset_hours != 0:
        df_eda["timestamp_local"] = df_eda["timestamp_local"] + pd.Timedelta(hours=offset_hours)

    if arousal_col not in df_eda.columns:
        raise ValueError(f"{eda_csv_path} 中没有 '{arousal_col}' 列")

    # ---- 读取 events ----
    events = pd.read_csv(events_csv_path)
    required_ev_cols = {"block", "clip_index", "video_id", "category", "start_time", "end_time"}
    if not required_ev_cols.issubset(events.columns):
        missing = required_ev_cols - set(events.columns)
        raise ValueError(f"{events_csv_path} 缺少列: {missing}")

    events["start_time"] = to_naive_datetime(events["start_time"])
    events["end_time"]   = to_naive_datetime(events["end_time"])

    if verbose:
        print(f"\n=== {participant_name} ===")
        print("EDA   range:", df_eda["timestamp_local"].min(), "→", df_eda["timestamp_local"].max())
        print("EVENT range:", events["start_time"].min(),      "→", events["end_time"].max())

    rows = []

    for _, ev in events.iterrows():
        mask = (df_eda["timestamp_local"] >= ev["start_time"]) & (
            df_eda["timestamp_local"] <= ev["end_time"]
        )
        df_seg = df_eda.loc[mask]

        if verbose:
            print(
                f"  clip {ev['block']}-{ev['clip_index']} ({ev['video_id']}-{ev['category']}): "
                f"{len(df_seg)} samples"
            )

        if len(df_seg) == 0:
            continue

        rows.append(
            {
                "participant": participant_name,
                "block": ev["block"],
                "clip_index": ev["clip_index"],
                "video_id": ev["video_id"],
                "category": ev["category"],
                "start_time": ev["start_time"],
                "end_time": ev["end_time"],
                "arousal_mean": df_seg[arousal_col].mean(),
                "arousal_std": df_seg[arousal_col].std(),
                "arousal_median": df_seg[arousal_col].median(),
                "n_samples": len(df_seg),
            }
        )

    return pd.DataFrame(rows)         

# -------------------------------------------------
# 4. 批量处理所有 participants
# -------------------------------------------------

all_results = []

for p in participants:
    print(f"\nProcessing {p['name']} ...")
    df_p = extract_video_arousal_from_participant(
        eda_csv_path=p["eda_csv"],
        events_csv_path=p["events_csv"],
        participant_name=p["name"],
        arousal_col="EDA_Phasic_z",   # 如果想用别的指标（比如 EDA_Clean_z_std_5s），改这里
        offset_hours=-5,              # 时差修正
        verbose=True,
    )
    all_results.append(df_p)

df_all = pd.concat(all_results, ignore_index=True)
print("\nTotal segments:", len(df_all))
print(df_all.head())

# -------------------------------------------------
# 5. 生成 video_key（例如 '1-A'）并排序
# -------------------------------------------------

df_all["video_key"] = df_all["video_id"].astype(str) + "-" + df_all["category"].astype(str)
df_all["sort_key"] = df_all["category"].astype(str) + df_all["video_id"].astype(int).astype(str).str.zfill(2)
df_all = df_all.sort_values(["sort_key", "participant"]).reset_index(drop=True)


import os
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# ==========================
# 公共：输出文件夹 & 时间戳
# ==========================
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ==========================
# 先确保有 video_key 列
# ==========================
df_all["video_key"] = df_all["video_id"].astype(str) + "-" + df_all["category"].astype(str)

# 按 category + video_id 排序（A1, A2, ... B1, B2, ...）
video_order = sorted(
    df_all["video_key"].unique(),
    key=lambda x: (x.split("-")[1], int(x.split("-")[0]))
)

# ==========================
# 定义类别颜色（A/B/C/D）
# ==========================
category_palette = {
    "A": "#FDB813",  # warm yellow / orange
    "B": "#6AAFE6",  # light blue
    "C": "#9BDF87",  # soft green
    "D": "#C497E6",  # purple
}

# 为每个 video_key 分配颜色（根据它的 category）
video_color_map = {
    vk: category_palette[vk.split("-")[1]]
    for vk in video_order
}

# =========================================================
# STEP 6: 每个视频的 box plot（颜色由 A/B/C/D 决定）+ 保存
# =========================================================
plt.figure(figsize=(22, 6))

sns.boxplot(
    data=df_all,
    x="video_key",
    y="arousal_mean",
    order=video_order,
    palette=video_color_map,      # 👈 关键：按 video_key → category 映射颜色
)

sns.stripplot(
    data=df_all,
    x="video_key",
    y="arousal_mean",
    color="black",
    alpha=0.6,
    jitter=0.15,
    order=video_order,
)

plt.xticks(rotation=45)
plt.xlabel("Video (video_id-category)")
plt.ylabel("Arousal (mean of EDA_Phasic_z)")
plt.title("Cross-participant arousal for each video (color-coded by category)")

# 自定义 legend（A/B/C/D）
legend_handles = [
    Patch(color=color, label=f"Category {cat}")
    for cat, color in category_palette.items()
]
plt.legend(
    handles=legend_handles,
    title="Video category",
    loc="upper right",
    frameon=True,
)

plt.tight_layout()

save_path_video = f"{plot_dir}/arousal_by_video_colored_{timestamp}.png"
plt.savefig(save_path_video, dpi=300)
print(f"Saved: {save_path_video}")

plt.show()


# =========================================================
# STEP 7: 按类别整体的 box plot（A/B/C/D）+ 保存
# =========================================================
plt.figure(figsize=(6, 5))

order_cat = sorted(category_palette.keys())

sns.boxplot(
    data=df_all,
    x="category",
    y="arousal_mean",
    order=order_cat,
    palette=category_palette,     # 直接用同一套颜色
)

sns.stripplot(
    data=df_all,
    x="category",
    y="arousal_mean",
    order=order_cat,
    color="black",
    alpha=0.6,
    jitter=0.15,
)

plt.xlabel("Video category")
plt.ylabel("Arousal (mean of EDA_Phasic_z)")
plt.title("Arousal differences across categories (A/B/C/D)")
plt.tight_layout()

save_path_cat = f"{plot_dir}/arousal_by_category_{timestamp}.png"
plt.savefig(save_path_cat, dpi=300)
print(f"Saved: {save_path_cat}")

plt.show()
