import csv
import math
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 设定模式：True 为动态实时刷新，False 为生成最终静态图 
REAL_TIME_MODE = True
CSV_FILE = "flight_log.csv"
SAVE_FIGURE = True
# 设为 None 时自动按时间命名，例如 flight_plot_20260420_153000.png
FIGURE_NAME = None
ACTIVE_CSV_FILE = CSV_FILE
_LAST_UPDATE_ERROR = ""

# 是否用固定控制周期 dt 重新构造仿真时间轴（推荐 True，和控制循环一致）
USE_DT_SIM_TIME = True
# run.py 外环调用 controller() 的周期（pos_control_timestep）
CONTROL_DT_SEC = 1.0 / 50.0

# --- 评分阈值（来自作业要求图）---
THRESH_POS_MEAN = 0.01   # m
THRESH_POS_STD  = 0.01
THRESH_YAW_MEAN = 0.01   # rad
THRESH_YAW_STD  = 0.001

# 每个目标点统计窗口长度（仿真时间，与 CSV「Time」列一致）
ANALYSIS_WINDOW_SEC = 10.0
# 目标在日志中切换后，再经过多少秒仿真时间才开始纳入统计（忽略切换后瞬态）
POST_TARGET_MEASURE_DELAY_SEC = 10.0
# 当严格窗口没有可用样本时，是否回退到“该段末尾可用窗口”进行统计（仅用于避免全空图）。
ALLOW_SHORT_SEGMENT_FALLBACK = True
# 作业写的是重复 50 次（每次随机目标）后取平均
REQUIRED_REPEAT_COUNT = 50

# 主图：创建 4x3 的画布
fig, axs = plt.subplots(4, 3, figsize=(14, 12))
fig.canvas.manager.set_window_title('UAV Controller Real-Time Telemetry (No Pandas)')
plt.subplots_adjust(hspace=0.45, wspace=0.3)

# 统计图：单独窗口展示评分要求对应的 4 个指标
mean_fig, mean_axs = plt.subplots(2, 2, figsize=(12, 7))
mean_fig.canvas.manager.set_window_title(
    f'Assessment Metrics ({POST_TARGET_MEASURE_DELAY_SEC:.0f}s delay + {ANALYSIS_WINDOW_SEC:.0f}s sim window)'
)
mean_fig.subplots_adjust(wspace=0.28, hspace=0.35)


def resolve_csv_file(preferred_name):
    """
    自动选择最新日志：
    - 兼容固定名（如 flight_log.csv）
    - 兼容时间戳名（如 flight_log_YYYYMMDD_HHMMSS.csv）
    返回修改时间最新的那个文件。
    """
    base, ext = os.path.splitext(preferred_name)
    prefix = f"{base}_"
    candidates = []
    try:
        if os.path.exists(preferred_name):
            candidates.append(preferred_name)
        for name in os.listdir("."):
            if name.endswith(ext) and name.startswith(prefix):
                candidates.append(name)
    except OSError:
        return preferred_name

    if not candidates:
        return preferred_name
    return max(candidates, key=lambda n: os.path.getmtime(n))


def load_data(filename):
    """使用原生 csv 库替代 pandas 读取数据"""
    data = {
        'Time': [], 'Target_X': [], 'Target_Y': [], 'Target_Z': [], 'Target_Yaw': [],
        'Pos_X': [], 'Pos_Y': [], 'Pos_Z': [], 'Roll': [], 'Pitch': [], 'Yaw': [],
        'Err_Yaw': [], 'Out_Vx': [], 'Out_Vy': [], 'Out_Vz': [], 'Out_YawRate': []
    }
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # 原子地解析一整行，避免实时写入期间出现半行导致各列长度不一致
                    parsed = {}
                    for key in data.keys():
                        if key not in row:
                            raise KeyError(key)
                        value = row[key].strip()
                        if value == '':
                            raise ValueError(f"empty field: {key}")
                        parsed[key] = float(value)
                    for key in data.keys():
                        data[key].append(parsed[key])
                except (ValueError, KeyError):
                    continue
    except Exception:
        pass
    if USE_DT_SIM_TIME:
        rebuild_sim_time_from_dt(data, CONTROL_DT_SEC)
    return data


def rebuild_sim_time_from_dt(data, dt):
    """用固定 dt 重建 Time 列，避免日志中时间列抖动影响统计窗口。"""
    if dt <= 0:
        return
    n = len(data['Time'])
    if n == 0:
        return
    data['Time'] = [i * dt for i in range(n)]


def wrap_pi(angle):
    """归一化到 [-π, π]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def _mean_std(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n
    return m, math.sqrt(var)


def compute_waypoint_stats(data):
    """
    按目标点分段：把相邻相同 (Tx,Ty,Tz,Tyaw) 的数据归为一段。
    对每段使用 CSV「Time」列（仿真时间，与 controller 累计 dt 一致）：
      - 记该段首行时刻为 t0（目标刚切换后的首次采样）
      - 统计区间 [t0 + POST_TARGET_MEASURE_DELAY_SEC, t0 + POST_TARGET_MEASURE_DELAY_SEC + ANALYSIS_WINDOW_SEC]
      - 在该区间内计算：
          * pos_err = sqrt(ex^2+ey^2+ez^2) 的 mean / std
          * yaw_err（已 wrap 到 [-π,π]）的 mean / std
      - 若严格区间无有效样本且 ALLOW_SHORT_SEGMENT_FALLBACK=True，
        回退到该段末尾窗口 [max(t0, t_end-ANALYSIS_WINDOW_SEC), t_end]
      - 最终区间内有效采样不足则该段不参与评分（reached=False）
    """
    n = len(data['Time'])
    if n == 0:
        return []

    # 1) 把行按 (Tx,Ty,Tz,Tyaw) 分段
    segments = []   # list of (start_idx, end_idx_exclusive, target_tuple)
    seg_start = 0
    prev_tgt = (data['Target_X'][0], data['Target_Y'][0],
                data['Target_Z'][0], data['Target_Yaw'][0])
    for i in range(1, n):
        cur_tgt = (data['Target_X'][i], data['Target_Y'][i],
                   data['Target_Z'][i], data['Target_Yaw'][i])
        if any(abs(cur_tgt[k] - prev_tgt[k]) > 1e-6 for k in range(4)):
            segments.append((seg_start, i, prev_tgt))
            seg_start = i
            prev_tgt = cur_tgt
    segments.append((seg_start, n, prev_tgt))

    # 2) 对每段计算（起止时刻一律用仿真时间 data['Time']）
    results = []
    for seg_idx, (s, e, tgt) in enumerate(segments):
        t0 = data['Time'][s]
        t_end_seg = data['Time'][e - 1]
        t_measure_start = t0 + POST_TARGET_MEASURE_DELAY_SEC
        t_measure_end = t_measure_start + ANALYSIS_WINDOW_SEC
        win_i = [
            i
            for i in range(s, e)
            if t_measure_start <= data['Time'][i] <= t_measure_end
        ]
        used_fallback = False

        if len(win_i) < 2 and ALLOW_SHORT_SEGMENT_FALLBACK:
            fb_start = max(t0, t_end_seg - ANALYSIS_WINDOW_SEC)
            fb_end = t_end_seg
            win_i = [
                i
                for i in range(s, e)
                if fb_start <= data['Time'][i] <= fb_end
            ]
            if len(win_i) >= 2:
                used_fallback = True
                t_measure_start = fb_start
                t_measure_end = fb_end

        if len(win_i) < 2:
            # 仿真不够长或该段无数据：无法按窗口评分
            results.append({
                'index': seg_idx + 1,
                'target': tgt,
                'reached': False,
                'duration': data['Time'][e - 1] - data['Time'][s],
            })
            continue

        pos_errs = []
        yaw_errs = []
        for i in win_i:
            ex = data['Target_X'][i] - data['Pos_X'][i]
            ey = data['Target_Y'][i] - data['Pos_Y'][i]
            ez = data['Target_Z'][i] - data['Pos_Z'][i]
            pos_errs.append(math.sqrt(ex * ex + ey * ey + ez * ez))
            # yaw 误差：优先用日志里的 Err_Yaw（已 wrap），没有就现算
            if i < len(data['Err_Yaw']) and data['Err_Yaw']:
                yaw_errs.append(wrap_pi(data['Err_Yaw'][i]))
            else:
                yaw_errs.append(wrap_pi(data['Target_Yaw'][i] - data['Yaw'][i]))

        pos_mean, pos_std = _mean_std(pos_errs)
        # yaw 指标使用有符号误差，不做绝对值处理。
        yaw_mean, yaw_std = _mean_std(yaw_errs)

        results.append({
            'index': seg_idx + 1,
            'target': tgt,
            'reached': True,
            't_stats_start': t_measure_start,
            't_stats_end': t_measure_end,
            'used_fallback': used_fallback,
            'samples': len(win_i),
            'duration': data['Time'][win_i[-1]] - data['Time'][win_i[0]],
            'pos_mean': pos_mean,
            'pos_std':  pos_std,
            'yaw_mean': yaw_mean,
            'yaw_std':  yaw_std,
        })
    return results


def summarize_assessment(results):
    """
    按作业要求做“多次测试后的最终指标”：
      - 每个 segment 是一次测试（一次目标点动作）
      - 每次测试在「目标切换后延迟 + 固定仿真时长」窗口内算 pos_mean/pos_std/yaw_mean/yaw_std
      - 最终分数 = 对这些测试结果再做平均
    """
    reached = [r for r in results if r.get('reached')]
    if not reached:
        return None

    final_pos_mean = sum(r['pos_mean'] for r in reached) / len(reached)
    final_pos_std = sum(r['pos_std'] for r in reached) / len(reached)
    final_yaw_mean = sum(r['yaw_mean'] for r in reached) / len(reached)
    final_yaw_std = sum(r['yaw_std'] for r in reached) / len(reached)

    return {
        'count': len(reached),
        'pos_mean': final_pos_mean,
        'pos_std': final_pos_std,
        'yaw_mean': final_yaw_mean,
        'yaw_std': final_yaw_std,
        'ok_pos_mean': final_pos_mean < THRESH_POS_MEAN,
        'ok_pos_std': final_pos_std < THRESH_POS_STD,
        'ok_yaw_mean': final_yaw_mean < THRESH_YAW_MEAN,
        'ok_yaw_std': final_yaw_std < THRESH_YAW_STD,
    }


def format_stats_text(results):
    """把统计结果格式化成一段多行字符串，并打 PASS/FAIL 标签"""
    if not results:
        return "尚无数据"

    header = (
        f"{'#':>2}  {'Target (x,y,z,yaw)':>28}  "
        f"{'pos_mean':>10}  {'pos_std':>9}  "
        f"{'yaw_mean':>10}  {'yaw_std':>9}   verdict"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    all_pass = True
    for r in results:
        tgt = r['target']
        tgt_str = f"({tgt[0]:+.2f},{tgt[1]:+.2f},{tgt[2]:+.2f},{tgt[3]:+.2f})"
        if not r['reached']:
            lines.append(f"{r['index']:>2}  {tgt_str:>28}  "
                         f"{'--':>10}  {'--':>9}  "
                         f"{'--':>10}  {'--':>9}   NOT REACHED")
            all_pass = False
            continue

        pm_ok = r['pos_mean'] < THRESH_POS_MEAN
        ps_ok = r['pos_std']  < THRESH_POS_STD
        ym_ok = r['yaw_mean'] < THRESH_YAW_MEAN
        ys_ok = r['yaw_std']  < THRESH_YAW_STD
        ok = pm_ok and ps_ok and ym_ok and ys_ok
        all_pass = all_pass and ok

        def tag(v, ok): return f"{v:>10.5f}" if ok else f"{v:>9.5f}*"
        lines.append(
            f"{r['index']:>2}  {tgt_str:>28}  "
            f"{tag(r['pos_mean'], pm_ok)}  "
            f"{r['pos_std']:>8.5f}{'' if ps_ok else '*'}  "
            f"{tag(r['yaw_mean'], ym_ok)}  "
            f"{r['yaw_std']:>8.5f}{'' if ys_ok else '*'}   "
            f"{'PASS' if ok else 'FAIL'}"
        )

    summary = summarize_assessment(results)
    lines.append(sep)
    lines.append(
        f"阈值: pos_mean<{THRESH_POS_MEAN}  pos_std<{THRESH_POS_STD}  "
        f"yaw_mean<{THRESH_YAW_MEAN}  yaw_std<{THRESH_YAW_STD}   "
        f"（'*' 代表超阈）"
    )
    if summary is not None:
        overall_ok = (
            summary['ok_pos_mean'] and summary['ok_pos_std'] and
            summary['ok_yaw_mean'] and summary['ok_yaw_std']
        )
        lines.append(
            f"最终(跨测试平均, n={summary['count']}/{REQUIRED_REPEAT_COUNT}): "
            f"pos_mean={summary['pos_mean']:.5f}, pos_std={summary['pos_std']:.5f}, "
            f"yaw_mean={summary['yaw_mean']:.5f}, yaw_std={summary['yaw_std']:.5f}  "
            f"=> {'PASS' if overall_ok else 'FAIL'}"
        )
    lines.append(f"分段判定: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    return "\n".join(lines)


def draw_mean_figure(results):
    """在单独窗口绘制与评分要求一致的 4 指标图"""
    axes = [mean_axs[0, 0], mean_axs[0, 1], mean_axs[1, 0], mean_axs[1, 1]]
    for ax in axes:
        ax.clear()

    reached = [r for r in results if r.get('reached')]
    if not reached:
        for ax in axes:
            ax.text(0.5, 0.5, "No reached waypoint yet", ha='center', va='center')
            ax.grid(True, axis='y')
        mean_fig.canvas.draw_idle()
        return

    labels = [f"T{idx + 1}" for idx in range(len(reached))]
    x = list(range(len(labels)))
    summary = summarize_assessment(results)

    metric_cfgs = [
        ("Position Mean (m)", [r['pos_mean'] for r in reached], THRESH_POS_MEAN, mean_axs[0, 0], 'pos_mean'),
        ("Position Std (m)", [r['pos_std'] for r in reached], THRESH_POS_STD, mean_axs[0, 1], 'pos_std'),
        ("Yaw Mean (rad)", [r['yaw_mean'] for r in reached], THRESH_YAW_MEAN, mean_axs[1, 0], 'yaw_mean'),
        ("Yaw Std (rad)", [r['yaw_std'] for r in reached], THRESH_YAW_STD, mean_axs[1, 1], 'yaw_std'),
    ]

    for title, values, threshold, ax, key in metric_cfgs:
        colors = ['tab:green' if v < threshold else 'tab:red' for v in values]
        ax.bar(
            x, values, color=colors, alpha=0.85,
            label=f'per-test ({POST_TARGET_MEASURE_DELAY_SEC:.0f}s delay + {ANALYSIS_WINDOW_SEC:.0f}s sim)',
        )
        ax.axhline(threshold, color='k', linestyle='--', linewidth=1.2, label=f"threshold={threshold}")
        if summary is not None:
            final_avg = summary[key]
            final_ok = final_avg < threshold
            ax.axhline(
                final_avg, color='tab:blue', linestyle='-.', linewidth=1.2,
                label=f"final avg={final_avg:.5f} ({'PASS' if final_ok else 'FAIL'})"
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_title(title)
        ax.grid(True, axis='y')
        ax.legend(loc='best', fontsize=8)

    if summary is not None:
        final_ok = (
            summary['ok_pos_mean'] and summary['ok_pos_std'] and
            summary['ok_yaw_mean'] and summary['ok_yaw_std']
        )
        mean_fig.suptitle(
            f"Assessment from reached tests: n={summary['count']}/{REQUIRED_REPEAT_COUNT}  "
            f"Overall={'PASS' if final_ok else 'FAIL'}",
            fontsize=11
        )

    mean_fig.tight_layout(rect=[0, 0, 1, 0.95])
    mean_fig.canvas.draw_idle()


def draw_plots(data):
    """绘制 12 个子图，并更新 mean 单独图"""

    for ax in axs.flatten():
        ax.clear()

    t = data['Time']

    def plot_single(ax, x, y, style, title, label=None):
        n = min(len(x), len(y))
        if n > 0:
            if label:
                ax.plot(x[:n], y[:n], style, label=label)
            else:
                ax.plot(x[:n], y[:n], style)
        ax.set_title(title)
        ax.grid(True)

    def plot_error(ax, x, target, actual, style, title):
        n = min(len(x), len(target), len(actual))
        if n > 0:
            err = [target[i] - actual[i] for i in range(n)]
            ax.plot(x[:n], err, style)
        ax.set_title(title)
        ax.grid(True)

    # --- 第一行：目标位置 vs 实际位置 ---
    plot_single(axs[0, 0], t, data['Target_X'], 'r--', "Position X (m)", label='Target')
    plot_single(axs[0, 0], t, data['Pos_X'], 'b-', "Position X (m)", label='Actual')
    axs[0, 0].legend(loc='best', fontsize=8)

    plot_single(axs[0, 1], t, data['Target_Y'], 'r--', "Position Y (m)", label='Target')
    plot_single(axs[0, 1], t, data['Pos_Y'], 'b-', "Position Y (m)", label='Actual')
    axs[0, 1].legend(loc='best', fontsize=8)

    plot_single(axs[0, 2], t, data['Target_Z'], 'r--', "Position Z (m)", label='Target')
    plot_single(axs[0, 2], t, data['Pos_Z'], 'b-', "Position Z (m)", label='Actual')
    axs[0, 2].legend(loc='best', fontsize=8)

    # --- 第二行：误差 ---
    plot_error(axs[1, 0], t, data['Target_X'], data['Pos_X'], 'g-', "Error X (m)")
    plot_error(axs[1, 1], t, data['Target_Y'], data['Pos_Y'], 'g-', "Error Y (m)")
    plot_error(axs[1, 2], t, data['Target_Z'], data['Pos_Z'], 'g-', "Error Z (m)")

    # --- 第三行：速度输出 ---
    plot_single(axs[2, 0], t, data['Out_Vx'], 'm-', "Velocity X Cmd (m/s)")
    plot_single(axs[2, 1], t, data['Out_Vy'], 'm-', "Velocity Y Cmd (m/s)")
    plot_single(axs[2, 2], t, data['Out_Vz'], 'm-', "Velocity Z Cmd (m/s)")

    # --- 第四行：旋转相关 ---
    # 把 Target_Yaw / Actual_Yaw 都 wrap 到 [-π, π]，避免 4.71 vs -1.57 显示断层
    tgt_yaw_wrapped = [wrap_pi(v) for v in data['Target_Yaw']]
    yaw_wrapped     = [wrap_pi(v) for v in data['Yaw']]
    plot_single(axs[3, 0], t, tgt_yaw_wrapped, 'r--', "Yaw (rad, wrapped)", label='Target')
    plot_single(axs[3, 0], t, yaw_wrapped,     'b-',  "Yaw (rad, wrapped)", label='Actual')
    axs[3, 0].legend(loc='best', fontsize=8)

    if data['Err_Yaw']:
        err_yaw_wrapped = [wrap_pi(v) for v in data['Err_Yaw']]
        plot_single(axs[3, 1], t, err_yaw_wrapped, 'g-', "Error Yaw (rad)")
    else:
        plot_error(axs[3, 1], t, data['Target_Yaw'], data['Yaw'], 'g-', "Error Yaw (rad)")

    plot_single(axs[3, 2], t, data['Out_YawRate'], 'm-', "Yaw Rate Cmd (rad/s)")

    # --- 单独窗口：每目标点 mean 指标图 ---
    results = compute_waypoint_stats(data)
    draw_mean_figure(results)

    return results


def print_stats_to_console(results):
    """静态模式下在终端也打一份，便于复制到报告里"""
    print()
    print("=" * 90)
    print(
        " 每个目标点的精度统计（仿真时间；切换目标后 "
        f"{POST_TARGET_MEASURE_DELAY_SEC:.0f}s 起算，连续 {ANALYSIS_WINDOW_SEC:.0f}s 窗口；"
        "pos = 三维欧几里得误差; yaw = wrap 到 [-π,π] 的有符号误差）"
    )
    if ALLOW_SHORT_SEGMENT_FALLBACK:
        print(" 注：严格窗口不足时，自动回退到该目标段末尾可用窗口统计。")
    print("=" * 90)
    print(format_stats_text(results))
    print("=" * 90)


def save_figure(filename=None):
    """保存当前图像，可自定义文件名。"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flight_plot_{timestamp}.png"

    _, ext = os.path.splitext(filename)
    if ext == "":
        filename = f"{filename}.png"

    fig.savefig(filename, dpi=180, bbox_inches="tight")

    base, ext = os.path.splitext(filename)
    mean_filename = f"{base}_mean{ext}"
    mean_fig.savefig(mean_filename, dpi=180, bbox_inches="tight")
    print(f"✅ 图像已保存: {filename}")
    print(f"✅ Mean 图已保存: {mean_filename}")


def update(frame):
    """动画更新函数：实时读取 CSV"""
    global ACTIVE_CSV_FILE, _LAST_UPDATE_ERROR
    latest_csv = resolve_csv_file(CSV_FILE)
    if latest_csv != ACTIVE_CSV_FILE:
        ACTIVE_CSV_FILE = latest_csv
        print(f"🔄 切换到最新日志: {ACTIVE_CSV_FILE}")
    data = load_data(ACTIVE_CSV_FILE)
    if not data['Time']:
        return
    try:
        draw_plots(data)
        _LAST_UPDATE_ERROR = ""
    except Exception as exc:
        # 实时模式下容忍读到写入中的 CSV 瞬态；同时打印告警避免“看起来完全没反应”。
        message = f"{type(exc).__name__}: {exc}"
        if message != _LAST_UPDATE_ERROR:
            print(f"⚠️ 绘图更新失败，下一帧重试: {message}")
            _LAST_UPDATE_ERROR = message
        return


if __name__ == "__main__":
    ACTIVE_CSV_FILE = resolve_csv_file(CSV_FILE)
    print(f"📄 使用日志文件: {ACTIVE_CSV_FILE}")

    if REAL_TIME_MODE:
        print("📈 正在启动实时绘图模式 (无 Pandas 版)... 请在另一个终端运行 run.py")
        ani = animation.FuncAnimation(fig, update, interval=200, cache_frame_data=False)
        plt.show()
        if SAVE_FIGURE:
            # 退出时再打一份最终统计到终端
            final_data = load_data(ACTIVE_CSV_FILE)
            if final_data['Time']:
                results = compute_waypoint_stats(final_data)
                print_stats_to_console(results)
            save_figure(FIGURE_NAME)
    else:
        print("📊 生成静态图表...")
        data = load_data(ACTIVE_CSV_FILE)
        if data['Time']:
            results = draw_plots(data)
            print_stats_to_console(results)
            if SAVE_FIGURE:
                save_figure(FIGURE_NAME)
            plt.show()
        else:
            print(f"❌ 找不到 {CSV_FILE}，请先运行 run.py！")
