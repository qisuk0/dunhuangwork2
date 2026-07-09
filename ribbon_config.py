# ribbon_config.py
# 指尖飞天飘带笔刷：根据食指轨迹绘制敦煌风格的岩彩飘带
import numpy as np
import cv2
import math

# 敦煌岩彩色系（BGR 格式）—— 降低饱和度，更淡雅
DUNHUANG_COLORS = [
    (170, 140, 100),  # 石青（淡化）
    (120, 170, 120),  # 石绿（淡化）
    (100, 120, 160),  # 赭石（淡化）
    ( 90, 115, 145),  # 褐土（淡化）
    (155, 170, 110),  # 青绿（淡化）
    (110, 160, 170),  # 土黄（淡化）
    (135, 115, 150),  # 藕紫（淡化）
    (135, 170, 130),  # 松石（淡化）
    (100, 110, 180),  # 朱砂（淡化）
    (155, 115,  95),  # 靛青（淡化）
]

# 采样与飘带几何参数
SAMPLE_INTERVAL_SEC = 0.008     # 更密集采样，提升流畅度
MAX_TRAIL_POINTS = 45           # 减少点数，降低延迟
BASE_WIDTH = 5                  # 主飘带基础半宽
MIN_WIDTH = 1.5                 # 最细处
ARCHIVE_GAP_SEC = 0.4           # 手势中断超过此时长，则固化旧轨迹

# 透明度：80% 不透明度（alpha 值 0.8）
RIBBON_OPACITY = 0.8

# 状态
_trail_points = []          # [(x, y, timestamp)] 当前活动轨迹
_last_sample_time = 0.0
_active_hand_id = None
_color_index = 0
_ribbon_ever_used = False
_current_color = None
_history_canvas = None      # 已固化历史飘带的画布

# 撤销栈：每次飘带固化时保存 ribbon_layer 快照
_ribbon_undo_stack = []


def reset_ribbon_state():
    """重置飘带状态（清空画布时调用）。"""
    global _trail_points, _last_sample_time, _active_hand_id, _color_index
    global _ribbon_ever_used, _current_color, _history_canvas, _ribbon_undo_stack
    _trail_points.clear()
    _last_sample_time = 0.0
    _active_hand_id = None
    _color_index = 0
    _ribbon_ever_used = False
    _current_color = None
    _ribbon_undo_stack.clear()
    if _history_canvas is not None:
        _history_canvas[:] = 0


def clear_ribbon_only():
    """仅清空飘带笔刷内容，不影响其他图层。"""
    global _trail_points, _history_canvas, _current_color, _ribbon_ever_used, _ribbon_undo_stack
    _trail_points.clear()
    _current_color = None
    _ribbon_ever_used = False
    _ribbon_undo_stack.clear()
    if _history_canvas is not None:
        _history_canvas[:] = 0
    print('[INFO] 画笔内容已清空。')


def has_ribbon_ever_used():
    """外部查询：是否已开始画过飘带（用于锁定贴图）。"""
    return _ribbon_ever_used


def is_ribbon_pose(hand, img_w, img_h):
    """判断是否为画飘带手势：仅识别伸出的食指指尖。"""
    if hand is None:
        return False

    idx_tip = hand.landmark[8]
    idx_pip = hand.landmark[6]
    mid_tip = hand.landmark[12]
    ring_tip = hand.landmark[16]
    pinky_tip = hand.landmark[20]
    wrist = hand.landmark[0]

    idx_len = math.hypot(idx_tip.x - wrist.x, idx_tip.y - wrist.y)
    pip_len = math.hypot(idx_pip.x - wrist.x, idx_pip.y - wrist.y)
    idx_extended = idx_len > pip_len * 1.15

    mid_curled = math.hypot(mid_tip.x - wrist.x, mid_tip.y - wrist.y) < idx_len * 0.60
    ring_curled = math.hypot(ring_tip.x - wrist.x, ring_tip.y - wrist.y) < idx_len * 0.60
    pinky_curled = math.hypot(pinky_tip.x - wrist.x, pinky_tip.y - wrist.y) < idx_len * 0.60

    return idx_extended and mid_curled and ring_curled and pinky_curled


def _get_fingertip(hand, img_w, img_h):
    tip = hand.landmark[8]
    return tip.x * img_w, tip.y * img_h


def _get_next_color():
    global _color_index
    color = DUNHUANG_COLORS[_color_index % len(DUNHUANG_COLORS)]
    _color_index += 1
    return color


# ---------------------------------------------------------------------------
# 几何计算
# ---------------------------------------------------------------------------

def _compute_tangent(pts, i):
    """计算第 i 点的单位切线方向。"""
    n = len(pts)
    if n < 2:
        return (1.0, 0.0)
    if i == 0:
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
    elif i == n - 1:
        dx = pts[-1][0] - pts[-2][0]
        dy = pts[-1][1] - pts[-2][1]
    else:
        dx = pts[i + 1][0] - pts[i - 1][0]
        dy = pts[i + 1][1] - pts[i - 1][1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (1.0, 0.0)
    return (dx / length, dy / length)


def _generate_edges(pts, times, width_scale=1.0):
    """根据轨迹生成飘带左右边缘。"""
    n = len(pts)
    left = []
    right = []
    for i in range(n):
        tx, ty = _compute_tangent(pts, i)
        nx, ny = -ty, tx
        w = BASE_WIDTH * width_scale
        # 首尾渐细
        taper = 1.0
        if i < 4:
            taper = min(taper, (i + 1) / 4.0)
        if i > n - 5:
            taper = min(taper, (n - i) / 4.0)
        w *= taper
        left.append((pts[i][0] + nx * w, pts[i][1] + ny * w))
        right.append((pts[i][0] - nx * w, pts[i][1] - ny * w))
    return left, right


# ---------------------------------------------------------------------------
# 绘制
# ---------------------------------------------------------------------------

def _densify(points, factor=3):
    """在相邻点之间线性插值，使边缘更平滑。"""
    if len(points) < 2:
        return points
    res = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        for j in range(factor):
            t = j / factor
            res.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
    res.append(points[-1])
    return res


def _draw_band(canvas, left, right, pts, color):
    """绘制飘带：填充 + 边缘线 + 中心线（带 80% 透明度混合）。"""
    n = len(pts)
    if n < 3:
        return

    left_d = _densify(left, factor=2)
    right_d = _densify(right, factor=2)

    # 先在临时图层绘制，再做透明度混合
    temp = np.zeros_like(canvas)
    poly = np.array(left_d + right_d[::-1], np.int32)

    # 主色填充
    cv2.fillPoly(temp, [poly], color, lineType=cv2.LINE_AA)

    # 边缘线：淡化颜色，与主色协调
    dark = (90, 100, 130)       # 赭褐（淡化）
    light = (190, 200, 110)     # 青绿（淡化）
    cv2.polylines(temp, [np.array(left_d, np.int32)], False, light, 1, cv2.LINE_AA)
    cv2.polylines(temp, [np.array(right_d, np.int32)], False, dark, 1, cv2.LINE_AA)

    # 中心线（淡化后的敦煌色）
    center = [(pts[i][0], pts[i][1]) for i in range(n)]
    center_d = _densify(center, factor=2)
    cv2.polylines(temp, [np.array(center_d, np.int32)], False, color, 1, cv2.LINE_AA)

    # 80% 透明度混合
    temp_mask = (temp > 0).any(axis=2)
    canvas[temp_mask] = (
        canvas[temp_mask].astype(np.float32) * (1 - RIBBON_OPACITY) +
        temp[temp_mask].astype(np.float32) * RIBBON_OPACITY
    ).astype(np.uint8)


def _render_ribbon(canvas):
    """将当前轨迹渲染为飘带。"""
    global _current_color
    if len(_trail_points) < 3:
        return
    if _current_color is None:
        _current_color = _get_next_color()

    pts = [(x, y) for x, y, _ in _trail_points]
    times = [t for _, _, t in _trail_points]
    left, right = _generate_edges(pts, times, 1.0)
    _draw_band(canvas, left, right, pts, _current_color)


def _archive_current_trail():
    """把当前轨迹固化到历史画布，并开始新轨迹。"""
    global _history_canvas, _trail_points, _current_color
    if _history_canvas is not None and len(_trail_points) >= 3:
        _render_ribbon(_history_canvas)
    _trail_points.clear()
    _current_color = None


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------

def track_fingertip(left_hand, right_hand, canvas_w, canvas_h, current_time):
    """追踪食指指尖并采样轨迹。"""
    global _trail_points, _last_sample_time, _active_hand_id, _current_color

    has_left = left_hand is not None
    has_right = right_hand is not None
    if not has_left and not has_right:
        return False

    target_hand = None
    hand_id = None

    if has_right and is_ribbon_pose(right_hand, canvas_w, canvas_h):
        target_hand = right_hand
        hand_id = "right"
    elif has_left and is_ribbon_pose(left_hand, canvas_w, canvas_h):
        target_hand = left_hand
        hand_id = "left"

    if target_hand is None:
        return False

    if _active_hand_id is not None and _active_hand_id != hand_id:
        _archive_current_trail()
    _active_hand_id = hand_id

    if current_time - _last_sample_time < SAMPLE_INTERVAL_SEC:
        return False
    _last_sample_time = current_time

    x, y = _get_fingertip(target_hand, canvas_w, canvas_h)

    if _trail_points:
        last_x, last_y, last_t = _trail_points[-1]
        dist = math.hypot(x - last_x, y - last_y)
        if dist < 1.5:
            return False
        if current_time - last_t > ARCHIVE_GAP_SEC:
            _archive_current_trail()

    _trail_points.append((float(x), float(y), current_time))

    if len(_trail_points) > MAX_TRAIL_POINTS:
        _archive_current_trail()

    if _current_color is None:
        _current_color = _get_next_color()

    return True


def try_place_ribbon(canvas, canvas_w, canvas_h, current_time):
    """将当前轨迹渲染到画布（使用 BGR + alpha 混合，保留历史轨迹）。"""
    global _history_canvas, _ribbon_ever_used, _current_color, _ribbon_undo_stack

    if len(_trail_points) < 3:
        return False

    if _history_canvas is None:
        _history_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    if _current_color is None:
        _current_color = _get_next_color()

    # 保存当前 ribbon_layer 快照（用于撤销）
    _ribbon_undo_stack.append(canvas.copy())

    # 合并历史轨迹 + 当前轨迹到一个临时 BGR 画布
    tmp = _history_canvas.copy()
    _render_ribbon(tmp)
    # 用 alpha 混合写入 canvas（3 通道）
    mask = (tmp > 0).any(axis=2)
    canvas[mask] = tmp[mask]
    _ribbon_ever_used = True
    return True


def undo_last_ribbon(canvas):
    """撤销最近一次飘带绘制，恢复到上一个快照。返回 True 表示成功撤销。"""
    global _ribbon_undo_stack, _history_canvas
    if not _ribbon_undo_stack:
        return False

    snapshot = _ribbon_undo_stack.pop()
    canvas[:] = snapshot

    # 同时重建 _history_canvas：提取画布上所有非零像素
    mask = (canvas > 0).any(axis=2)
    if _history_canvas is not None:
        _history_canvas[:] = 0
        _history_canvas[mask] = canvas[mask]

    print('[UNDO] 已撤销最近一次飘带绘制')
    return True
