# bjwy_config.py
# 管理 bjwy 画布层的素材贴图逻辑
import numpy as np
from PIL import Image
import os
import random
import time
import math

# bjwy 素材目录
BJWY_DIR = "bjwy"
WENLU_DIR = os.path.join(BJWY_DIR, "wenlu")

# wenlu 文件夹内的素材（纹路类，只能在画面上方/下方挥手时使用）
_wenlu_files = [os.path.join("wenlu", f) for f in os.listdir(WENLU_DIR) if f.endswith(".tif")]

# 双手挥动时用的特殊素材
SPECIAL_BJWY = [
    "山水3.tif",
    "m.tif",
    "中间小龛下.tif",
    "九色鹿(600dpi)-王宫.tif",
]

# 所有 bjwy 文件（不含子文件夹）
_all_bjwy_files = [f for f in os.listdir(BJWY_DIR) if f.endswith(".tif")]
# 普通素材 = 全部 - 特殊素材（不含 wenlu 子文件夹素材，wenlu 独立管理）
NORMAL_BJWY = [f for f in _all_bjwy_files if f not in SPECIAL_BJWY]

# 手势状态追踪：用于检测挥手动作（仅 wenlu）
_prev_wrist_y = {}   # {hand_id: prev_y}
_wave_phase = {}     # {hand_id: "up"|"down"|None}  记录当前挥手阶段
_wave_cooldown = {}  # {hand_id: cooldown_end_time}  挥手冷却时间，防止重复触发
WAVE_COOLDOWN_SEC = 0.8  # 挥手冷却秒数

# 拳头→张开手掌 状态追踪（用于普通素材贴图）
# 核心思路：用指尖到手腕距离的"弯曲程度"来追踪手势变化
# 握拳时 finger_curl 低（指尖靠近手掌），张开时 finger_curl 高
_fist_state = {}           # {hand_id: "fist"|"transition"|None}  当前手势状态
_fist_open_cooldown = {}   # {hand_id: cooldown_end_time}
FIST_OPEN_COOLDOWN_SEC = 1.5

# 握拳阈值：四指平均弯曲度 < CURL_FIST（越小越弯）
CURL_FIST = 0.75
# 张开阈值：四指平均弯曲度 > CURL_OPEN（越大越直）
CURL_OPEN = 0.95
# 弯曲度 = 指尖到手腕距离 / PIP到手腕距离，握拳时接近0.6~0.8，张开时≥1.0

# 全局贴图间隔：任意两次贴图之间至少间隔 1 秒
_last_paste_time = 0.0
PASTE_INTERVAL_SEC = 1.0

# 已贴图区域记录：用于避免重叠贴图
# 列表元素: (x1, y1, x2, y2) 画布像素坐标
_placed_regions = []

# 已使用素材记录：优先贴未使用过的素材
_used_bjwy = set()       # 已贴过的普通素材文件名
_used_wenlu = set()      # 已贴过的 wenlu 素材文件名

# 撤销栈：记录每次贴图操作，支持按 z 撤销
# 格式: {"type": "bjwy"|"wenlu", "region": (x1,y1,x2,y2), "backup": ndarray, "placed_idx": int}
_undo_stack = []


def reset_wave_state():
    """重置挥手状态（清空画布时调用）。"""
    global _placed_regions, _last_paste_time, _used_bjwy, _used_wenlu, _undo_stack
    _prev_wrist_y.clear()
    _wave_phase.clear()
    _wave_cooldown.clear()
    _fist_open_cooldown.clear()
    _fist_state.clear()
    _placed_regions.clear()
    _used_bjwy.clear()
    _used_wenlu.clear()
    _undo_stack.clear()
    _last_paste_time = 0.0


def _finger_curl_ratio(hand):
    """
    计算四指（食指中指无名指小指）的平均弯曲程度。
    返回值：指尖到手腕距离 / PIP到手腕距离 的平均值。
    握拳 ≈ 0.6~0.8，张开 ≈ 1.0~1.3
    """
    wrist = hand.landmark[0]
    ratios = []
    for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        tip = hand.landmark[tip_idx]
        pip = hand.landmark[pip_idx]
        tip_len = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        pip_len = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
        if pip_len > 0.001:
            ratios.append(tip_len / pip_len)
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def _is_open_palm(hand, img_w, img_h):
    """
    判断是否为张开的手掌：四指平均弯曲度 >= CURL_OPEN。
    不再检查并拢条件，只看手指是否伸直。
    """
    return _finger_curl_ratio(hand) >= CURL_OPEN


def _is_fist(hand, img_w, img_h):
    """
    判断是否为握拳姿势：四指平均弯曲度 <= CURL_FIST。
    """
    return _finger_curl_ratio(hand) <= CURL_FIST


def detect_fist_open(left_hand, right_hand, img_w, img_h, current_time):
    """
    检测"拳头张开为手掌"的过渡动作。
    
    逻辑：
    1. 看到握拳（手部相对静止时）→ 标记 saw_fist = True
    2. 在 saw_fist=True 的前提下，手静止且张开 → 触发贴图
    3. 中间状态（既不是握拳也不是张开）不清除 saw_fist 标记
    4. 手在快速移动（挥手）时不触发，避免和 wenlu 冲突
    
    返回: {"hand_x": float, "hand_y": float} 或 None
    """
    from ribbon_config import is_ribbon_pose

    for hand, hand_id in [(right_hand, "right"), (left_hand, "left")]:
        if hand is None:
            _fist_state.pop(hand_id, None)
            continue

        # 飘带手势不触发
        if is_ribbon_pose(hand, img_w, img_h):
            _fist_state.pop(hand_id, None)
            continue

        # 冷却检查
        if _fist_open_cooldown.get(hand_id, 0) >= current_time:
            _fist_state.pop(hand_id, None)
            continue

        wrist = hand.landmark[0]
        wrist_x = wrist.x * img_w
        wrist_y = wrist.y * img_h
        curl = _finger_curl_ratio(hand)

        # 判断手是否在快速移动（挥手）
        prev_wrist = _prev_wrist_y.get(f"fist_{hand_id}", None)
        hand_speed = 0.0
        if prev_wrist is not None:
            hand_speed = math.hypot(wrist_x - prev_wrist[0], wrist_y - prev_wrist[1])
        _prev_wrist_y[f"fist_{hand_id}"] = (wrist_x, wrist_y)

        # 手在快速移动（> 画面2%/帧），说明在挥手，不作为 fist_open 触发
        is_waving = hand_speed > img_h * 0.015

        # 判断当前手势
        is_fist_now = curl <= CURL_FIST
        is_open_now = curl >= CURL_OPEN

        prev_state = _fist_state.get(hand_id, None)

        # 调试：每秒打印一次
        debug_key = f"_debug_{hand_id}"
        if debug_key not in _fist_state:
            _fist_state[debug_key] = 0
        if current_time - _fist_state[debug_key] > 1.0:
            _fist_state[debug_key] = current_time
            print(f"[DEBUG] hand={hand_id} curl={curl:.2f} fist={is_fist_now} open={is_open_now} prev={prev_state} waving={is_waving}")

        if is_waving:
            # 正在挥手，不更新状态，避免和 wenlu 冲突
            continue

        if is_fist_now:
            # 看到握拳 → 记住
            _fist_state[hand_id] = "fist"
        elif is_open_now:
            if prev_state == "fist":
                # 之前见过握拳，现在张开了 → 触发！
                # 贴图位置 = 掌心中心（手腕 landmark 0 和 中指根部 landmark 9 的中点）
                palm_center_x = (wrist_x + hand.landmark[9].x * img_w) / 2
                palm_center_y = (wrist_y + hand.landmark[9].y * img_h) / 2
                _fist_state[hand_id] = "open"
                _fist_open_cooldown[hand_id] = current_time + FIST_OPEN_COOLDOWN_SEC
                print(f"[FIST_OPEN] 拳头→张开！hand={hand_id} curl={curl:.2f} palm=({palm_center_x/img_w:.2f}, {palm_center_y/img_h:.2f})")
                return {
                    "hand_x": palm_center_x / img_w,
                    "hand_y": palm_center_y / img_h,
                }
            elif prev_state == "open":
                # 已经触发过了，保持 open 状态
                pass
            else:
                # prev 是 None 或 transition，说明没看到握拳就直接张开了，不触发
                _fist_state[hand_id] = "open"
        else:
            # 中间状态：既不是握拳也不是张开（手指半弯）
            # 不清除状态！保持之前的标记，这样 fist→中间→open 也能触发
            if prev_state is None:
                _fist_state[hand_id] = "transition"
            # 如果 prev_state 是 "fist" 或 "open"，保持不变

    return None


def detect_wave_for_wenlu(left_hand, right_hand, face, img_w, img_h, current_time):
    """
    检测用户正上方/正下方的挥手动作（仅用于 wenlu 纹路素材）。
    要求手在面部正上/下方持续来回移动（累积位移超过阈值），
    避免单帧抖动或握拳张开导致误触发。
    返回: {"hand_x": float, "hand_y": float, "wenlu_dir": "up"|"down"} 或 None
    """
    from ribbon_config import is_ribbon_pose

    has_left = left_hand is not None
    has_right = right_hand is not None

    # 任何一只手处于飘带手势时，不触发
    if (has_right and is_ribbon_pose(right_hand, img_w, img_h)) or \
       (has_left and is_ribbon_pose(left_hand, img_w, img_h)):
        return None

    # 获取人脸鼻尖坐标
    if face is None:
        return None
    nose_y = face.landmark[1].y * img_h
    nose_x = face.landmark[1].x * img_w

    # 对每只手分别检测
    for hand, hand_id in [(right_hand, "right"), (left_hand, "left")]:
        if hand is None:
            # 手消失，清除累积状态
            _wave_phase.pop(hand_id, None)
            _prev_wrist_y.pop(f"wave_accum_{hand_id}", None)
            continue

        if not _is_open_palm(hand, img_w, img_h):
            _wave_phase.pop(hand_id, None)
            _prev_wrist_y.pop(f"wave_accum_{hand_id}", None)
            continue

        wrist_y = hand.landmark[0].y * img_h
        wrist_x = hand.landmark[0].x * img_w
        prev_y = _prev_wrist_y.get(hand_id, wrist_y)
        dy = prev_y - wrist_y  # 正 = 向上移动
        _prev_wrist_y[hand_id] = wrist_y

        # 判断是否在面部正上方/正下方
        wenlu_dir = None
        if wrist_y < nose_y - img_h * 0.02:
            if abs(wrist_x - nose_x) < img_w * 0.15:
                wenlu_dir = "up"
        elif wrist_y > nose_y + img_h * 0.02:
            if abs(wrist_x - nose_x) < img_w * 0.25:
                wenlu_dir = "down"

        if wenlu_dir is None:
            _wave_phase.pop(hand_id, None)
            _prev_wrist_y.pop(f"wave_accum_{hand_id}", None)
            continue

        # 累积位移：需要持续来回移动才触发
        accum_key = f"wave_accum_{hand_id}"
        accum = _prev_wrist_y.get(accum_key, 0.0)
        accum += abs(dy)
        _prev_wrist_y[accum_key] = accum

        # 需要累积位移超过画面 6% 才算真正在挥手
        if accum < img_h * 0.06:
            continue

        # 冷却检查
        if _wave_cooldown.get(hand_id, 0) >= current_time:
            continue

        # 触发！重置累积
        _prev_wrist_y[accum_key] = 0.0
        _wave_cooldown[hand_id] = current_time + WAVE_COOLDOWN_SEC
        print(f"[WENLU_WAVE] 面部正{wenlu_dir}方挥手！accum={accum:.1f}px pos=({wrist_x/img_w:.2f}, {wrist_y/img_h:.2f})")
        return {
            "hand_x": wrist_x / img_w,
            "hand_y": wrist_y / img_h,
            "wenlu_dir": wenlu_dir,
        }

    return None


def pick_random_bjwy(is_wenlu=False, wenlu_dir=None):
    """随机选择一张 bjwy 素材文件路径。
    is_wenlu=True 时从 wenlu 文件夹选，否则从普通素材中选。
    优先选择未使用过的素材，全部用过后才重置。
    返回 (路径, 是否wenlu, wenlu贴图方向) 或 (None, False, None)。"""
    global _used_bjwy, _used_wenlu

    if is_wenlu and _wenlu_files:
        unused = [f for f in _wenlu_files if f not in _used_wenlu]
        if not unused:
            # 全部用过一轮，重置
            _used_wenlu.clear()
            unused = _wenlu_files
        chosen = random.choice(unused)
        _used_wenlu.add(chosen)
        return os.path.join(BJWY_DIR, chosen), True, wenlu_dir

    if NORMAL_BJWY:
        unused = [f for f in NORMAL_BJWY if f not in _used_bjwy]
        if not unused:
            # 全部用过一轮，重置
            _used_bjwy.clear()
            unused = NORMAL_BJWY
        chosen = random.choice(unused)
        _used_bjwy.add(chosen)
        return os.path.join(BJWY_DIR, chosen), False, None

    return None, False, None




def _hstack_images(pil_img, n):
    """将 PIL 图片水平拼接 n 份，返回拼接后的 PIL Image。"""
    w, h = pil_img.size
    merged = Image.new("RGBA", (w * n, h))
    for i in range(n):
        merged.paste(pil_img, (w * i, 0))
    return merged


def _overlap_ratio(x1, y1, x2, y2, ox1, oy1, ox2, oy2):
    """计算两个矩形区域的重叠比例（交集面积 / 新图面积）。"""
    ix1 = max(x1, ox1)
    iy1 = max(y1, oy1)
    ix2 = min(x2, ox2)
    iy2 = min(y2, oy2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    new_area = (x2 - x1) * (y2 - y1)
    return inter_area / new_area if new_area > 0 else 0.0


def _find_non_overlap_position(px, py, target_w, target_h, canvas_w, canvas_h, hand_x, hand_y):
    """在目标位置附近找一个不与已有贴图严重重叠的位置。"""
    global _placed_regions
    x1, y1 = px, py
    x2, y2 = px + target_w, py + target_h

    # 检查与所有已有区域的重叠度
    max_overlap = 0.0
    for rx1, ry1, rx2, ry2 in _placed_regions:
        overlap = _overlap_ratio(x1, y1, x2, y2, rx1, ry1, rx2, ry2)
        max_overlap = max(max_overlap, overlap)

    if max_overlap < 0.3:
        return px, py  # 重叠不大，直接使用

    # 重叠超过 30%，尝试偏移
    offsets = [
        (target_w // 2, 0), (-target_w // 2, 0), (0, target_h // 2), (0, -target_h // 2),
        (target_w // 3, target_h // 3), (-target_w // 3, -target_h // 3),
        (target_w // 3, -target_h // 3), (-target_w // 3, target_h // 3),
    ]
    best_px, best_py = px, py
    best_overlap = max_overlap

    for dx, dy in offsets:
        nx, ny = px + dx, py + dy
        nx = max(0, min(nx, canvas_w - target_w))
        ny = max(0, min(ny, canvas_h - target_h))
        nx1, ny1 = nx, ny
        nx2, ny2 = nx + target_w, ny + target_h

        cur_overlap = 0.0
        for rx1, ry1, rx2, ry2 in _placed_regions:
            overlap = _overlap_ratio(nx1, ny1, nx2, ny2, rx1, ry1, rx2, ry2)
            cur_overlap = max(cur_overlap, overlap)

        if cur_overlap < best_overlap:
            best_overlap = cur_overlap
            best_px, best_py = nx, ny

    if best_overlap < max_overlap:
        print(f"[OVERLAP] 偏移贴图：重叠度 {max_overlap:.1%} -> {best_overlap:.1%}")

    return best_px, best_py


def paste_bjwy_to_canvas(target_canvas, img_path, wenlu_direction, hand_x, hand_y, is_wenlu=False):
    """
    将 bjwy 素材去白底后贴到画布指定位置。
    wenlu_direction: wenlu 时为 "up"|"down"（贴顶部/底部），普通素材时忽略
    hand_x, hand_y: 归一化坐标 (0-1)，普通素材以此确定贴图位置
    is_wenlu: 是否为 wenlu 纹路素材
    """
    global _last_paste_time, _placed_regions

    # 全局贴图间隔检查
    now = time.time()
    if now - _last_paste_time < PASTE_INTERVAL_SEC:
        return
    _last_paste_time = now

    canvas_h, canvas_w = target_canvas.shape[:2]

    try:
        bjwy_img = Image.open(img_path).convert("RGBA")
    except Exception as e:
        print(f"[ERROR] 无法加载 bjwy 素材：{img_path}，{e}")
        return

    img_array = np.array(bjwy_img)

    # 去白底
    white_threshold = 180
    white_area = (img_array[..., 0] > white_threshold) & (img_array[..., 1] > white_threshold) & (img_array[..., 2] > white_threshold)
    img_array[white_area, 3] = 0
    bjwy_img = Image.fromarray(img_array)

    # 缩放
    if is_wenlu:
        # wenlu 纹路素材：水平拼接使总宽度 >= 画布宽度
        orig_w, orig_h = bjwy_img.size
        max_h = int(canvas_h * 0.18)
        if orig_h > max_h:
            ratio = max_h / orig_h
            new_w = int(orig_w * ratio)
            bjwy_img = bjwy_img.resize((new_w, max_h), Image.LANCZOS)
            orig_w, orig_h = new_w, max_h
        n_tiles = max(2, int(np.ceil(canvas_w / orig_w)))
        target_w = orig_w * n_tiles
        target_h = orig_h
        bjwy_img = _hstack_images(bjwy_img, n_tiles)
        if target_w > canvas_w:
            bjwy_img = bjwy_img.crop((0, 0, canvas_w, target_h))
            target_w = canvas_w
    else:
        # 普通素材：中等大小
        target_w = int(canvas_w * 0.30)
        ratio = target_w / bjwy_img.width
        target_h = int(bjwy_img.height * ratio)

    bjwy_img = bjwy_img.resize((target_w, target_h))

    img_arr = np.array(bjwy_img)
    rgba = img_arr.copy()
    rgba[..., :3] = rgba[..., :3][..., ::-1]  # RGB -> BGR
    alpha_f = rgba[..., 3:4] / 255.0

    # 计算贴图位置
    if is_wenlu:
        if wenlu_direction == "up":
            px, py = 0, 0
        else:
            px, py = 0, canvas_h - target_h
    else:
        # 普通素材：手掌张开位置对应画布位置
        px = int(hand_x * canvas_w - target_w // 2)
        py = int(hand_y * canvas_h - target_h // 2)

    # 边界裁剪
    px = max(0, min(px, canvas_w - target_w))
    py = max(0, min(py, canvas_h - target_h))

    # 避免与已有图片重叠（wenlu 不参与）
    if not is_wenlu:
        px, py = _find_non_overlap_position(px, py, target_w, target_h, canvas_w, canvas_h, hand_x, hand_y)

    x1, y1 = px, py
    x2, y2 = px + target_w, py + target_h
    draw_w = x2 - x1
    draw_h = y2 - y1

    if draw_w <= 0 or draw_h <= 0:
        return

    # 保存原始区域数据（用于撤销）
    backup = target_canvas[y1:y2, x1:x2].copy()

    roi = target_canvas[y1:y2, x1:x2].astype(np.float32)
    fg_bgr = rgba[:draw_h, :draw_w, :3].astype(np.float32)
    fg_a = alpha_f[:draw_h, :draw_w]
    roi[:, :, :3] = (fg_bgr * fg_a + roi[:, :, :3] * (1 - fg_a)).astype(np.uint8)
    roi[:, :, 3] = np.maximum(roi[:, :, 3], (fg_a[:, :, 0] * 255).astype(np.uint8))
    target_canvas[y1:y2, x1:x2] = roi.astype(np.uint8)

    # 记录到撤销栈
    placed_idx = len(_placed_regions)
    _placed_regions.append((x1, y1, x2, y2))
    _undo_stack.append({
        "type": "wenlu" if is_wenlu else "bjwy",
        "region": (x1, y1, x2, y2),
        "backup": backup,
        "placed_idx": placed_idx,
    })
    wenlu_tag = "[wenlu]" if is_wenlu else ""
    print(f"[OK] bjwy 素材已贴到画布{wenlu_tag}：{os.path.basename(img_path)}，pos=({px},{py})")


def undo_last_paste(target_canvas):
    """撤销最近一次 bjwy 贴图操作，恢复该区域原始数据。返回 True 表示成功撤销。"""
    global _undo_stack, _placed_regions
    if not _undo_stack:
        return False

    record = _undo_stack.pop()
    x1, y1, x2, y2 = record["region"]
    target_canvas[y1:y2, x1:x2] = record["backup"]

    # 从已贴图区域列表中移除
    if record["placed_idx"] < len(_placed_regions):
        _placed_regions.pop(record["placed_idx"])

    print(f"[UNDO] 已撤销最近一次 {record['type']} 贴图，区域=({x1},{y1})-({x2},{y2})")
    return True
