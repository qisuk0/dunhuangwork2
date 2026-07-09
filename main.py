# main.py
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os
import time
# 导入手势配置文件
from gesture_config import draw_poster_on_canvas
from bjwy_config import detect_wave_for_wenlu, detect_fist_open, pick_random_bjwy, paste_bjwy_to_canvas, reset_wave_state, undo_last_paste
from ribbon_config import track_fingertip, try_place_ribbon, reset_ribbon_state, is_ribbon_pose, has_ribbon_ever_used, clear_ribbon_only, undo_last_ribbon

try:
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    use_mediapipe = True
    print('[INFO] MediaPipe mp.solutions API 可用，启用 Holistic 检测。')
except Exception as e:
    print('[WARNING] MediaPipe mp.solutions API 不可用，降级为仅显示画布。', e)
    mp_drawing = None
    mp_holistic = None
    use_mediapipe = False

# 自动创建保存作品文件夹
os.makedirs("posters", exist_ok=True)
save_count = 0

# 画布尺寸定义（画布比摄像头画面稍大，素材更满）
SCREEN_HEIGHT = 540
CAM_WIDTH = 640
CAM_HEIGHT = 480   # 摄像头实际分辨率高度
CANVAS_WIDTH = 640

# 摄像头初始化
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# 画布图层（从下到上）：
#   bg_layer:     背景层（灰色底）
#   dun_layer:    敦煌人物层（assets 素材，第2层）
#   bjwy_layer:   背景元素层（bjwy 素材，第3层）
#   draw_layer:   预留绘制层
#   ribbon_layer: 飘带层（最上层/第1层，指尖飘带，直接覆盖）
# 敦煌壁画经典底色：HSV 管理色相，固定饱和度/明度
# 敦煌壁画红棕沙壁底色：偏红褐，略暗、低饱和
bg_hue = 12           # 色相 0~179（红棕色，敦煌壁画常见土红底）
BG_SAT = 38           # 饱和度（固定，微调）
BG_VAL = 175          # 明度（固定，略暗）
_hsv_bg = np.full((SCREEN_HEIGHT, CANVAS_WIDTH, 3), (bg_hue, BG_SAT, BG_VAL), dtype=np.uint8)
bg_layer = cv2.cvtColor(_hsv_bg, cv2.COLOR_HSV2BGR)
# dun / bjwy 用 BGRA（4 通道），ribbon 用 BGR（3 通道，直接覆盖）
bjwy_layer = np.zeros((SCREEN_HEIGHT, CANVAS_WIDTH, 4), dtype=np.uint8)
dun_layer = np.zeros((SCREEN_HEIGHT, CANVAS_WIDTH, 4), dtype=np.uint8)
draw_layer = np.zeros((SCREEN_HEIGHT, CANVAS_WIDTH, 4), dtype=np.uint8)
ribbon_layer = np.zeros((SCREEN_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

# 画布状态
image_placed = False          # 敦煌人物层是否已贴图
dun_placed_time = 0.0         # 敦煌贴图时间（用于 bjwy 延迟）
current_gesture_name = None

# 截图动效状态
flash_start = 0.0             # 闪白开始时间
flash_duration = 0.3          # 闪白持续秒数

if not cap.isOpened():
    raise RuntimeError("无法打开摄像头，请关闭占用摄像头的软件后重试。")

cv2.namedWindow('Track & Dunhuang Canvas', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Track & Dunhuang Canvas', CAM_WIDTH * 2, SCREEN_HEIGHT)
# 为了 hstack 对齐，让摄像头输出高度也匹配画布高度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
next_draw_time = time.time() + 1
print('[INFO] 程序启动，1秒后开始作画。')


def try_place_image(result):
    """尝试根据当前姿势在敦煌层贴图，成功返回 True。"""
    global image_placed, current_gesture_name, dun_placed_time
    dun_layer.fill(0)
    current_gesture_name = draw_poster_on_canvas(
        dun_layer,
        result.left_hand_landmarks,
        result.right_hand_landmarks,
        result.face_landmarks,
        result.pose_landmarks,
        CANVAS_WIDTH,
        SCREEN_HEIGHT,
    )
    if current_gesture_name is not None:
        dun_placed_time = time.time()
        print(f"[INFO] 已贴图到画布：{current_gesture_name}（敦煌层已锁定，bjwy层1秒后开始响应）")
        image_placed = True
        return True
    else:
        print("[INFO] 当前姿势未匹配到素材，将继续尝试...")
        return False


def handle_wave_for_wenlu(result, current_time):
    """检测正上方/正下方挥手 → 贴 wenlu 纹路素材。返回 True 表示已触发贴图。"""
    wave = detect_wave_for_wenlu(
        result.left_hand_landmarks,
        result.right_hand_landmarks,
        result.face_landmarks,
        CANVAS_WIDTH,
        SCREEN_HEIGHT,
        current_time,
    )
    if wave is None:
        return False

    wenlu_dir = wave["wenlu_dir"]
    img_path, _, paste_dir = pick_random_bjwy(is_wenlu=True, wenlu_dir=wenlu_dir)
    if img_path is None:
        print("[WARN] 没有可用的 wenlu 素材")
        return False

    paste_bjwy_to_canvas(
        bjwy_layer,
        img_path,
        paste_dir,
        wave["hand_x"],
        wave["hand_y"],
        is_wenlu=True,
    )
    return True


def handle_fist_open(result, current_time):
    """检测拳头张开为手掌 → 在对应位置贴普通 bjwy 素材。"""
    fist = detect_fist_open(
        result.left_hand_landmarks,
        result.right_hand_landmarks,
        CANVAS_WIDTH,
        SCREEN_HEIGHT,
        current_time,
    )
    if fist is None:
        return

    img_path, _, _ = pick_random_bjwy(is_wenlu=False)
    if img_path is None:
        print("[WARN] 没有可用的 bjwy 素材")
        return

    paste_bjwy_to_canvas(
        bjwy_layer,
        img_path,
        "hand_pos",
        fist["hand_x"],
        fist["hand_y"],
        is_wenlu=False,
    )


def handle_ribbon(result, current_time):
    """追踪指尖轨迹并在飘带层绘制飘带。"""
    track_fingertip(
        result.left_hand_landmarks,
        result.right_hand_landmarks,
        CANVAS_WIDTH,
        SCREEN_HEIGHT,
        current_time,
    )
    try_place_ribbon(ribbon_layer, CANVAS_WIDTH, SCREEN_HEIGHT, current_time)


def clear_canvas():
    """清空所有画布层，重置状态。"""
    global image_placed, current_gesture_name, next_draw_time, dun_placed_time
    dun_layer.fill(0)
    bjwy_layer.fill(0)
    draw_layer.fill(0)
    ribbon_layer.fill(0)
    image_placed = False
    dun_placed_time = 0.0
    current_gesture_name = None
    next_draw_time = time.time() + 1
    reset_wave_state()
    reset_ribbon_state()
    print('[INFO] 画布已清空，1秒后重新检测姿势贴图。')


def _blend_rgba_layer(bg, fg_rgba, mask=None):
    """将 RGBA 前景图层混合到 BGR 背景上（alpha 混合）。
    mask: 可选，为 True 的位置不进行混合（保护背景）。"""
    if fg_rgba is None or fg_rgba.size == 0:
        return bg
    fg_bgr = fg_rgba[..., :3].astype(np.float32)
    alpha = fg_rgba[..., 3:4].astype(np.float32) / 255.0
    bg_f = bg.astype(np.float32)
    if mask is not None:
        alpha = alpha * (~mask[..., None]).astype(np.float32)
    result = (fg_bgr * alpha + bg_f * (1 - alpha)).astype(np.uint8)
    return result


def composite_canvas():
    """合成所有图层：bg -> dun(assets第2层) -> bjwy(第3层，不覆盖dun) -> ribbon(最上层)。"""
    canvas = bg_layer.astype(np.float32)

    # dun 层（assets 素材，第2层）
    if dun_layer[..., 3].any():
        canvas = _blend_rgba_layer(canvas, dun_layer)

    # 记录 dun 层有内容的位置（bjwy 不能覆盖这些位置）
    dun_mask = (dun_layer[..., 3] > 0) if dun_layer[..., 3].any() else None

    # bjwy 层（挥手素材，第3层，不覆盖 dun 层已有内容）
    if bjwy_layer[..., 3].any():
        canvas = _blend_rgba_layer(canvas, bjwy_layer, mask=dun_mask)

    # draw 层
    if draw_layer[..., 3].any():
        canvas = _blend_rgba_layer(canvas, draw_layer)

    # ribbon 层（飘带，最上层，直接覆盖）
    if ribbon_layer.any():
        mask = (ribbon_layer > 0).any(axis=2)
        canvas[mask] = ribbon_layer[mask]

    canvas = canvas.astype(np.uint8)

    # 截图闪白动效
    global flash_start
    if flash_start > 0:
        elapsed = time.time() - flash_start
        if elapsed < flash_duration:
            alpha = 0.6 * (1 - elapsed / flash_duration)
            white_overlay = np.full_like(canvas, 255, dtype=np.uint8)
            canvas = cv2.addWeighted(canvas, 1, white_overlay, alpha, 0)
        else:
            flash_start = 0.0

    return canvas


if use_mediapipe:
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = holistic.process(rgb)
            rgb.flags.writeable = True

            # 绘制骨骼追踪
            # 画布边界参考线：在摄像头画面上画出画布裁剪区域
            cam_h, cam_w = frame.shape[:2]
            canvas_aspect = CANVAS_WIDTH / SCREEN_HEIGHT
            if canvas_aspect >= cam_w / cam_h:
                ref_w = cam_w
                ref_h = int(cam_w / canvas_aspect)
                ref_x, ref_y = 0, (cam_h - ref_h) // 2
            else:
                ref_h = cam_h
                ref_w = int(cam_h * canvas_aspect)
                ref_x, ref_y = (cam_w - ref_w) // 2, 0
            cv2.rectangle(frame, (ref_x, ref_y), (ref_x + ref_w, ref_y + ref_h),
                          (180, 180, 180), 1, cv2.LINE_AA)
            # 十字分割线（三等分）
            for i in range(1, 3):
                lx = ref_x + ref_w * i // 3
                ly = ref_y + ref_h * i // 3
                cv2.line(frame, (lx, ref_y), (lx, ref_y + ref_h), (140, 140, 140), 1, cv2.LINE_AA)
                cv2.line(frame, (ref_x, ly), (ref_x + ref_w, ly), (140, 140, 140), 1, cv2.LINE_AA)

            if result.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, result.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1)
                )
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            if result.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                # 绘制左掌心识别点（手腕和中指根部的中心 = 贴图位置）
                lm = result.left_hand_landmarks.landmark
                palm_cx = int((lm[0].x + lm[9].x) / 2 * CAM_WIDTH)
                palm_cy = int((lm[0].y + lm[9].y) / 2 * CAM_HEIGHT)
                cv2.circle(frame, (palm_cx, palm_cy), 6, (0, 0, 255), -1)
                cv2.circle(frame, (palm_cx, palm_cy), 9, (255, 255, 255), 2)
            if result.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
                # 绘制右掌心识别点（手腕和中指根部的中心 = 贴图位置）
                lm = result.right_hand_landmarks.landmark
                palm_cx = int((lm[0].x + lm[9].x) / 2 * CAM_WIDTH)
                palm_cy = int((lm[0].y + lm[9].y) / 2 * CAM_HEIGHT)
                cv2.circle(frame, (palm_cx, palm_cy), 6, (0, 0, 255), -1)
                cv2.circle(frame, (palm_cx, palm_cy), 9, (255, 255, 255), 2)

            # 状态提示文字
            status_text = f'ESC退出 | c清空 | x清画笔 | z撤销 | a/d色相({bg_hue}) | q截图'
            if image_placed:
                status_text += f' | 敦煌: {current_gesture_name}'
            else:
                status_text += ' | 等待姿势匹配...'
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            now = time.time()

            # === 敦煌层：贴图后锁定 ===
            if not image_placed:
                if now >= next_draw_time:
                    try_place_image(result)

            # === bjwy 层：每次挥手贴一张 ===
            if image_placed:
                # 飘带层：指尖追踪（敦煌贴图后即可使用，始终运行）
                handle_ribbon(result, now)

                # 一旦开始画过飘带，就不再贴 bjwy 图
                if not has_ribbon_ever_used():
                    bjwy_ready = (now - dun_placed_time >= 1.0)
                    if bjwy_ready:
                        # 只要任意一只手处于画飘带手势，就不触发贴图
                        ribbon_active = (
                            (result.right_hand_landmarks is not None and is_ribbon_pose(result.right_hand_landmarks, CANVAS_WIDTH, SCREEN_HEIGHT)) or
                            (result.left_hand_landmarks is not None and is_ribbon_pose(result.left_hand_landmarks, CANVAS_WIDTH, SCREEN_HEIGHT))
                        )
                        if not ribbon_active:
                            # wenlu 优先：先检测挥手贴纹路
                            wenlu_triggered = handle_wave_for_wenlu(result, now)
                            # 只有 wenlu 没触发时，才检测 fist_open
                            if not wenlu_triggered:
                                handle_fist_open(result, now)

            # 合成并显示
            canvas = composite_canvas()
            # 对齐摄像头帧高度到画布高度
            frame_resized = cv2.resize(frame, (CAM_WIDTH, SCREEN_HEIGHT))
            combined_window = np.hstack((frame_resized, canvas))
            cv2.imshow('Track & Dunhuang Canvas', combined_window)

            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord('c'):
                clear_canvas()
            if key == ord('x'):
                ribbon_layer.fill(0)
                clear_ribbon_only()
            if key == ord('z'):
                # 撤销：优先撤销 bjwy 贴图，再撤销飘带
                if not undo_last_paste(bjwy_layer):
                    undo_last_ribbon(ribbon_layer)
            if key == ord('a'):
                bg_hue = (bg_hue - 5) % 180
                _hsv_bg[:] = (bg_hue, BG_SAT, BG_VAL)
                bg_layer[:] = cv2.cvtColor(_hsv_bg, cv2.COLOR_HSV2BGR)
                print(f'[INFO] 背景色相 H={bg_hue}')
            if key == ord('d'):
                bg_hue = (bg_hue + 5) % 180
                _hsv_bg[:] = (bg_hue, BG_SAT, BG_VAL)
                bg_layer[:] = cv2.cvtColor(_hsv_bg, cv2.COLOR_HSV2BGR)
                print(f'[INFO] 背景色相 H={bg_hue}')
            if key == ord('q'):
                save_path = f'posters/canvas_{save_count}.png'
                cv2.imwrite(save_path, canvas)
                flash_start = time.time()  # 触发闪白动效
                print(f'[INFO] 作品已保存至posters文件夹：{save_path}')
                save_count += 1
else:
    print('[INFO] 仅使用固定画布渲染，未启用 MediaPipe Holistic。')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        status_text = 'ESC退出 | c清空画布 | q保存画布'
        if image_placed:
            status_text += f' | 当前: {current_gesture_name}'
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        now = time.time()
        if not image_placed and now >= next_draw_time:
            current_gesture_name = draw_poster_on_canvas(
                dun_layer, None, None, None, None,
                CANVAS_WIDTH, SCREEN_HEIGHT,
                forced_gesture="fly_3",
            )
            if current_gesture_name is not None:
                print(f"[INFO] 已贴图到画布：{current_gesture_name}（画布已锁定）")
                image_placed = True

        canvas = composite_canvas()
        frame_resized = cv2.resize(frame, (CAM_WIDTH, SCREEN_HEIGHT))
        combined_window = np.hstack((frame_resized, canvas))
        cv2.imshow('Track & Dunhuang Canvas', combined_window)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            break
        if key == ord('c'):
            clear_canvas()
        if key == ord('x'):
            ribbon_layer.fill(0)
            clear_ribbon_only()
        if key == ord('z'):
            if not undo_last_paste(bjwy_layer):
                undo_last_ribbon(ribbon_layer)
        if key == ord('a'):
            bg_hue = (bg_hue - 5) % 180
            _hsv_bg[:] = (bg_hue, BG_SAT, BG_VAL)
            bg_layer[:] = cv2.cvtColor(_hsv_bg, cv2.COLOR_HSV2BGR)
        if key == ord('d'):
            bg_hue = (bg_hue + 5) % 180
            _hsv_bg[:] = (bg_hue, BG_SAT, BG_VAL)
            bg_layer[:] = cv2.cvtColor(_hsv_bg, cv2.COLOR_HSV2BGR)
        if key == ord('q'):
            save_path = f'posters/canvas_{save_count}.png'
            cv2.imwrite(save_path, canvas)
            flash_start = time.time()  # 触发闪白动效
            print(f'[INFO] 作品已保存至posters文件夹：{save_path}')
            save_count += 1

cap.release()
cv2.destroyAllWindows()
