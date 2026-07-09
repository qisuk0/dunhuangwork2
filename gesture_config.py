# gesture_config.py
import numpy as np
import random
from PIL import Image

# 严格匹配assets文件夹内tif文件名
gesture_img_map = {
    "fly_1": "assets/fly_1.tif",
    "heshou_bodhisattvava": "assets/heshou_bodhisattva.tif",
    "warrior_stick": "assets/warrior_stick.tif",
    "lotus_hand": "assets/lotus_hand.tif",
    "lady_robe": "assets/lady_robe.tif",
    "fly_2": "assets/fly_2.tif",
    "green_lion": "assets/green_lion.tif",
    "wing_spirit": "assets/wing_spirit.tif",
    "fly_3": "assets/fly_3.tif"
}


def judge_gesture(left_hand, right_hand, face, pose, img_w, img_h):
    """根据手部、面部和身体姿态关键点判断当前手势，返回素材名称或 None。
    
    参数:
        left_hand: 左手 landmarks 或 None
        right_hand: 右手 landmarks 或 None
        face: 面部 landmarks 或 None
        pose: 身体姿态 landmarks 或 None（用于辅助判断身体朝向）
        img_w, img_h: 画布宽高
    """
    has_left = left_hand is not None
    has_right = right_hand is not None
    has_face = face is not None
    has_pose = pose is not None
    print("===== 检测到画面，进入手势判断 =====")
    print(f"左手:{has_left}, 右手:{has_right}, 人脸:{has_face}, 姿态:{has_pose}")

    single_hand = right_hand if has_right else left_hand

    def pt(hand, idx):
        lm = hand.landmark[idx]
        return lm.x * img_w, lm.y * img_h

    def wrist(hand):
        return pt(hand, 0)

    def hand_h(hand):
        return hand.landmark[0].y * img_h

    def hand_x(hand):
        return hand.landmark[0].x * img_w

    def mid_tip(hand):
        return pt(hand, 12)

    def index_tip(hand):
        return pt(hand, 8)

    def pinky_tip(hand):
        return pt(hand, 20)

    def mid_mcp(hand):
        return pt(hand, 9)

    # ---------- 辅助：身体姿态判断 ----------
    def get_pose_pt(idx):
        """获取身体姿态关键点坐标。"""
        if not has_pose:
            return None, None
        lm = pose.landmark[idx]
        return lm.x * img_w, lm.y * img_h

    # 肩部关键点用于判断身体朝向和手臂相对位置
    l_shoulder = get_pose_pt(11) if has_pose else (None, None)
    r_shoulder = get_pose_pt(12) if has_pose else (None, None)
    l_elbow = get_pose_pt(13) if has_pose else (None, None)
    r_elbow = get_pose_pt(14) if has_pose else (None, None)
    l_hip = get_pose_pt(23) if has_pose else (None, None)
    r_hip = get_pose_pt(24) if has_pose else (None, None)

    # 肩部水平线高度（用于判断手臂相对身体的位置）
    shoulder_avg_y = None
    if l_shoulder[1] is not None and r_shoulder[1] is not None:
        shoulder_avg_y = (l_shoulder[1] + r_shoulder[1]) / 2
    elif l_shoulder[1] is not None:
        shoulder_avg_y = l_shoulder[1]
    elif r_shoulder[1] is not None:
        shoulder_avg_y = r_shoulder[1]

    # ---------- 双手姿势 ----------
    if has_left and has_right:
        l_wx, l_wy = wrist(left_hand)
        r_wx, r_wy = wrist(right_hand)
        l_mid_x, l_mid_y = mid_tip(left_hand)
        r_mid_x, r_mid_y = mid_tip(right_hand)
        l_mcp_x, l_mcp_y = mid_mcp(left_hand)
        r_mcp_x, r_mcp_y = mid_mcp(right_hand)
        span_wrist = abs(l_wx - r_wx)    # 手腕水平距离
        span_mid = abs(l_mid_x - r_mid_x)  # 中指尖水平距离
        mid_x = img_w / 2

        # 握拳检测（双手）
        def _is_fist(hand):
            mx, my = mid_tip(hand)
            wx_, wy_ = wrist(hand)
            mcp_x_, mcp_y_ = mid_mcp(hand)
            tip_dist = np.hypot(mx - wx_, my - wy_)
            mcp_dist = np.hypot(mcp_x_ - wx_, mcp_y_ - wy_)
            return tip_dist / (mcp_dist + 0.001) < 1.05
        l_fist = _is_fist(left_hand)
        r_fist = _is_fist(right_hand)
        both_fist = l_fist and r_fist

        # 1) warrior_stick：一手高举，另一只手平举 —— 区分度最高优先判断
        y_diff = abs(l_wy - r_wy)
        high_above_shoulder = shoulder_avg_y is None or min(l_wy, r_wy) < shoulder_avg_y - img_h * 0.05
        print(f"[warrior] highY={min(l_wy,r_wy):.1f}, lowY={max(l_wy,r_wy):.1f}, yDiff={y_diff:.1f}, highAbove={high_above_shoulder}")
        if y_diff > img_h * 0.12 and (high_above_shoulder or min(l_wy, r_wy) < img_h * 0.30):
            print("[MATCH] warrior_stick：一手高举、一手平举")
            return "warrior_stick"

        # 2) fly_1：一只手向前方伸展探出，另一只手向下垂落舒展
        #    一上一下的姿势（排除 warrior_stick 那种极端高差后）
        one_up_one_down = (l_wy < img_h * 0.45 and r_wy > img_h * 0.55) or (r_wy < img_h * 0.45 and l_wy > img_h * 0.55)
        y_gap = abs(l_wy - r_wy)
        print(f"[fly_1] yGap={y_gap:.1f}, upDown={one_up_one_down}, Lwy={l_wy:.1f}, Rwy={r_wy:.1f}")
        if one_up_one_down and y_gap > img_h * 0.08:
            print("[MATCH] fly_1：一手前伸、一手下垂")
            return "fly_1"

        # 3) fly_3：双手握拳、手臂向上抬起
        if both_fist and l_wy < img_h * 0.50 and r_wy < img_h * 0.50:
            print("[MATCH] fly_3：双手握拳上抬")
            return "fly_3"

        # 4) heshou_bodhisattvava：双臂向上弯曲抬起，双手靠近
        hands_close = span_wrist < img_w * 0.30
        both_upper = l_wy < img_h * 0.50 and r_wy < img_h * 0.50
        print(f"[heshou] L_wy={l_wy:.1f}, R_wy={r_wy:.1f}, upper={both_upper}, close={hands_close}")
        if both_upper and hands_close and not both_fist:
            print("[MATCH] heshou_bodhisattvava：双臂向上弯曲抬起")
            return "heshou_bodhisattvava"

        # 5) fly_2：单侧手臂朝一侧平伸张开，另一只手臂向后摆开
        #    放宽：只要两手水平距离较大、高度差不太大即可
        height_close = abs(l_wy - r_wy) < img_h * 0.30
        l_wide = l_wx < mid_x * 0.55
        r_wide = r_wx > mid_x * 1.45
        print(f"[fly_2] span={span_wrist:.1f}, hDiff={abs(l_wy-r_wy):.1f}, Lwide={l_wide}, Rwide={r_wide}")
        if span_wrist > img_w * 0.22 and height_close and (l_wide or r_wide):
            print("[MATCH] fly_2：单侧手臂平伸张开")
            return "fly_2"

        # 6) lady_robe：一手抬至头部旁，另一只手在腰侧（双手都在时）
        face_nose_y = face.landmark[1].y * img_h if has_face else img_h * 0.3
        l_near_head = abs(l_wy - face_nose_y) < img_h * 0.20
        r_near_head = abs(r_wy - face_nose_y) < img_h * 0.20
        l_near_waist = img_h * 0.48 < l_wy < img_h * 0.78
        r_near_waist = img_h * 0.48 < r_wy < img_h * 0.78
        head_and_waist = (l_near_head and r_near_waist) or (r_near_head and l_near_waist)
        print(f"[lady_robe] L_nearHead={l_near_head}, R_nearWaist={r_near_waist}, R_nearHead={r_near_head}, L_nearWaist={l_near_waist}")
        if head_and_waist:
            print("[MATCH] lady_robe：一手抬至头部旁、一手在腰侧")
            return "lady_robe"

        # 7) wing_spirit：前后双臂分别向前、向后张开
        #    放宽：双手分列两侧、水平距离够大
        l_side = l_wx < img_w * 0.42
        r_side = r_wx > img_w * 0.58
        print(f"[wing] span={span_wrist:.1f}, lSide={l_side}, rSide={r_side}")
        if span_wrist > img_w * 0.22 and l_side and r_side:
            print("[MATCH] wing_spirit：双臂前后张开")
            return "wing_spirit"

        # 8) green_lion：两只手下垂靠拢在身前
        l_y = hand_h(left_hand)
        r_y = hand_h(right_hand)
        wrist_dist = np.hypot(l_wx - r_wx, l_wy - r_wy)
        both_mid_low = l_y > img_h * 0.45 and r_y > img_h * 0.45
        hands_near = wrist_dist < img_w * 0.30
        print(f"[green_lion] lY={l_y:.1f}, rY={r_y:.1f}, dist={wrist_dist:.1f}, midLow={both_mid_low}, near={hands_near}")
        if both_mid_low and hands_near:
            print("[MATCH] green_lion：两手下垂靠拢身前")
            return "green_lion"

        # 9) lotus_hand：双手掌心朝上（两手都在时，要求两手手腕都低于 MCP）
        l_palm_up = l_wy > l_mcp_y + img_h * 0.015
        r_palm_up = r_wy > r_mcp_y + img_h * 0.015
        both_palm_up = l_palm_up and r_palm_up
        print(f"[lotus_dual] L_palmUp={l_palm_up}, R_palmUp={r_palm_up}")
        if both_palm_up:
            print("[MATCH] lotus_hand：双手掌心朝上")
            return "lotus_hand"

        # 双手兜底：根据双手位置分配不同图片，避免总出同一张
        # 双手在上部 → heshou 或 fly_3
        if l_wy < img_h * 0.48 and r_wy < img_h * 0.48:
            if both_fist:
                print("[MATCH] fly_3（双手兜底）：双手握拳上抬")
                return "fly_3"
            if hands_close:
                print("[MATCH] heshou_bodhisattvava（双手兜底）：双手在上靠近")
                return "heshou_bodhisattvava"
            print("[MATCH] fly_2（双手兜底）：双手在上方张开")
            return "fly_2"
        # 双手在下部
        if l_y > img_h * 0.55 and r_y > img_h * 0.55:
            if hands_near:
                print("[MATCH] green_lion（双手兜底）：双手下垂靠拢")
                return "green_lion"
            print("[MATCH] wing_spirit（双手兜底）：双手在下")
            return "wing_spirit"
        # 一上一下
        if (l_wy < img_h * 0.48) != (r_wy < img_h * 0.48):
            print("[MATCH] fly_1（双手兜底）：一手上一手下")
            return "fly_1"
        # 其余 → 轮流
        fallback_list = ["lady_robe", "wing_spirit", "fly_2", "heshou_bodhisattvava"]
        fb = fallback_list[hash(str((l_wy, r_wy, l_wx, r_wx))) % len(fallback_list)]
        print(f"[MATCH] {fb}（双手兜底轮转）")
        return fb

    # ---------- 单手姿势（仅当只有一只手时） ----------
    if single_hand is not None:
        wx, wy = wrist(single_hand)
        mid_x, mid_y = mid_tip(single_hand)
        idx_x, idx_y = index_tip(single_hand)
        pinky_x, pinky_y = pinky_tip(single_hand)
        mcp_x, mcp_y = mid_mcp(single_hand)

        finger_spread = abs(mid_x - pinky_x)
        tip_dist = np.hypot(mid_x - wx, mid_y - wy)
        mcp_dist = np.hypot(mcp_x - wx, mcp_y - wy)
        fist_ratio = tip_dist / (mcp_dist + 0.001)
        is_fist = fist_ratio < 1.05
        palm_up = wy > mcp_y + img_h * 0.02  # 手腕明显低于MCP才是掌心朝上

        # lotus_hand：掌心朝上，手指张开
        print(f"[lotus] palmUp={palm_up}, spread={finger_spread:.1f}, wy={wy:.1f}, mcpY={mcp_y:.1f}")
        if palm_up and finger_spread > img_w * 0.04:
            print("[MATCH] lotus_hand：掌心朝上")
            return "lotus_hand"

        # fly_3：手握拳上抬
        print(f"[fly_3] wy={wy:.1f}, fistRatio={fist_ratio:.2f}")
        if wy < img_h * 0.45 and is_fist:
            print("[MATCH] fly_3：手臂上抬握拳")
            return "fly_3"

        # fly_1 单手版：手在上方向前伸
        if wy < img_h * 0.42 and not is_fist:
            print("[MATCH] fly_1（单手）：一手向前伸展")
            return "fly_1"

        # fly_2 单手版：手臂向一侧平伸
        if abs(wx - img_w / 2) > img_w * 0.20 and abs(wy - img_h * 0.5) < img_h * 0.25 and not is_fist:
            print("[MATCH] fly_2（单手）：单侧手臂平伸")
            return "fly_2"

        # 单手兜底：按位置区域轮转分配
        if wy < img_h * 0.45:
            print("[MATCH] fly_1（单手兜底）：手在上方")
            return "fly_1"
        if wy > img_h * 0.58:
            print("[MATCH] green_lion（单手兜底）：手下垂")
            return "green_lion"
        if finger_spread > img_w * 0.05:
            print("[MATCH] lotus_hand（单手兜底）：手指张开")
            return "lotus_hand"
        print("[MATCH] lady_robe（单手兜底）：手在中间")
        return "lady_robe"

    # ---------- 纯人脸/身体姿势（无手时） ----------
    if has_face:
        # 侧头检测 → lady_robe
        nose_tip = face.landmark[1]
        left_eye_inner = face.landmark[133]
        right_eye_inner = face.landmark[362]
        eye_center_x = (left_eye_inner.x + right_eye_inner.x) / 2
        nose_offset = abs(nose_tip.x - eye_center_x)
        print(f"[lady_robe] noseOffset={nose_offset:.4f}")
        if nose_offset > 0.018:
            print("[MATCH] lady_robe：侧身头部偏斜")
            return "lady_robe"
        # 有脸无手兜底：轮转
        fallback_faces = ["lady_robe", "heshou_bodhisattvava", "lotus_hand"]
        fb = random.choice(fallback_faces)
        print(f"[MATCH] {fb}（纯人脸兜底轮转）")
        return fb

    # 无匹配
    print("[NO MATCH] 无匹配手势，画布保持空白")
    return None


def draw_poster_on_canvas(target_canvas, left_hand, right_hand, face, pose, canvas_w, canvas_h, forced_gesture=None):
    if forced_gesture:
        current_gest = forced_gesture
    else:
        current_gest = judge_gesture(left_hand, right_hand, face, pose, canvas_w, canvas_h)

    if current_gest is None:
        return None

    print(f"开始加载素材：{current_gest}")
    img_path = gesture_img_map[current_gest]
    dun_img = Image.open(img_path).convert("RGBA")
    img_array = np.array(dun_img)
    white_threshold = 180
    white_area = (img_array[..., 0] > white_threshold) & (img_array[..., 1] > white_threshold) & (img_array[..., 2] > white_threshold)
    img_array[white_area, 3] = 0
    dun_img = Image.fromarray(img_array)

    # 均衡缩放：根据图片原始比例，统一缩放到画布内合适大小，给上下左右留出空间
    orig_w, orig_h = dun_img.size
    # 目标区域：画布四周各留 8% 的边距
    margin = 0.08
    max_w = int(canvas_w * (1 - 2 * margin))
    max_h = int(canvas_h * (1 - 2 * margin))
    ratio = min(max_w / orig_w, max_h / orig_h)
    target_w = int(orig_w * ratio)
    target_h = int(orig_h * ratio)
    dun_img = dun_img.resize((target_w, target_h), Image.LANCZOS)

    img_arr = np.array(dun_img)
    rgba = img_arr.copy()  # RGBA
    rgba[..., :3] = rgba[..., :3][..., ::-1]  # RGB -> BGR
    alpha_f = rgba[..., 3:4] / 255.0

    # 居中贴图
    px = (canvas_w - target_w) // 2
    py = (canvas_h - target_h) // 2
    x1, y1 = px, py
    x2, y2 = px + target_w, py + target_h
    draw_w = target_w
    draw_h = target_h

    # 混合到 RGBA 画布（4 通道）
    roi = target_canvas[y1:y2, x1:x2].astype(np.float32)
    fg_bgr = rgba[:draw_h, :draw_w, :3].astype(np.float32)
    fg_a = alpha_f[:draw_h, :draw_w]
    roi[:, :, :3] = (fg_bgr * fg_a + roi[:, :, :3] * (1 - fg_a)).astype(np.uint8)
    roi[:, :, 3] = np.maximum(roi[:, :, 3], (fg_a[:, :, 0] * 255).astype(np.uint8))
    target_canvas[y1:y2, x1:x2] = roi.astype(np.uint8)
    print("[OK] 图片已绘制到画布中心！")
    return current_gest
