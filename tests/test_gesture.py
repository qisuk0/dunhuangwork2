"""手势识别基本功能测试。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_config import gesture_img_map


def test_gesture_map_has_all_entries():
    """验证手势映射表包含所有 9 组敦煌形象。"""
    expected_gestures = [
        "fly_1",
        "fly_2",
        "fly_3",
        "warrior_stick",
        "heshou_bodhisattvava",
        "lotus_hand",
        "lady_robe",
        "green_lion",
        "wing_spirit",
    ]
    for gesture in expected_gestures:
        assert gesture in gesture_img_map, f"缺失手势: {gesture}"
        path = gesture_img_map[gesture]
        assert os.path.exists(path), f"素材文件不存在: {path}"


def test_gesture_count():
    """验证手势映射表恰好 9 组。"""
    assert len(gesture_img_map) == 9, f"期望 9 组手势，实际 {len(gesture_img_map)} 组"


def test_ribbon_colors():
    """验证飘带颜色定义正常。"""
    from ribbon_config import DUNHUANG_COLORS
    assert len(DUNHUANG_COLORS) == 10, f"期望 10 种敦煌色，实际 {len(DUNHUANG_COLORS)} 种"
    for color in DUNHUANG_COLORS:
        assert len(color) == 3, "颜色应为 BGR 三元组"
        for c in color:
            assert 0 <= c <= 255, f"颜色值超出范围: {c}"


def test_bjwy_directory_exists():
    """验证背景元素素材目录存在。"""
    from bjwy_config import BJWY_DIR, WENLU_DIR
    assert os.path.isdir(BJWY_DIR), f"bjwy 目录不存在: {BJWY_DIR}"
    assert os.path.isdir(WENLU_DIR), f"wenlu 目录不存在: {WENLU_DIR}"
