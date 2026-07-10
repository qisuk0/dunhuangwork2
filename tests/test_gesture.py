"""手势识别、飘带、装饰元素功能测试。"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from gesture_config import gesture_img_map, judge_gesture
from ribbon_config import DUNHUANG_COLORS


# ========== 手势配置测试 ==========

def test_gesture_map_has_all_entries():
    """验证手势映射表包含所有 9 组敦煌形象。"""
    expected = [
        "fly_1", "fly_2", "fly_3",
        "warrior_stick", "heshou_bodhisattvava",
        "lotus_hand", "lady_robe",
        "green_lion", "wing_spirit",
    ]
    for gesture in expected:
        assert gesture in gesture_img_map, f"缺失手势: {gesture}"
        path = gesture_img_map[gesture]
        assert os.path.exists(path), f"素材文件不存在: {path}"


def test_gesture_count():
    """验证手势映射表恰好 9 组。"""
    assert len(gesture_img_map) == 9, f"期望 9 组，实际 {len(gesture_img_map)} 组"


def test_all_assets_loadable():
    """验证所有 assets 素材文件可被 PIL 正常打开。"""
    from PIL import Image
    for name, path in gesture_img_map.items():
        try:
            img = Image.open(path)
            assert img.size[0] > 0 and img.size[1] > 0, f"{name} 尺寸异常: {img.size}"
            img.close()
        except Exception as e:
            assert False, f"无法加载素材 {name} ({path}): {e}"


def test_judge_gesture_no_hands():
    """无手无脸时返回 None。"""
    result = judge_gesture(None, None, None, None, 640, 540)
    assert result is None, f"无输入时应返回 None，实际: {result}"


# ========== 飘带测试 ==========

def test_dunhuang_colors_count():
    """验证飘带敦煌色为 10 种。"""
    assert len(DUNHUANG_COLORS) == 10, f"期望 10 种颜色，实际 {len(DUNHUANG_COLORS)} 种"


def test_dunhuang_colors_valid_bgr():
    """验证所有颜色都是合法的 BGR 值。"""
    for color in DUNHUANG_COLORS:
        assert len(color) == 3, f"颜色格式错误: {color}"
        for c in color:
            assert isinstance(c, int), f"颜色值不是整数: {c}"
            assert 0 <= c <= 255, f"颜色值超出范围: {c}"


def test_dunhuang_colors_no_duplicate():
    """验证 10 种颜色无重复。"""
    seen = set()
    for color in DUNHUANG_COLORS:
        t = tuple(color)
        assert t not in seen, f"重复颜色: {t}"
        seen.add(t)


# ========== 装饰素材测试 ==========

def test_decoration_directory_exists():
    """验证装饰素材目录存在。"""
    assert os.path.isdir("decoration"), "decoration 目录不存在"


def test_decoration_has_tif_files():
    """验证 decoration 目录中有 tif 素材文件。"""
    files = [f for f in os.listdir("decoration") if f.endswith(".tif")]
    assert len(files) > 0, "decoration 目录中没有 tif 素材"


def test_assets_directory_has_tif_files():
    """验证 assets 目录中有 9 个 tif 文件。"""
    files = [f for f in os.listdir("assets") if f.endswith(".tif")]
    assert len(files) == 9, f"assets 目录应有 9 个 tif，实际 {len(files)} 个"


# ========== 版本信息测试 ==========

def test_requirements_txt_exists():
    """验证 requirements.txt 存在且非空。"""
    assert os.path.exists("requirements.txt"), "requirements.txt 不存在"
    with open("requirements.txt") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    assert len(lines) >= 2, f"依赖项过少: {len(lines)} 项"


def test_readme_exists():
    """验证 README.md 存在且包含关键章节。"""
    assert os.path.exists("README.md"), "README.md 不存在"
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    required_sections = ["# 身绘", "Demo", "Features", "Installation", "Usage", "License"]
    for section in required_sections:
        assert section.lower() in content.lower(), f"README 缺少章节: {section}"


def test_license_exists():
    """验证 LICENSE 文件存在且为 MIT。"""
    assert os.path.exists("LICENSE"), "LICENSE 不存在"
    with open("LICENSE", encoding="utf-8") as f:
        content = f.read()
    assert "MIT" in content, "LICENSE 不是 MIT 许可证"


if __name__ == "__main__":
    import traceback
    tests = [
        ("test_gesture_map_has_all_entries", test_gesture_map_has_all_entries),
        ("test_gesture_count", test_gesture_count),
        ("test_all_assets_loadable", test_all_assets_loadable),
        ("test_judge_gesture_no_hands", test_judge_gesture_no_hands),
        ("test_dunhuang_colors_count", test_dunhuang_colors_count),
        ("test_dunhuang_colors_valid_bgr", test_dunhuang_colors_valid_bgr),
        ("test_dunhuang_colors_no_duplicate", test_dunhuang_colors_no_duplicate),
        ("test_decoration_directory_exists", test_decoration_directory_exists),
        ("test_decoration_has_tif_files", test_decoration_has_tif_files),
        ("test_assets_directory_has_tif_files", test_assets_directory_has_tif_files),
        ("test_requirements_txt_exists", test_requirements_txt_exists),
        ("test_readme_exists", test_readme_exists),
        ("test_license_exists", test_license_exists),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
    if passed < len(tests):
        sys.exit(1)
