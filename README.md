# 身绘（ShenHui）

> 以身为笔，以手绘敦煌 —— 手势交互生成敦煌壁画，零美术基础也能创作数字艺术

## Demo 🎬

> 左半屏为摄像头追踪画面，右半屏为敦煌画布合成结果。

[演示视频](https://github.com/qisuk0/dunhuangwork1/blob/main/demo.mp4)

## Features

- **实时姿态捕捉**：基于摄像头实时检测人体全身骨骼、双手 21 关键点及面部 478 关键点，并在画面中实时绘制识别骨架
- **9 组专属手势动作**：预设飞天、合手菩萨、金刚力士、青狮瑞兽、双翼神兽、天女等敦煌壁画形象，一一对应不同肢体姿势
- **自动识别匹配**：识别用户肢体动作后，自动在画布中心渲染对应的透明 TIFF 格式敦煌壁画素材
- **手势操控装饰**：伸出食指在空中绘制敦煌岩彩色系飞天飘带；面部上方/下方挥手贴纹路素材；握拳张开贴背景元素
- **画布交互操作**：支持画布清空、逐层撤销、一键保存作品，窗口实时预览，背景色相实时调节
- **多层合成渲染**：敦煌壁画底色 → 人物素材层 → 背景元素层 → 飘带层，多层 RGBA/BGR 混合，素材间智能避让

## System Architecture

```
Camera
   │
   ▼
MediaPipe Holistic
(Pose · Hand · Face Landmarks)
   │
   ▼
Gesture Recognition Engine
   │
   ├──▶ gesture_config.py   人物姿势匹配（9 组敦煌形象）
   ├──▶ ribbon_config.py    飘带笔刷追踪
   └──▶ bjwy_config.py      背景元素 & 纹路触发
   │
   ▼
Painting Engine (main.py)
   │
   ├──▶ 人物素材层（居中拼贴）
   ├──▶ 背景元素层（智能避让）
   └──▶ 飘带层（透明度混合）
   │
   ▼
Dunhuang Artwork Output
```

## Installation

### 环境要求

- **Python 3.10+**
- **摄像头**（用于实时手势识别）
- 推荐使用虚拟环境

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/qisuk0/dunhuangwork1.git
cd dunhuangwork1

# 创建并激活虚拟环境（推荐）
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行
python main.py
```

## Usage

| 操作 | 说明 |
|------|------|
| **摆姿势** | 对着摄像头摆出敦煌人物姿势，自动匹配并贴图 |
| **伸出食指** | 在空中移动绘制飞天飘带 |
| **面部上/下方挥手** | 贴 wenlu 纹路素材 |
| **握拳 → 张开** | 在张开位置贴背景元素 |
| **Esc** | 退出程序 |
| **c** | 清空画布 |
| **x** | 清空画笔（仅飘带） |
| **z** | 撤销（先撤销背景元素，再撤销飘带） |
| **a / d** | 调节背景色相（左减右加） |
| **q** | 保存画布截图 |

## Gesture Reference

| 姿势名称 | 说明 | 对应素材 |
|----------|------|----------|
| `fly_1` | 一手向前方伸展探出，另一只手向下垂落舒展 | `assets/fly_1.tif` |
| `fly_2` | 单侧手臂朝一侧平伸张开，另一只手臂向后摆开 | `assets/fly_2.tif` |
| `fly_3` | 双手握拳、手臂向上抬起 | `assets/fly_3.tif` |
| `warrior_stick` | 一手高举过头，另一只手平举 | `assets/warrior_stick.tif` |
| `heshou_bodhisattvava` | 双臂向上弯曲抬起，双手靠近 | `assets/heshou_bodhisattva.tif` |
| `lotus_hand` | 双手掌心朝上 | `assets/lotus_hand.tif` |
| `lady_robe` | 一手抬至头部旁，另一只手在腰侧 | `assets/lady_robe.tif` |
| `green_lion` | 两只手下垂靠拢在身前 | `assets/green_lion.tif` |
| `wing_spirit` | 前后双臂分别向前、向后张开 | `assets/wing_spirit.tif` |

## Project Structure

```
dunhuangwork1/
├── main.py                  # 程序入口，主循环与图层合成
├── gesture_config.py        # 手势识别与敦煌人物贴图逻辑
├── bjwy_config.py           # 背景元素管理（挥手纹路、握拳张开贴图）
├── ribbon_config.py         # 飞天飘带笔刷系统
├── assets/                  # 敦煌人物素材（9 张 TIF）
│   ├── fly_1.tif
│   ├── fly_2.tif
│   ├── fly_3.tif
│   ├── warrior_stick.tif
│   ├── heshou_bodhisattva.tif
│   ├── lotus_hand.tif
│   ├── lady_robe.tif
│   ├── green_lion.tif
│   └── wing_spirit.tif
├── bjwy/                    # 背景元素素材（35 张 TIF）
│   └── wenlu/               # 纹路类素材
├── posters/                 # 作品保存目录
├── tests/                   # 单元测试
├── requirements.txt         # 运行依赖
├── .gitignore               # Git 忽略规则
├── LICENSE                  # MIT 开源许可证
└── README.md
```

## Tech Stack

- **MediaPipe Holistic**：同时检测手部（21 关键点）、面部（478 关键点）和身体姿态（33 关键点），实现多维度手势识别
- **OpenCV**：实时摄像头采集、画面渲染、图层合成与混合
- **Pillow (PIL)**：TIF 素材加载与预处理（去白底、缩放、通道转换）
- **NumPy**：像素级图层混合运算（RGBA Alpha 混合）
- **手势识别算法**：
  - 握拳检测：指尖-手腕距离 / MCP-手腕距离 < 1.05
  - 掌心朝上检测：手腕 Y 坐标低于 MCP Y 坐标
  - 挥手检测：累积位移阈值 + 方向判断
  - 手指弯曲度：四指 curl ratio 平均值
- **飘带系统**：基于轨迹点的切线方向生成左右边缘，线性插值平滑，敦煌岩彩色系渲染，80% 透明度混合
- **图层架构**：背景层 → 敦煌人物层 → 背景元素层（避让人物） → 飘带层，多层 RGBA/BGR 合成

## Future Work

- [ ] 支持更多敦煌壁画人物姿势
- [ ] 增加背景音乐与音效反馈
- [ ] 导出高清大图 / 视频录制
- [ ] Web 前端部署，无需本地安装

## License

本项目基于 MIT 许可证开源。详见 [LICENSE](LICENSE) 文件。

敦煌壁画素材版权归原作者所有，仅用于学习与艺术创作目的。

## Acknowledgments

- [MediaPipe](https://github.com/google-ai-edge/mediapipe) - Google 的跨平台机器学习解决方案
- 敦煌壁画 - 中华文化瑰宝，灵感源泉
