# 身绘 ShenHui

> 以身为笔，以手绘敦煌 —— 基于计算机视觉的手势交互式敦煌数字艺术创作系统  
> Create Dunhuang-inspired digital artworks through body gestures and hand movements.

---

## 演示视频

> 左半屏为摄像头实时追踪画面，右半屏为敦煌数字画布合成结果。

项目演示视频：

https://github.com/qisuk0/dunhuangwork2/blob/main/demo.mp4

---

# 项目背景

敦煌壁画是中国古代丝绸之路艺术的重要组成部分，承载着丰富的历史文化价值与东方美学内涵。其中飞天、菩萨、瑞兽等艺术形象具有独特的艺术魅力。

然而，传统壁画创作通常需要专业绘画技能和复杂制作流程，普通用户难以参与其中。

**身绘（ShenHui）** 是一个基于计算机视觉与人机交互技术的敦煌数字艺术创作系统。

项目结合 **MediaPipe Holistic 人体关键点检测技术、OpenCV 实时图像处理以及多层图像合成算法**，实现通过身体姿态和手势动作生成敦煌风格数字艺术作品。

用户无需绘画基础，只需面对摄像头完成指定动作：

- 身体姿态 → 自动匹配敦煌人物形象
- 手势动作 → 控制飘带、纹路以及背景元素
- 实时交互 → 创作属于自己的敦煌数字作品

让身体成为画笔，让手势成为语言，实现传统文化与人工智能技术的融合创新。

---

# 核心功能

## 1. 实时人体姿态捕捉

基于 **MediaPipe Holistic**：

- 检测人体 33 个姿态关键点
- 检测双手 21 个关键点
- 检测面部 478 个关键点

实时获取用户动作信息，并在摄像头画面中显示人体骨架。

---

## 2. 敦煌人物姿态识别

系统预设 9 组敦煌艺术形象：

- 飞天
- 合手菩萨
- 金刚力士
- 青狮瑞兽
- 双翼神兽
- 天女等

用户摆出对应姿态后，系统自动识别动作并匹配对应敦煌壁画素材。

---

## 3. 手势驱动数字创作

支持多种交互方式：

| 手势 | 功能 |
|---|---|
| ☝️ 食指移动 | 绘制敦煌岩彩风格飞天飘带 |
| 👋 面部区域挥手 | 添加壁画纹路元素 |
| ✊ 握拳 → 张开 | 添加背景装饰元素 |

---

## 4. 多层数字绘制与合成

采用多层 RGBA 图像合成技术：

```
背景层
   ↓
敦煌人物层
   ↓
背景元素层
   ↓
飘带绘制层
   ↓
最终作品输出
```

支持：

- 图层叠加
- Alpha 透明混合
- 素材智能避让
- 实时预览
- 作品保存

---

# 系统架构

```
Camera Input

        │

        ▼

MediaPipe Holistic

(Pose / Hand / Face Landmarks)

        │

        ▼

Gesture Recognition Engine

        │

        ├── gesture_config.py
        │       人物姿态识别与素材匹配
        │
        ├── ribbon_config.py
        │       飘带轨迹绘制系统
        │
        └── bjwy_config.py
                背景元素触发系统


        │

        ▼

Painting Engine

        │

        ├── Character Layer
        │
        ├── Background Layer
        │
        ├── Decoration Layer
        │
        └── Ribbon Layer


        │

        ▼

Dunhuang Artwork Output
```

---

# 安装运行

## 环境要求

- Python 3.10+
- 摄像头设备
- 推荐使用 Python 虚拟环境运行


## 1. 克隆项目

```bash
git clone https://github.com/qisuk0/dunhuangwork2.git

cd dunhuangwork2
```


## 2. 创建虚拟环境（推荐）

```bash
python -m venv .venv
```


### Windows

```bash
.venv\Scripts\activate
```


### macOS / Linux

```bash
source .venv/bin/activate
```


## 3. 安装依赖

```bash
pip install -r requirements.txt
```


## 4. 运行项目

```bash
python main.py
```

---

# 使用说明

| 操作 | 功能 |
|-|-|
| 对摄像头摆出敦煌姿势 | 自动识别并生成对应人物 |
| 食指移动 | 绘制飞天飘带 |
| 面部上/下方挥手 | 添加纹路素材 |
| 握拳 → 张开 | 添加背景元素 |
| Esc | 退出程序 |
| c | 清空画布 |
| x | 清空飘带 |
| z | 撤销操作 |
| a / d | 调节背景色相 |
| q | 保存当前作品 |

---

# 手势参考

| 姿势名称 | 描述 | 对应素材 |
|-|-|-|
| fly_1 | 一手向前伸展，另一手向下舒展 | fly_1.tif |
| fly_2 | 单侧手臂平伸展开 | fly_2.tif |
| fly_3 | 双手握拳向上抬起 | fly_3.tif |
| warrior_stick | 一手高举，一手平举 | warrior_stick.tif |
| heshou_bodhisattva | 双手靠近合十姿态 | heshou_bodhisattva.tif |
| lotus_hand | 双手掌心向上 | lotus_hand.tif |
| lady_robe | 一手靠近头部，一手位于腰侧 | lady_robe.tif |
| green_lion | 双手靠近身体前方 | green_lion.tif |
| wing_spirit | 双臂前后展开 | wing_spirit.tif |

---

# 项目结构

```
dunhuangwork2/

├── main.py                  # 程序入口，主循环与渲染流程

├── gesture_config.py        # 手势识别与人物素材匹配

├── ribbon_config.py         # 飘带绘制系统

├── bjwy_config.py           # 背景元素管理


├── assets/                  # 敦煌人物素材

│   ├── fly_1.tif
│   ├── fly_2.tif
│   ├── fly_3.tif
│   └── ...


├── bjwy/                    # 背景纹路与装饰素材

├── posters/                 # 用户生成作品保存目录

├── tests/                   # 测试代码

├── requirements.txt         # Python依赖

├── .gitignore               # Git忽略配置

├── LICENSE                  # MIT许可证

└── README.md
```

---

# 技术栈

## Computer Vision

### MediaPipe Holistic

用于：

- 人体姿态检测
- 手部关键点检测
- 面部关键点检测


## Image Processing

### OpenCV

用于：

- 摄像头采集
- 实时渲染
- 图像处理
- 图层合成


### Pillow (PIL)

用于：

- TIFF 素材读取
- 图片透明处理
- 图像预处理


### NumPy

用于：

- 像素级运算
- Alpha 通道混合

---

# 算法实现

## Gesture Recognition

### 握拳检测

通过：

- 指尖与手腕距离
- MCP 与手腕距离比例

判断手部弯曲状态。


### 掌心方向检测

通过：

- 手腕坐标
- 掌指关节位置关系

判断手掌朝向。


### 挥手检测

通过：

- 历史轨迹累计位移
- 移动方向判断

识别挥手动作。


---

## 飘带系统

实现：

- 根据轨迹点计算切线方向
- 生成飘带左右边缘
- 线性插值平滑
- 半透明颜色混合

生成具有敦煌岩彩风格的动态飘带效果。

---

# 未来规划

- [ ] 支持更多敦煌人物姿态
- [ ] 增加声音交互反馈
- [ ] 支持高清作品导出
- [ ] 支持视频生成
- [ ] Web 前端部署
- [ ] 引入深度学习姿态分类模型

---

# License

本项目基于 MIT License 开源。

敦煌壁画素材版权归原作者所有，仅用于学习研究与艺术创作。

---

# Acknowledgments

- [MediaPipe](https://github.com/google-ai-edge/mediapipe)  
  Google 开源跨平台机器学习解决方案

- 敦煌壁画艺术  
  中华传统文化的重要组成部分，为本项目提供创作灵感
