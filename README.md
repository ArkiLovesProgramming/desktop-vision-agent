# GUI Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

基于阿里云 Qwen-VL 多模态大模型的桌面自动化代理。通过视觉 - 语言模型实现"看 - 思考 - 行动"的智能交互循环。

## 功能特点

- **视觉感知**: 通过屏幕截图"观察"当前电脑状态，支持 10x10 网格坐标定位
- **智能决策**: 调用 Qwen-VL 模型进行推理和决策
- **精准操作**: 通过 PyAutoGUI 实际操作鼠标和键盘
- **ReAct 循环**: 持续执行"感知 - 思考 - 行动"循环直到任务完成
- **相对坐标**: 使用 0.0-1.0 相对坐标，跨分辨率兼容
- **多显示器支持**: 自动处理多屏幕偏移计算
- **DPI 感知**: 正确处理逻辑分辨率与物理分辨率
- **安全保护**: PyAutoGUI 故障安全机制，紧急情况下可快速停止

## 项目架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           GUI Agent Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  Perception  │───▶│    Brain     │───▶│    Action    │              │
│  │              │    │              │    │              │              │
│  │  ScreenCap   │    │  QwenClient  │    │  Executor    │              │
│  │  + Grid      │    │  + Prompt    │    │  + PyAutoGUI │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                       │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  Screenshot  │    │  JSON Parse  │    │ CLICK/TYPE   │              │
│  │  Base64 Enc  │    │  Validation  │    │ SCROLL/HOTKEY│              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          ┌──────────────────┐
                          │   ReAct Loop     │
                          │  until DONE      │
                          └──────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，并填入你的配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```ini
DASHSCOPE_API_KEY=sk-your-api-key-here
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen3-vl-plus
```

**获取 API Key 的步骤：**
1. 登录 [阿里云百炼控制台](https://bailian.console.aliyun.com/)
2. 进入 API-KEY 管理
3. 创建新的 API Key

### 3. 运行 Agent

```bash
python gui_agent.py
```

然后根据提示输入任务指令，例如：
- "打开记事本"
- "在浏览器中搜索 Python"
- "最小化所有窗口"

## 使用示例

### 基础示例

```python
# 打开计算器
"打开 Windows 计算器"

# 网页搜索
"在 Google 上搜索 Python 教程"

# 文本输入
"在记事本中输入'Hello, World!'"

# 窗口操作
"最小化所有窗口"
```

### 高级示例

```python
# 多步骤任务
"打开浏览器，访问 GitHub，登录我的账户"

# 文件操作
"打开文件资源管理器，新建一个文件夹"

# 系统控制
"调整系统音量到 50%"

# 表单填写
"在表单中输入姓名、邮箱和消息，然后点击提交按钮"
```

### 编程辅助示例

```python
# 代码执行
"打开 VS Code，创建一个新的 Python 文件"

# 调试辅助
"运行当前 Python 脚本并查看输出"
```

## 支持的动作类型

| 动作 | 说明 | 参数 | 示例 |
|------|------|------|------|
| `CLICK` | 点击指定坐标 | x, y (0.0-1.0 相对坐标) | `CLICK(0.5, 0.5)` - 点击屏幕中心 |
| `TYPE` | 键盘输入文本 | text (要输入的字符串) | `TYPE("Hello World")` |
| `SCROLL` | 滚动鼠标滚轮 | scroll_amount (负数向下，正数向上) | `SCROLL(-100)` - 向下滚动 |
| `HOTKEY` | 键盘快捷键 | keys (快捷键组合) | `HOTKEY("ctrl+c")` - 复制 |
| `DONE` | 任务完成 | 无 | `DONE` - 结束任务 |

## 配置选项

### 环境变量

| 环境变量 | 说明 | 默认值 | 必填 |
|----------|------|--------|------|
| `DASHSCOPE_API_KEY` | 阿里云 API Key | - | 是 |
| `BASE_URL` | API 端点 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 否 |
| `MODEL_NAME` | 模型名称 | `qwen3-vl-plus` | 否 |
| `MAX_ITERATIONS` | Maximum iterations per task | `50` | No |
| `TEMPERATURE` | 模型温度 | `0.7` | 否 |

### 可选模型

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| `qwen3-vl-plus` | 平衡性能和成本 | 日常使用，推荐 |
| `qwen3-vl-flash` | 快速响应 | 需要快速反馈的场景 |
| `qwen-vl-max` | 最强性能 | 复杂任务，高精度要求 |

## 项目结构

```
GUI-agent/
├── gui_agent.py          # 主程序入口
├── cli.py                # CLI 交互模块 (美化终端输出)
├── config.py             # 配置管理模块
├── requirements.txt      # Python 依赖
├── LICENSE               # MIT 许可证
├── CONTRIBUTING.md       # 贡献指南
├── .env.example          # 配置模板
├── .gitignore            # Git 忽略规则
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── pull_request_template.md
├── examples/             # 示例脚本目录
└── test-archive/         # 测试套件
    ├── test_agent.py
    ├── test_coordinate.py
    ├── test_integration.py
    └── TEST_REPORT.md
```

## 测试

运行测试套件：

```bash
python -m pytest test-archive/
```

查看测试报告：

```bash
cat test-archive/TEST_REPORT.md
```

### 测试结果

最近测试日期：2026-02-23

- 坐标变换测试：5/5 通过
- 集成测试：4/4 通过
- 优化功能测试：全部通过

## 日志输出

程序会输出详细的执行日志：

```
2026-02-23 10:00:00 - INFO - GUI Agent 启动
2026-02-23 10:00:00 - INFO - 任务目标：打开记事本
2026-02-23 10:00:01 - INFO - 正在截取屏幕...
2026-02-23 10:00:01 - INFO - 屏幕分辨率：1920x1080
2026-02-23 10:00:01 - INFO - 正在调用 Qwen-VL 模型...
2026-02-23 10:00:05 - INFO - [思考] 我看到当前桌面，任务栏在底部。要打开记事本，需要点击开始菜单...
2026-02-23 10:00:05 - INFO - [动作] CLICK
2026-02-23 10:00:05 - INFO - 正在执行动作...
2026-02-23 10:00:05 - INFO - 点击坐标：(100, 1050)
```

## 安全提示

**PyAutoGUI 内置 FailSafe 机制：** 将鼠标快速移到屏幕四个角落之一可以紧急停止程序。

> **重要提醒：**
> - 始终监控 Agent 的操作
> - 不要在敏感数据或关键系统上运行
> - 先在安全环境中测试
> - 随时准备按 Ctrl+C 中断

## 限制与注意事项

1. **坐标定位精度**: 模型基于视觉定位估算坐标，可能存在偏差
2. **循环次数限制**: 默认最多 15 次迭代，防止失控
3. **Base64 图像**: 使用 Data URI 格式发送图像，如遇到问题可考虑改用临时文件上传方式
4. **语言支持**: 模型对中文和英文指令都支持良好
5. **屏幕分辨率**: 支持多分辨率，使用相对坐标确保兼容性

## 故障排除

### API 连接失败
- 检查 `.env` 文件中的 API Key 是否正确
- 确认网络连接正常
- 检查阿里云账户余额

### JSON 解析失败
- 模型可能返回非标准 JSON 格式
- 代码已内置多种解析策略，但极端情况仍可能失败

### 动作执行不准确
- 在 Prompt 中已强调屏幕分辨率信息
- 如问题持续，可尝试更详细的任务描述

### 模型响应慢
- 检查网络连接
- 尝试使用 `qwen3-vl-flash` 模型获取更快响应
- 降低截图质量配置

## 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解：
- 代码规范
- PR 流程
- 测试要求
- Issue 和 PR 模板

### 贡献者

<!-- You can add contributors list here -->

## 开发说明

### 添加新动作类型

1. 在 `ActionType` 枚举中添加新类型
2. 在 `ActionExecutor` 中添加对应的执行方法
3. 更新 `SYSTEM_PROMPT` 说明新动作

### 修改 System Prompt

`SYSTEM_PROMPT` 变量定义了模型的角色和输出格式要求，可根据实际需求调整。

### 配置开发环境

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/gui-agent.git
cd gui-agent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 运行测试
python -m pytest test-archive/
```

## 相关链接

- [阿里云百炼控制台](https://bailian.console.aliyun.com/)
- [Qwen-VL 文档](https://help.aliyun.com/zh/dashscope/)
- [PyAutoGUI 文档](https://pyautogui.readthedocs.io/)
- [GitHub 仓库](https://github.com/YOUR_USERNAME/gui-agent)

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

<div align="center">
  <p>Made with ❤️ using Qwen-VL Vision-Language Models</p>
</div>
