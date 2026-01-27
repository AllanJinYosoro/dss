# DSS Appointment Simulator

基于 `uv` 的可重复模拟，用于复现/探索 Case 5 “Decision Support System for Allocating Appointment Times to Calling Patients at a Medical Facility” 的核心流程：患者分配、调度与医生补充。

## 快速开始
- 虚拟环境与依赖请用 `uv sync` 同步（见 `pyproject.toml`），无需手工创建。
- 运行两年模拟并显示概览图：
  ```bash
  uv run dss --years 2 --patients-per-year 30000 --plot
  ```
- 导出明细与静态图（无弹窗）：
  ```bash
  uv run dss --png-out results.png --csv-out appointments.csv --no-plot
  ```

## 模块概览
- `config.py`：默认参数（季节性、容量、手术时长、需求冲击等）。
- `data_generation.py`：带季节性与季度冲击的患者/医生合成数据。
- `allocation.py`：估计 `Cp`（PCP 适配度），排序医生偏好，季度校准。
- `scheduling.py`：考虑缺席率 `pr` 的最早可行日调度与轻量超订。
- `staffing.py`：季度评估超订/满额，按专科增补医生。
- `simulation.py`：端到端编排，输出 `pandas.DataFrame` 与关键指标。
- `visualize.py`：简单 Matplotlib 可视化（需求 vs. 分配、等待分布、缺席率等）。

## 场景假设
- 第一年度按基线需求运行。
- 第二年度：Q1–Q2 家庭医生需求抬升；Q3–Q4 全专科需求普遍升高。
- 患者含个人信息、PCP 请求、最晚接受日、偏好向量、缺席风险；医生含地域/语言/质量评分与每日可用时长。

## 结果输出
- 控制台表格：填充率、平均等待天数、缺席率、专科匹配率、真实专科匹配度。
- CSV/图表便于进一步分析或与论文情景对照。

## 设计文档
- 详见 `docs/architecture.md`：涵盖数据生成、患者偏好、分配、调度、医生补充和整体仿真流程。
- 数据生成/导入导出详见 `docs/data_generation.md`，包括字段、分布与示例代码。

## 仿真数据
- 默认在项目根 `data/` 下查找 `patients.csv`、`doctors.csv`、`arrivals.csv`。
- 若不存在或想强制重生成（默认 2 年、每年 30,000 到达基准，生成 60,000+ 唯一患者与更多到达）：
  ```bash
  uv run dss --regen-data
  ```
- 使用自定义规模或目录：
  ```bash
  uv run dss --years 2 --patients-per-year 40000 --data-dir path/to/data --regen-data
  ```
