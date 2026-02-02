# 数据生成说明（基于《数据说明.html》）

本文件描述当前模拟生成的三份核心数据：`patients.csv`、`doctors.csv`、`arrivals.csv`。字段、分布和生成逻辑与《数据说明.html》保持一致，并已在 `data_generation_nb.py`（由 `data generation.ipynb` 导出）中实现。

## 总览
- 生成规模：默认 2 年、约 60,000 名患者，≈100,000+ 到达记录，40 名医生（基于基线配置）。
- 起始日期：`2023-01-01`（Notebook 默认；可在 `SimulationConfig` 中调整）。
- 最大可接受等待天数：30 天。
- 医生日工作容量：360 分钟。

## patients.csv（患者静态信息）

| 字段 | 类型 | 说明 | 生成/分布 |
| --- | --- | --- | --- |
| patient_id | int | 唯一 ID | 自增 1~60000 |
| age | int | 实际年龄 | AD:18-39, MA:40-59, SE:60-74, EL:75-99 |
| age_group | str | 年龄段 | AD 35%, MA 30%, SE 22%, EL 13% |
| gender | str | 性别 | M/F 各 50% |
| race | str | 种族 | White/Black/Asian/Hispanic/Other 等概率 |
| region | str | 区域 | North/South/East/West/Central 等概率 |
| language | str | 语言 | en 72%, es 18%, zh 10% |
| historical_visits | float | 过去就诊次数 | AD 1–5, MA 3–8, SE 6–15, EL 8–20（性别 1.2x 调节 + 随机扰动） |
| expected_visits_per_year | float | 预期年访问次数 | 来自 16 组 `class_code` 对应的 cp 值 |
| class_code | str | 患者分组 | 16 组：性别(M/F) × 年龄段(AD/MA/SE/EL) × 访问频率(V1/V2) |
| specialty_request | str | 请求专科 | family_practice / internal_medicine / pediatrics |
| service_minutes | int | 单次所需分钟 | FP:15–25, IM:20–30, PD:15–20 |

### 患者偏好字段
| 字段 | 取值范围 | 含义 |
| --- | --- | --- |
| region_bias | 0.35–0.9 | 区域偏好权重 |
| language_bias | 0.1–0.5 | 语言偏好权重 |
| quality_bias | 0.2–0.6 | 质量偏好权重 |
| gender_bias | 0.05–0.25 | 医生性别偏好 |
| race_bias | 0.05–0.25 | 医生种族偏好 |
| service_type_bias | 0.05–0.3 | 服务形态偏好（solo/group） |
| service_count_bias | 0.05–0.3 | 诊所提供服务数量偏好 |
| wait_time_bias | 0.3–0.7 | 等候时间偏好 |
| experience_bias | 0.2–0.6 | 医生经验偏好 |

## doctors.csv（医生信息）

| 字段 | 类型 | 说明 | 生成/分布 |
| --- | --- | --- | --- |
| doctor_id | int | 唯一 ID | 自增 |
| specialty | str | 专科 | family_practice / internal_medicine / pediatrics |
| region | str | 区域 | North/South/East/West/Central |
| language | str | 语言 | en/es/zh 等概率 |
| quality_score | float | 质量分 | 0.55–1.0；IM +0.05，PD +0.03 调节 |
| daily_minutes | int | 日可用分钟 | 360 |
| gender | str | M/F 各 50% |
| age | int | 30–65 |
| race | str | White/Black/Asian/Hispanic/Other |
| service_type | str | solo/group 等概率 |
| services_count | int | 1–5 |
| experience_years | int | 1–40 |
| board_certified | bool | 90% 为 true |
| hires_at | date | 招聘生效日 | 基线为空，增补时写入 |
| current_panel_size | int | 面板量 | 初始 0 |
| expected_workload | float | 预估工作量 | 初始 0.0 |

## arrivals.csv（到达记录）

| 字段 | 类型 | 说明 | 生成/分布 |
| --- | --- | --- | --- |
| arrival_id | int | 唯一到达 ID | 自增 |
| patient_id | int | 对应患者 | 关联 patients.csv |
| arrival_date | date | 到达日 | 患者来电日期（带季节性与冲击） |
| latest_date | date | 最晚接受日 | arrival_date + N(14,5)，截断 [3,30] |
| service_minutes | int | 本次需求时长 | 继承患者 service_minutes |
| specialty_request | str | 请求专科 | 继承患者 specialty_request |
| no_show_risk | float | 缺席风险 | 基线 0.08，按年龄/季节等扰动 |
| patient_class | str | 患者分组 | 同 patients 的 class_code |
| expected_visits | float | 预期年访问次数 | 同 patients 的 expected_visits_per_year |

## 生成流程摘要
1. **患者生成**：按年龄段、性别、种族、区域、语言分布采样；计算历史/预期访问次数并确定 `class_code`；生成偏好权重与专科需求、服务时长。
2. **医生生成**：按基线数量及分布生成三类医生，含质量分、经验等。
3. **到达生成**：基于患者的预期访问频率与季节性（含二年需求冲击：上半年 family_practice 升高，下半年全专科升高）做泊松采样；为每次到达生成最晚接受日和缺席风险。
4. **数据落盘**：输出至项目根 `data/` 目录的 `patients.csv` / `doctors.csv` / `arrivals.csv`。

## 代码位置与使用
- Notebook 源：`data generation.ipynb`
- 导出脚本：`src/dss/data_generation_nb.py`（由 nbconvert 生成，可直接导入复用其中函数/常量）。
- 运行再生成示例：
  ```bash
  uv run python -m dss.data_generation_nb  # 如需自写入口，可在脚本末尾添加 main
  ```
  或继续使用项目内的 CLI：`uv run dss --regen-data`（沿用现有 pipeline）。

