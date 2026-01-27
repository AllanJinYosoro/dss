# DSS 模块说明

本文档详细说明模拟中各核心模块的角色、输入输出与关键假设，方便进一步调优或与论文 Case 5 的设定对照。

## 数据生成（`data_generation.py`）
- **医生基线**：三类专科 `general/internal/pediatrics`，随机地域/语言/性别/种族/质量分/服务类型（group/solo）与提供的专业服务数量；日工作上限 `doctor_daily_minutes`。
- **患者面板**：生成固定面板，属性含年龄段（AD/MA/SE/EL）、性别、种族、地域、语言、历史年就诊频率、专科请求（即真实需求），以及偏好权重（地域/语言/质量/性别/种族/服务类型/服务数量）。
- **到达序列**：对每位患者按其年就诊频率 + 季节性做泊松抽样，得到多次到达；每次到达生成 `latest_date`、`service_minutes` 和 `no_show_risk`。结果存入 `arrivals.csv`，同一患者会多次出现。

## 患者偏好表示
- 偏好向量包含 `region_bias`、`language_bias`、`quality_bias`，范围各自随机。
- 排序医生时，将偏好分数与当天负载（越空闲越优）相加，再加入轻微噪声（`preference_noise`）。

## 分配模块（`allocation.py`）
- **Cp 估计**：逻辑函数结合 PCP 请求、急诊度、慢病数、年龄及季度偏置 `QuarterState.cp_bias`，输出 PCP 适配概率。
- **专科选择**：`Cp >= 0.5` 分配 PCP，否则使用 `patient.true_specialty`。
- **医生排序**：同专科医生按偏好得分 + 容量剩余度排序，返回优先列表。
- **季度校准**：`cp_bias_from_history` 用上一季度的 PCP 选取命中率修正偏置，防止长期漂移。

## 调度模块（`scheduling.py`）
- **超订策略**：按季度观测到的缺席率 `no_show_rate` 计算超订系数，限制在 `[overbook_floor, overbook_ceiling]`。
- **找位逻辑**：从患者到达日起，查找首个满足 “已排分钟 + 需求分钟 ≤ 日容量*(1+超订)” 的日期；若超出最晚可接受日则视为未分配。
- **结果**：生成 `Appointment`，包含等待天数与未分配原因。

## 医生补充（`staffing.py`）
- 每季度检查：按专科的 `turnaways / bookings` 或绝对量判断过载，超过阈值则新增对应专科医生（带统一质量分数，记录 `hires_at` 生效日期）。

## 仿真编排（`simulation.py`）
1. 生成患者序列与初始医生。
2. 逐患者：
   - 确定季度，必要时：更新 `cp_bias`、估计 `no_show_rate`、触发补充招聘、重置季度统计。
   - 选择专科 → 排序医生 → 调度。
   - 记录预约与缺席（基于患者风险与季度缺席率混合抽样）。
3. 结束后生成 `pandas.DataFrame` 明细，并计算 KPI：
   - `fill_rate`、`avg_wait_if_scheduled`、`no_show_rate`
   - `pcp_match_rate`（最终分配为 PCP 的比例）
   - `true_specialty_match`（最终分配与真实需求一致的比例）。

## 扩展建议
- 调整 `service_minutes`、`base_doctor_counts` 以贴合真实工作量。
- 用真实缺席率替换当前基于风险的简单混合模型。
- 增加日历层面的医生不在岗/假期，以测试鲁棒性。
- 将季度校准改为贝叶斯更新，降低噪声。
