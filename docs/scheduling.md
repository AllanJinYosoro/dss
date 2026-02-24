# 调度算法（Scheduling）

目标：对每次患者到达（arrival）寻找最早可行的预约日期，允许受控超订以吸收缺席风险。

## 输入
- `Arrival`：arrival_id, arrival_date, latest_date, service_minutes, specialty_request, no_show_risk
- 候选医生序列（同专科，已按偏好排序）
- `QuarterState.no_show_rate`：季度观测缺席率
- 配置：`doctor_daily_minutes`, `overbook_floor`, `overbook_ceiling`

## 核心逻辑
1. 确定医生尝试顺序  
   - 首次到达：使用分配模块产出的排序。
   - 后续到达：先尝试患者的主治医生，若其在最晚接受日之前无空档，再依次尝试其他候选医生。
2. 计算超订系数  
   `overbook = clip(no_show_rate*1.2, overbook_floor, overbook_ceiling)`
   - `no_show_rate` 为全局缺席率估计：初始取配置中的基线值，之后仅在每个季度结束时用该季度真实观测的缺席情况重新估计。
3. 从 `arrival_date` 起逐日检查至 `latest_date`：
   - 可用容量 = `daily_minutes * (1 + overbook)`
   - 若 `已排分钟 + service_minutes <= 可用容量`，选中该日。
4. 若找到日期：
   - 在医生日程中累加分钟
   - 记录 `scheduled_date` 与 `wait_days`
5. 若未找到：
   - 返回未分配预约，原因 “No capacity before latest acceptable date”

## 输出
`Appointment`：patient_id, arrival_id, doctor_id, specialty, scheduled_date, wait_days, allocated flag, reason, no_show (后续模拟抽样)。
