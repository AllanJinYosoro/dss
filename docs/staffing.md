# 医生招聘算法（Staffing / Supplementation）

目标：在季度层面监控饱和度并补充对应专科医生。

## 输入
- `quarter_turnaways[specialty]`：该季度未能安排的到达量
- `quarter_bookings[specialty]`：该季度尝试分配的到达量
- 配置：`doctor_daily_minutes`

## 触发规则
- 过载比率 `overload = turnaways / bookings` 大于 0.08 **或** `bookings > 800`（硬阈值）。
- 满足条件则为该专科新增 1 名医生。

## 新医生特征
- gender：交替 F/M
- age：42
- race：other
- service_type：group
- services_count：3
- region / language：按 doctor_id 轮转从 `REGIONS` / `LANGUAGES` 取值
- daily_minutes：`doctor_daily_minutes`
- hires_at：当前季度切换时刻（即新医生仅在后续到达中可用）。

## 输出
- 补充医生列表，并立即加入在岗队列（从 `hires_at` 当日起可调度）。
