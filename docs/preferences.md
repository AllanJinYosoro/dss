# 患者偏好算法（Preferences）

目标：为同专科医生生成可比评分，体现患者个体偏好并兼顾当前负载；患者频次分组由数据生成阶段完成（visit_freq 仅 high/low）。

## 输入
- 患者属性：region, language, gender, race, specialty_request
- 偏好权重：region_bias, language_bias, quality_bias, gender_bias, race_bias, service_type_bias, service_count_bias
- 医生属性：region, language, gender, race, service_type, services_count, quality_score
- 医生日负载：当天已排分钟 / 日容量
- 配置：`preference_noise`（随机扰动幅度）

## 评分公式（逐医生）
```
score =
  region_bias * 1(region match)
+ language_bias * 1(language match)
+ quality_bias * quality_score
+ gender_bias * 1(gender match)
+ race_bias * 1(race match)
+ service_type_bias * 1(service_type == "group")
+ service_count_bias * (services_count / 5)
+ noise ~ U(-preference_noise, preference_noise)
+ capacity_term (0.2 * (1 - load_ratio))
```

## 排序与输出
- 对候选医生计算 score，按降序排序，得到调度用的优先列表。
- 不直接决定是否分配，只提供偏好顺序；后续由调度模块寻找最早可行日。
