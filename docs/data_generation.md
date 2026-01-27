# 数据生成与导入指南

本文档介绍如何生成、保存、加载患者/到达/医生数据，便于复用或替换为外部数据集。

## 默认路径
- 目录：`data/`
- 文件：`patients.csv`, `doctors.csv`, `arrivals.csv`
- 当运行 `uv run dss ...` 时：
  - 若上述文件已存在且未指定 `--regen-data`，系统直接读取。
  - 若不存在或指定 `--regen-data`，系统按配置生成新数据并覆盖保存到该目录。

### 控制规模的超参数
- `years`：默认 2。
- `patients_per_year`：默认 30,000。当前生成逻辑会创建 `years * patients_per_year` 个唯一患者面板（默认 60,000 人），再按各自年就诊频率生成更高数量的到达记录。
- 通过 CLI 覆盖：
  ```bash
  uv run dss --years 2 --patients-per-year 40000 --regen-data
  ```
  或在代码中设置 `SimulationConfig(years=..., patients_per_year=...)`。

## 患者字段（patients.csv）
| 列名 | 说明 | 生成逻辑/分布 |
| --- | --- | --- |
| patient_id | 自增编号 | 顺序分配 |
| age_group | AD/MA/SE/EL | 权重 0.35/0.30/0.22/0.13 |
| gender | 性别 | 均匀 F/M |
| race | 种族 | 均匀自 `RACES` |
| region | 所在区域 | 均匀自 `REGIONS` |
| language | 语言 | 权重 0.72(en)/0.18(es)/0.10(zh) |
| visit_freq | 年就诊频率（期望值） | Lognormal(μ=0, σ=0.5) 截断下限 0.2 |
| specialty_request | 患者自报/真实需求专科 | 在 {general, internal, pediatrics} 抽样（第二年 pediatrics/internal 权重略升） |
| region_bias / language_bias / quality_bias | 偏好权重 | region U(0.35,0.9); language U(0.1,0.5); quality U(0.2,0.6) |
| gender_bias / race_bias / service_type_bias / service_count_bias | 其他偏好权重 | 各自 U(0.05, 0.3) |

## 医生字段（doctors.csv）
| 列名 | 说明 | 生成逻辑/分布 |
| --- | --- | --- |
| doctor_id | 自增编号 | 基线 + 追加招聘 |
| specialty | 专科 | 按 `base_doctor_counts` |
| region / language | 区域/语言 | 均匀抽样 |
| gender | 性别 | 均匀 F/M |
| age | 年龄 | N(45, 10) 截断 28–70 |
| race | 种族 | 均匀自 `RACES` |
| service_type | 服务形态 | group / solo 均匀 |
| services_count | 诊所内提供的专业服务数量 | randint 1–5 |
| quality_score | 质量评分 | U(0.55, 0.92) |
| daily_minutes | 每日可用分钟 | `doctor_daily_minutes` |
| hires_at | 生效日期 | 基线为空；补充招聘时写入 |

## 到达字段（arrivals.csv）
| 列名 | 说明 | 生成逻辑/分布 |
| --- | --- | --- |
| arrival_id | 自增编号 | 顺序分配 |
| patient_id | 对应患者 | 关联 `patients.csv` |
| arrival_date | 到达日 | 按患者年频率 * 季节性泊松抽样 |
| latest_date | 最晚接受日 | `N(14,5)` 截断到 `[3, max_wait_days]` |
| service_minutes | 需求时长 | 由患者的 `specialty_request` 映射 |
| specialty_request | 请求的专科 | 等同患者的 `specialty_request` |
| no_show_risk | 本次到访缺席基础概率 | 0.08 + U(0,0.1)，上限 0.4 |

## 生成与保存
```python
from dss.config import SimulationConfig
from dss.data_generation import generate_patients, generate_doctors, generate_arrivals, save_data

cfg = SimulationConfig()
patients = generate_patients(cfg)
doctors = generate_doctors(cfg)
arrivals, _ = generate_arrivals(cfg, patients)
save_data(patients, doctors, arrivals)  # 保存到默认 data/ 目录
```

## 手动加载外部数据
1. 准备 `patients.csv`、`doctors.csv`、`arrivals.csv`，列名需与上表一致。
2. 将文件放入目标目录（默认 `data/`）。
3. 运行模拟时可指定目录或直接使用默认目录：
   ```bash
   uv run dss --data-dir path/to/your/data
   # 如需忽略已存在文件并重生成：
   uv run dss --data-dir path/to/your/data --regen-data
   ```
4. 系统自动读取并转为内部对象；若列名或类型缺失将触发加载错误。

## 导出数据结构
运行模拟时，若需要重新生成数据：
- 患者文件：`data/patients.csv`
- 医生文件：`data/doctors.csv`
可直接被 `pandas.read_csv` 读取或导入至其他分析工具。
