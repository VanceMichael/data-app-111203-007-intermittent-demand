# 数据契约

补货分析按产品族汇总销量、可售小时和销售额，保留逐日可售时长而不推算未满足需求。

主测量 `sold` 的单位为件，主体字段是 `sku`，核算维度是 `family`。`observation_id` 是文件内唯一观测编号，主体允许重复观测。数值只接受有限值；空单元格解析为 None，不能直接当成零。额外列不进入当前核算。

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `observation_id` | str | 必填 |
| `sku` | str | 必填 |
| `family` | str | 必填 |
| `day` | int | 允许空值，空值不是零 |
| `sold` | float | 必填 |
| `available_hours` | float | 允许空值，空值不是零 |
| `lead_days` | int | 允许空值，空值不是零 |
| `price` | float | 允许空值，空值不是零 |
| `promotion` | int | 允许空值，空值不是零 |

时间字段 day、week、tick、hour、minute、cycle、start_day、reported_day、known_at、visible_at、opened、observed 均以各字段名所述的相对时间单位表示整数；不同字段并不必然代表同一种时间。缺少的登记码与浓度等信息保持为空。示例只表达产品数据结构，不能代替实际调查资料。
