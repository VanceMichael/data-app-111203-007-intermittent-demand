# intermittent-demand

面向补货的间歇需求分析。`summary` 按产品族汇总销量、可售小时和销售额；`backtest` 做逐 SKU、按 lead_days 窗口的滚动回测，把「连续零销量」拆成**没有销售暴露**与**真正零需求**，并按产品族的缺货成本与持有成本选择补货分位数。

分析不会把断货期间的丢失销量「恢复」成真实订单，只给出可解释的下界、模型敏感上界与敏感性区间。

## 使用

运行需要 Python 3.11 或更新版本，没有第三方运行依赖。数据在本地处理，不请求外部接口。

```bash
python3 -m analysis_app summary
python3 -m analysis_app summary --input data/observations.csv --output output/summary.json
python3 -m analysis_app backtest --input data/observations.csv \
    --config config/replenishment.json --output output/backtest.json
```

标准输出是完整 JSON，输出文件使用临时文件原子替换；失败退出码为 2，不覆盖既有结果。空数据集保留零计数与空分组，缺失数值不进入数值分母。数据指纹由按 observation_id 排序后的解析记录计算。

回测的成本参数与阈值**只从配置读入**，并原样写回结果，附带 `config_fingerprint` 以便复现。缺货成本、持有成本可写成标量、产品族映射、SKU 映射或含 `*` 的兜底映射。

## 回测口径

- **逐起点**：对每个 SKU、每个有足够历史的起点各产生一条预测；训练只用 `day <= 起点` 的记录。
- **按 lead_days 评估**：每个起点只在自身 `lead_days` 窗口上评分，不是统一下一天。窗口超出可用日历的起点不生成。
- **可售分类**：窗口内每天分为完整可售（24h）、部分可售（0<h<24）、关闭（0h）、缺失。
  - 全部完整 → `exact`，可辨识，可进入成本选择；
  - 含部分可售 → `censored`，售出量只作下界，给上下界与敏感性；
  - 含关闭或缺失 → `unidentifiable`，不当作零需求，不进入成本选择。
- **模型**：Croston-SBA（按暴露单位计间隔）、暴露感知 hurdle 模型、暴露感知泊松基线。`available_hours=0` 的日子不贡献暴露，也不被算作零需求；售出量不会被按小时比例无条件放大。
- **族级选择**：按产品族、用决策起点之前**已完整结束**的可辨识窗口，比较缺货 + 持有成本（固定订货成本对单窗口数量选择恒定，不参与比较，仅回显），先对每个 SKU 取均值再对 SKU 等权平均，并设缺货率上限；金额总额只作展示，不用于加权，避免高价单品掩盖低价件的缺货率。
- **零事件 ≠ 零需求**：有暴露但无事件时，用 rule-of-three 给出发生率上界并标注为界，而不是给出确定的零。

输出包含逐起点预测、每次族决策及其候选、哪些窗口不可辨识、补货分位数、上下界与敏感性。缺货期需求只有区间，**没有**被声称为真实丢失订单。

## 验证与容器

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q analysis_app
docker build -t intermittent-demand .
docker run --rm intermittent-demand
```

容器内也可执行 `python -m unittest discover -s tests -v`。镜像构建阶段取得 Python 镜像，产品启动、测试和演示均无需下载。

## 代码与数据

- `analysis_app/io.py`：有类型的 CSV 读取、完整性校验及内容指纹。
- `analysis_app/schema.py`：字段类型、主体与聚合维度。
- `analysis_app/config.py`：成本与回测参数的校验、成本解析与指纹。
- `analysis_app/demand_models.py`：暴露口径、候选需求模型与分布工具。
- `analysis_app/backtest.py`：逐起点回测、窗口分类、族级成本选择与敏感性。
- `analysis_app/summary.py`：确定性的分组与数值口径。
- `analysis_app/cli.py`：本地命令及原子输出。
- `docs/data.md`：数据契约。

演示数据全部为本地合成，不对应现实机构、设备或人员；不能用来推断真实业务效果。
