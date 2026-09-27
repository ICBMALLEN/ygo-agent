# 遮牌预测器

这个工具从牌表学习：给定主卡组和额外卡组中随机展示的 1–10 张卡，分别预测仍未展示的主卡组和额外卡组构成。每张展示牌是一个独立卡片实例；Side Deck 不参与训练。训练和推理只使用 Python 标准库，不加载 OCGCore、`ygoenv` 或决斗环境。

## 数据来源与本地快照

- Meta 来源： [Yu-Gi-Oh! Meta 的 Top Decks 页面](https://www.yugiohmeta.com/top-decks)，筛选 OCG 格式和 2026 年 7 月 1 日生效的禁限卡表。筛选结果快照以 JSON 文件提供给导入器。
- 禁限卡表来源：[Yu-Gi-Oh! Meta 的 OCG 2026-07 禁限卡表说明](https://www.yugiohmeta.com/articles/news/july-1-2026/ocg-forbidden-list)，对应官方生效日期为 2026-07-01。
- 牌表数据归属：[Yu-Gi-Oh! Meta 服务条款](https://www.yugiohmeta.com/terms-of-service)。查阅的页面没有明确授予牌表数据再分发许可，因此原始筛选快照、导入结果、报告和模型都保存在被 `.gitignore` 排除的 `data/deck_prediction/`，不要把这些文件加入公开提交。

导入器接收本地 JSON 快照，不会抓取或复制网站内容。JSON 可是数组，也可用 `decks`、`results` 或 `items` 包一层。每条记录需要 `main` 和 `extra` 卡片列表；卡片可写成卡片密码、牌名、重复密码数组或 `{密码: 数量}` 映射。若行中含有 `format` / `banlist` 字段，导入器会排除非 OCG 或非 2026 年 7 月结果；字段缺失时，将文件视为已按指定筛选条件导出的结果。例子：

```json
{
  "format": "OCG",
  "banlist": "2026-07-01",
  "source_url": "https://www.yugiohmeta.com/top-decks",
  "decks": [
    {
      "id": "meta-deck-id",
      "name": "Example Deck",
      "main": [101, 101, 102],
      "extra": [103],
      "side": [104]
    }
  ]
}
```

## 导入牌表

在仓库根目录运行。Meta 的 JSON 文件是本机输入文件；命令可重复传入多个快照。

```bash
python3 -m ygoai.deck_prediction.import_decks \
  --meta-json data/deck_prediction/raw/ocg-july-2026.json \
  --cards-db assets/locale/en/cards.cdb
```

导入器还合并 `assets/deck/` 下的顶层 `.ydk` 牌组。若某个已跟踪的 `.ydk` 在工作区被删除，会从当前 Git `HEAD` 读取其内容，不会恢复或改写工作区文件；嵌套在 `unsupported/`、`unused/` 等目录中的牌组不会自动加入。Side Deck 会忽略。牌表按主卡组与额外卡组的卡片数量去重，牌序和 Side Deck 不参与比较。

本地 `cards.cdb` 用于卡片密码、牌名和 Token 校验。缺少映射或含 Token 的牌组会被排除，原因与卡片密码写入 `import-report.json`。维护资源后重跑导入即可重新评估这些记录。

默认输出为：

- `data/deck_prediction/decks.jsonl`：校验、去重后的牌组。
- `data/deck_prediction/import-report.json`：导入计数、去重项、缺失卡片和排除原因。
- `data/deck_prediction/source-metadata.json`：来源链接和筛选条件。

原始 JSON 快照也应放在 `data/deck_prediction/` 下，以确保不会被 Git 跟踪。

## 训练和推理

```bash
python3 -m ygoai.deck_prediction.train \
  --dataset data/deck_prediction/decks.jsonl \
  --output data/deck_prediction/model.json \
  --seed 0

python3 -m ygoai.deck_prediction.infer \
  --model data/deck_prediction/model.json \
  --main 101 102 \
  --extra 103
```

训练按唯一牌组划分训练集和验证集，再逐副等权生成遮牌样本。模型是带区域标签的监督近邻模型：训练阶段保存可见牌与剩余牌计数的对应关系；推理阶段根据可见牌相似度，对近邻样本中的剩余主卡组和额外卡组计数加权平均。验证集牌组 ID、拆分种子和计数 MAE 会写在 `model.json.metrics.json` 中。

后续策略可以直接使用 Python 接口：

```python
from ygoai.deck_prediction import DeckMaskPredictor

predictor = DeckMaskPredictor.load("data/deck_prediction/model.json")
expected_remaining = predictor.predict(
    observed_main=[101, 102],
    observed_extra=[103],
)
```

`expected_remaining` 的 `main` 和 `extra` 值分别是 `{卡片密码: 期望剩余数量}`。展示的副本已从监督目标中扣除；若牌组中有同名副本，模型仍可预测尚未展示的其他副本。

## 验证

```bash
python3 -m unittest discover -s tests -p 'test_deck_prediction.py'
```
