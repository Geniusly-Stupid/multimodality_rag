# 批量测试使用指南

## 功能说明

`batch_test.py` 支持使用不同的检索模式测试查询，可以比较文本、图片和混合检索的效果。

## 使用方法

### 1. 测试所有模式（推荐）

这会依次测试 text_only、image_only 和 both 三种模式，并分别保存结果：

```bash
python batch_test.py --mode all
```

输出文件：
- `rag_results/batch_test_results_text_only.json` - 仅文本检索结果
- `rag_results/batch_test_results_image_only.json` - 仅图片检索结果
- `rag_results/batch_test_results_both.json` - 混合检索结果
- `rag_results/batch_test_results.json` - 所有结果的合并文件

### 2. 测试单个模式

```bash
# 只测试文本检索
python batch_test.py --mode text_only

# 只测试图片检索
python batch_test.py --mode image_only

# 只测试混合检索
python batch_test.py --mode both
```

### 3. 自定义参数

```bash
python batch_test.py \
  --mode all \
  --test_file data/test.txt \
  --output_file rag_results/my_results.json \
  --top_k 10
```

## 参数说明

- `--mode`: 检索模式
  - `text_only`: 只使用文本检索
  - `image_only`: 只使用图片检索
  - `both`: 文本和图片混合检索（text_weight=0.6, image_weight=0.4）
  - `all`: 测试所有三种模式（推荐）
  
- `--test_file`: 测试文件路径（默认：`data/test.txt`）
- `--output_file`: 输出文件路径（默认：`rag_results/batch_test_results.json`）
- `--top_k`: 每个查询检索的文档数量（默认：5）

## 结果格式

每个查询的结果包含：

```json
{
  "query_id": 1,
  "query": "查询内容",
  "ground_truth": "标准答案",
  "mode": "text_only",  // 或 "image_only" 或 "both"
  "retrieved_count": 5,
  "text_count": 5,
  "image_count": 0,
  "retrieved_chunks": [
    {
      "modality": "text",
      "score": 0.85,
      "text": "检索到的文本...",
      "source_file": "SI650-Week-01-Course_enriched"
    }
  ],
  "generated_answer": "生成的答案..."
}
```

## 比较不同模式

运行 `--mode all` 后，可以比较不同模式的效果：

```python
import json

# 加载结果
with open('rag_results/batch_test_results.json', 'r') as f:
    results = json.load(f)

# 按模式分组
text_results = [r for r in results if r['mode'] == 'text_only']
image_results = [r for r in results if r['mode'] == 'image_only']
both_results = [r for r in results if r['mode'] == 'both']

# 比较答案
for i in range(len(text_results)):
    print(f"\nQuery {i+1}:")
    print(f"  Text only: {text_results[i]['generated_answer'][:100]}...")
    print(f"  Image only: {image_results[i]['generated_answer'][:100]}...")
    print(f"  Both: {both_results[i]['generated_answer'][:100]}...")
```

## 注意事项

1. **图片检索可能较慢**：CLIP 模型加载需要时间
2. **混合检索**：默认文本权重 0.6，图片权重 0.4
3. **结果保存**：每个查询处理完后立即保存，避免崩溃丢失数据
