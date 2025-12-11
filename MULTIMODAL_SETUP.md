# 多模态 RAG 设置指南

本指南说明如何在 RAG pipeline 中添加图片支持。

## 概述

我们使用 CLIP 模型处理图片，将 Flickr30K 数据集的图片编码为向量，并添加到 FAISS 索引中，实现文本和图片的统一检索。

## 步骤 1: 构建图片索引

首先，需要从 Flickr30K 数据集加载图片并构建 FAISS 索引：

```bash
python -m retriever.build_image_index
```

或者：

```bash
cd retriever
python build_image_index.py
```

这个脚本会：
1. 从 `nlphuji/flickr30k` 数据集加载 100 张图片（可在脚本中修改 `NUM_IMAGES`）
2. 使用 CLIP 模型（`openai/clip-vit-base-patch32`）编码图片
3. 构建 FAISS 索引并保存到 `retriever/faiss_index/images/`

输出文件：
- `index.faiss`: FAISS 索引文件
- `images.jsonl`: 图片元数据（ID、caption 等）
- `embeddings.npy`: 图片向量
- `metadata.json`: 索引元信息

## 步骤 2: 使用多模态检索器

### 方式 1: 在 RAG Pipeline 中使用（已集成）

`rag_pipeline.py` 已经更新为使用 `MultimodalRetriever`：

```bash
python rag_pipeline.py --query "a dog playing in the park" --top_k 5
```

这会同时检索文本和图片，并按相关性排序返回。

### 方式 2: 单独使用多模态检索器

```python
from retriever.multimodal_retriever import MultimodalRetriever, MultimodalRetrieverConfig

# 配置检索器
config = MultimodalRetrieverConfig(
    use_text_retrieval=True,   # 启用文本检索
    use_image_retrieval=True,  # 启用图片检索
    text_weight=0.6,           # 文本结果权重
    image_weight=0.4,          # 图片结果权重
)

retriever = MultimodalRetriever(config=config)

# 检索
results = retriever.retrieve("a beautiful sunset", top_k=10)

# 结果包含文本和图片
for item in results:
    if item["modality"] == "text":
        print(f"Text: {item['text'][:100]}...")
    elif item["modality"] == "image":
        print(f"Image: {item['image_id']} - {item['caption']}")
```

## 步骤 3: 测试完整流程

运行测试脚本：

```bash
python test_multimodal_rag.py
```

这会测试：
1. 多模态检索器的初始化
2. 文本和图片的混合检索
3. 生成器如何处理多模态结果

## 文件结构

新增的文件：

```
retriever/
├── image_encoder.py          # CLIP 图片编码器
├── build_image_index.py      # 构建图片索引的脚本
├── multimodal_retriever.py   # 多模态检索器（文本+图片）
└── faiss_index/
    └── images/               # 图片索引文件
        ├── index.faiss
        ├── images.jsonl
        ├── embeddings.npy
        └── metadata.json
```

修改的文件：

- `generator/rag_generator.py`: 更新 `build_prompt()` 以支持图片信息
- `rag_pipeline.py`: 使用 `MultimodalRetriever` 替代 `RAGRetriever`

## 技术细节

### CLIP 模型

- **图片编码器**: 使用 CLIP 的 vision encoder 将图片编码为 512 维向量（`clip-vit-base-patch32`）
- **文本编码器**: 使用 CLIP 的 text encoder 将查询编码为相同维度的向量
- **相似度计算**: 使用内积（Inner Product）计算文本和图片的相似度

### 检索策略

1. **独立检索**: 文本和图片分别检索 top-k 个结果
2. **加权合并**: 使用配置的权重合并结果
3. **统一排序**: 按加权分数排序返回最终结果

### 生成器集成

生成器会将检索到的图片信息（caption、image_id）添加到 prompt 中，让 LLM 知道有哪些相关图片可用。

## 注意事项

1. **依赖**: 需要安装 `transformers`、`torch`、`PIL`、`datasets` 等库
2. **CLIP 模型**: 首次运行会自动下载 CLIP 模型（约 150MB）
3. **内存**: 100 张图片的索引很小，但大规模使用时需要注意内存
4. **维度匹配**: CLIP 的文本和图片编码器维度相同，可以直接比较

## 扩展

要添加更多图片：

1. 修改 `build_image_index.py` 中的 `NUM_IMAGES` 参数
2. 或修改数据源（可以加载其他图片数据集）
3. 重新运行构建脚本

要调整检索权重：

修改 `MultimodalRetrieverConfig` 中的 `text_weight` 和 `image_weight`。

