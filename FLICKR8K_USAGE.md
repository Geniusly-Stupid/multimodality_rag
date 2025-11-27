# Flickr8K 数据集使用说明

## 数据集结构

你的 Flickr8K 数据集已经正确放置在：
- **图片目录**: `data/flickr8k/Images/` (8091 张图片)
- **标注文件**: `data/flickr8k/captions.txt` (CSV 格式，每张图片约 5 条 captions)

## 配置

代码已经配置为使用 Flickr8K 数据集：

```python
# 在 retriever/build_image_index.py 中
USE_DUMMY_IMAGES = False
LOCAL_IMAGE_DIR = "data/flickr8k/Images"
CAPTIONS_FILE = "data/flickr8k/captions.txt"
NUM_IMAGES = 100  # 可以修改为需要的数量
```

## 使用方法

### 1. 测试图片加载（已验证可用）

```bash
source env/bin/activate
python test_image_loading.py
```

这会加载 10 张图片并显示它们的 captions。

### 2. 构建图片索引

```bash
source env/bin/activate
python -m retriever.build_image_index
```

**注意**: 如果遇到 CLIP 模型加载问题（exit code 139），可以尝试：

1. **减少图片数量**：修改 `NUM_IMAGES = 10` 先测试
2. **使用 CPU 模式**：确保没有 GPU 内存问题
3. **检查依赖**：确保 transformers 和 torch 版本兼容

### 3. 调整配置

如果需要修改设置，编辑 `retriever/build_image_index.py`：

```python
NUM_IMAGES = 100        # 要处理的图片数量
BATCH_SIZE = 8          # 编码时的批次大小（如果内存不足可以减小）
LOCAL_IMAGE_DIR = "data/flickr8k/Images"  # 图片目录
CAPTIONS_FILE = "data/flickr8k/captions.txt"  # 标注文件
```

## 数据格式

### Captions 文件格式

`captions.txt` 是 CSV 格式：
```csv
image,caption
1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs in an entry way .
1000268201_693b08cb0e.jpg,A girl going into a wooden building .
...
```

每张图片有多个 captions（通常 5 条）。

### 图片文件

- 格式：JPG
- 命名：`{image_id}.jpg`
- 位置：`data/flickr8k/Images/`

## 输出

构建成功后，会在 `retriever/faiss_index/images/` 目录下生成：

- `index.faiss`: FAISS 索引文件
- `images.jsonl`: 图片元数据（包含 captions）
- `embeddings.npy`: 图片向量
- `metadata.json`: 索引元信息

## 故障排除

### 问题 1: CLIP 模型加载失败（exit code 139）

**可能原因**：
- 内存不足
- 库版本冲突
- 模型文件损坏

**解决方案**：
1. 减少 `NUM_IMAGES` 和 `BATCH_SIZE`
2. 检查 transformers 版本：`pip show transformers`
3. 重新下载模型：删除 `~/.cache/huggingface/` 中的 CLIP 模型缓存

### 问题 2: 图片加载失败

确保：
- 图片目录路径正确
- 图片文件可读
- captions.txt 文件存在且格式正确

## 验证

运行测试脚本验证各个步骤：

```bash
# 测试图片加载
python test_image_loading.py

# 测试完整流程（如果 CLIP 加载正常）
python test_build_index_stepwise.py
```

## 下一步

构建索引成功后，可以使用多模态检索器：

```python
from retriever.multimodal_retriever import MultimodalRetriever, MultimodalRetrieverConfig

config = MultimodalRetrieverConfig(
    use_text_retrieval=True,
    use_image_retrieval=True,
)
retriever = MultimodalRetriever(config=config)
results = retriever.retrieve("a dog playing", top_k=5)
```

