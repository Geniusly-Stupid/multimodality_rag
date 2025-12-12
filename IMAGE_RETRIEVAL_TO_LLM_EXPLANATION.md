# Image-Only 模式下图片如何输入 LLM 的详细解释

## 重要发现 ⚠️

**当前实现中，图片的像素数据并没有直接输入到 LLM，而是使用了图片的文本描述（caption）作为上下文。**

## 完整流程

### 1. 图片检索阶段 (`MultimodalRetriever._retrieve_images`)

```python
# retriever/multimodal_retriever.py:140-166
def _retrieve_images(self, query: str, top_k: int = 5) -> List[Dict]:
    # 使用 CLIP 文本编码器将查询编码为向量
    query_vec = self.image_encoder.encode([query])
    
    # 在 FAISS 索引中搜索最相似的图片
    distances, indices = self.image_index.search(query_vec, top_k)
    
    # 返回图片的元数据（包括 caption）
    results.append({
        "image_id": image_meta.get("image_id"),
        "caption": image_meta.get("caption", ""),      # ← 关键：文本描述
        "captions": image_meta.get("captions", []),
        "score": float(distances[0][rank]),
        "modality": "image",
    })
```

**返回的数据结构：**
- `image_id`: 图片标识符
- `caption`: 图片的文本描述（例如："A dog running in a park"）
- `captions`: 多个描述（如果有）
- `score`: 相似度分数
- `modality`: "image"

**注意：** 这里返回的是图片的**元数据**，而不是图片的像素数据。

### 2. Prompt 构建阶段 (`RAGGenerator.build_prompt`)

```python
# generator/rag_generator.py:45-79
def build_prompt(self, query: str, retrieved_chunks: List[Dict]) -> str:
    evidence_lines = []
    image_info = []
    
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        modality = chunk.get("modality", "text")
        
        if modality == "text":
            # 文本块：直接使用文本内容
            text = chunk.get("text", "").strip()
            if text:
                evidence_lines.append(f"[{idx}] {text}")
                
        elif modality == "image":
            # 图片块：使用 caption（文本描述）
            caption = chunk.get("caption", "").strip()
            image_id = chunk.get("image_id", "")
            if caption:
                image_info.append(f"[Image {idx}] ID: {image_id}, Caption: {caption}")
    
    # 构建 prompt
    prompt_parts = [
        "You are a helpful assistant that answers questions using the supplied evidence.\n",
        f"User question:\n{query}\n\n",
    ]
    
    if image_block:
        prompt_parts.append(f"Relevant images found:\n{image_block}\n\n")
    
    prompt_parts.append("Answer:\n")
    
    return "".join(prompt_parts)
```

**生成的 Prompt 示例：**

```
You are a helpful assistant that answers questions using the supplied evidence.
User question:
What is shown in the diagram?

Relevant images found:
[Image 1] ID: abc123, Caption: A flowchart showing the data processing pipeline
[Image 2] ID: def456, Caption: A bar chart comparing different algorithms

Answer:
```

### 3. LLM 生成阶段 (`RAGGenerator.generate`)

```python
# generator/rag_generator.py:81-94
@torch.inference_mode()
def generate(self, query: str, retrieved_chunks: List[Dict]) -> str:
    # 构建包含图片 caption 的文本 prompt
    prompt = self.build_prompt(query, retrieved_chunks)
    
    # 将 prompt 编码为 token
    encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    
    # 使用纯文本 LLM 生成答案（当前使用的是 Qwen2.5-1.5B-Instruct）
    output_ids = self.model.generate(**encoded, generation_config=gen_config)
    
    # 解码生成的答案
    generated = output_ids[0, encoded["input_ids"].shape[1] :]
    return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
```

## 关键点总结

### ✅ 当前实现做了什么：

1. **图片检索**：使用 CLIP 模型找到与查询最相关的图片
2. **提取元数据**：从图片索引中获取图片的文本描述（caption）
3. **文本化**：将图片的 caption 转换为文本格式放入 prompt
4. **LLM 推理**：使用纯文本 LLM（Qwen2.5）基于文本描述生成答案

### ❌ 当前实现没有做什么：

1. **没有加载图片像素数据**：图片文件本身没有被读取
2. **没有视觉编码**：图片的视觉特征没有被输入到 LLM
3. **不是视觉-语言模型**：当前使用的 LLM（Qwen2.5-1.5B-Instruct）不支持图像输入

## 实际效果

在 `image_only` 模式下：

- ✅ **优点**：
  - 可以找到与查询相关的图片
  - 图片的文本描述提供了上下文信息
  - 实现简单，不需要视觉-语言模型

- ⚠️ **限制**：
  - LLM 只能看到图片的文本描述，看不到图片本身
  - 如果 caption 不准确或不完整，LLM 可能无法正确理解图片内容
  - 对于需要视觉理解的问题（如"图片中有什么颜色"），效果可能不佳

## 示例对比

### 场景 1：查询 "What does the diagram show?"

**检索到的图片：**
- Image ID: `abc123`
- Caption: "A flowchart showing the data processing pipeline"

**输入到 LLM 的 Prompt：**
```
User question:
What does the diagram show?

Relevant images found:
[Image 1] ID: abc123, Caption: A flowchart showing the data processing pipeline

Answer:
```

**LLM 生成的答案：**
基于文本描述 "A flowchart showing the data processing pipeline" 生成答案，而不是基于实际的图片内容。

### 场景 2：查询 "What color is the car in the image?"

**检索到的图片：**
- Image ID: `xyz789`
- Caption: "A car on the road"  ← caption 中没有颜色信息

**问题：**
LLM 无法从 caption 中获取颜色信息，可能无法准确回答。

## 如何改进（未来方向）

如果要真正支持视觉输入，需要：

1. **使用视觉-语言模型**：
   - GPT-4V
   - LLaVA
   - Qwen-VL
   - BLIP-2

2. **修改生成器**：
   ```python
   # 伪代码示例
   def generate(self, query: str, retrieved_chunks: List[Dict]) -> str:
       # 加载图片文件
       images = [load_image(chunk['img_path']) for chunk in retrieved_chunks if chunk['modality'] == 'image']
       
       # 使用视觉-语言模型
       prompt = self.build_prompt(query, retrieved_chunks)
       answer = self.vision_model.generate(prompt, images=images)
       return answer
   ```

3. **在 prompt 中包含图片**：
   - 将图片编码为 base64 或直接传递图片 tensor
   - 使用支持多模态的模型架构

## 当前代码位置

- **图片检索**：`retriever/multimodal_retriever.py:140-166`
- **Prompt 构建**：`generator/rag_generator.py:45-79`
- **LLM 生成**：`generator/rag_generator.py:81-94`
- **Pipeline 调用**：`rag_pipeline.py:33-36`
