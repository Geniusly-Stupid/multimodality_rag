# 确认：最终给 LLM 的是 Embedding 还是 Caption？

## 完整流程追踪

### 1. 检索阶段 (`multimodal_retriever.py`)

```python
# 第 145-147 行：使用 CLIP 将查询编码为 embedding
query_vec = self.image_encoder.encode([query])  # ← 生成 embedding
distances, indices = self.image_index.search(query_vec, top_k)  # ← 用 embedding 搜索

# 第 155-182 行：返回的是元数据（包括 caption），不是 embedding
image_meta = self.image_metadata[idx]
results.append({
    "image_id": image_meta.get("image_id"),
    "img_path": img_path,
    "caption": caption,  # ← 返回的是 caption 文本
    "captions": captions,
    "score": float(distances[0][rank]),
    "modality": "image",
})
# 注意：这里没有返回 embedding！
```

**结论**：检索阶段使用 embedding 进行搜索，但返回给 pipeline 的是 **caption 文本**，embedding 被丢弃。

### 2. Pipeline 阶段 (`rag_pipeline.py`)

```python
# 第 33-36 行
def answer(self, query: str, top_k: int = 5) -> RAGOutput:
    retrieved = self.retriever.retrieve(query, top_k=top_k)  # ← 得到的是包含 caption 的字典列表
    generated_answer = self.generator.generate(query, retrieved)  # ← 传递给 generator
    return RAGOutput(query=query, retrieved_chunks=retrieved, generated_answer=generated_answer)
```

**结论**：Pipeline 传递的是检索结果（包含 caption），没有 embedding。

### 3. Prompt 构建阶段 (`generator/rag_generator.py`)

```python
# 第 45-79 行：build_prompt 方法
def build_prompt(self, query: str, retrieved_chunks: List[Dict]) -> str:
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        modality = chunk.get("modality", "text")
        
        if modality == "image":
            caption = chunk.get("caption", "").strip()  # ← 只使用 caption 文本
            image_id = chunk.get("image_id", "")
            if caption:
                image_info.append(f"[Image {idx}] ID: {image_id}, Caption: {caption}")  # ← caption 转为文本
    
    # 构建纯文本 prompt
    prompt_parts = [
        "You are a helpful assistant...",
        f"User question:\n{query}\n\n",
        f"Relevant images found:\n{image_block}\n\n",  # ← 只有 caption 文本
        "Answer:\n",
    ]
    return "".join(prompt_parts)
```

**结论**：Prompt 构建时只使用 **caption 文本**，完全没有使用 embedding。

### 4. LLM 生成阶段 (`generator/rag_generator.py`)

```python
# 第 81-94 行：generate 方法
@torch.inference_mode()
def generate(self, query: str, retrieved_chunks: List[Dict]) -> str:
    prompt = self.build_prompt(query, retrieved_chunks)  # ← 纯文本 prompt
    encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)  # ← 文本 tokenizer
    output_ids = self.model.generate(**encoded, generation_config=gen_config)  # ← 纯文本 LLM
    generated = output_ids[0, encoded["input_ids"].shape[1] :]
    return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
```

**结论**：LLM 接收的是 **纯文本 prompt**（包含 caption），完全没有 embedding。

## 最终答案

✅ **最终给 LLM 的是 CAPTION（文本描述），不是 EMBEDDING**

### Embedding 的作用：
- **仅用于检索阶段**：CLIP 将查询编码为 embedding，在 FAISS 索引中搜索相似图片
- **检索完成后被丢弃**：embedding 不会传递给 LLM

### Caption 的作用：
- **检索阶段**：从元数据中提取 caption
- **生成阶段**：caption 文本被放入 prompt，传递给 LLM

## 代码证据

### 检索返回的数据结构（`multimodal_retriever.py:170-182`）：
```python
results.append({
    "image_id": ...,
    "img_path": ...,
    "caption": caption,  # ← 只有 caption，没有 embedding
    "captions": captions,
    "score": ...,
    "modality": "image",
})
# 注意：没有 "embedding" 字段！
```

### Prompt 构建（`rag_generator.py:57-61`）：
```python
elif modality == "image":
    caption = chunk.get("caption", "").strip()  # ← 只提取 caption
    image_id = chunk.get("image_id", "")
    if caption:
        image_info.append(f"[Image {idx}] ID: {image_id}, Caption: {caption}")  # ← caption 转为文本
```

### LLM 输入（`rag_generator.py:84-85`）：
```python
prompt = self.build_prompt(query, retrieved_chunks)  # ← 纯文本 prompt
encoded = self.tokenizer(prompt, return_tensors="pt")  # ← 文本 tokenizer，不是 embedding
```

## 总结

| 阶段 | 使用 Embedding | 使用 Caption |
|------|---------------|--------------|
| 检索 | ✅ 是（用于搜索） | ❌ 否 |
| 返回结果 | ❌ 否 | ✅ 是（包含在元数据中） |
| Prompt 构建 | ❌ 否 | ✅ 是（提取 caption 文本） |
| LLM 生成 | ❌ 否 | ✅ 是（caption 在 prompt 中） |

**最终结论**：LLM 接收的是 **caption 文本**，embedding 只在检索阶段使用，不会传递给 LLM。
