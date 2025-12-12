"""验证最终给 LLM 的是 caption 还是 embedding"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import gc
from retriever.multimodal_retriever import MultimodalRetriever, MultimodalRetrieverConfig
from generator.rag_generator import RAGGenerator, GeneratorConfig
from rag_pipeline import RAGPipeline

def verify_caption_vs_embedding():
    print("=" * 80)
    print("验证：最终给 LLM 的是 Caption 还是 Embedding？")
    print("=" * 80)
    
    gc.collect()
    
    # 初始化 retriever
    config = MultimodalRetrieverConfig(
        use_text_retrieval=False,
        use_image_retrieval=True,
    )
    retriever = MultimodalRetriever(config=config)
    
    gc.collect()
    
    # 测试查询
    query = "diagram chart"
    print(f"\n查询: {query}")
    print("-" * 80)
    
    # 步骤 1: 检索
    print("\n[步骤 1] 检索阶段")
    retrieved = retriever.retrieve(query, top_k=2)
    print(f"检索到 {len(retrieved)} 个图片结果")
    
    for i, chunk in enumerate(retrieved, 1):
        print(f"\n  结果 {i}:")
        print(f"    - modality: {chunk.get('modality')}")
        print(f"    - image_id: {chunk.get('image_id')}")
        print(f"    - caption: {chunk.get('caption', '')[:60]}...")
        print(f"    - img_path: {chunk.get('img_path', '')[:60]}...")
        print(f"    - score: {chunk.get('score', 0):.4f}")
        
        # 检查是否有 embedding
        has_embedding = 'embedding' in chunk or 'embedding_vector' in chunk or 'vec' in chunk
        print(f"    - 包含 embedding: {has_embedding}")
        if has_embedding:
            print(f"      ⚠️  发现 embedding 字段！")
        else:
            print(f"      ✓ 没有 embedding 字段（只有 caption）")
    
    # 步骤 2: 构建 prompt
    print("\n" + "=" * 80)
    print("[步骤 2] Prompt 构建阶段")
    print("=" * 80)
    
    gen_config = GeneratorConfig()
    generator = RAGGenerator(gen_config)
    
    prompt = generator.build_prompt(query, retrieved)
    print("\n生成的 Prompt:")
    print("-" * 80)
    print(prompt)
    print("-" * 80)
    
    # 检查 prompt 中是否包含 embedding
    has_embedding_in_prompt = 'embedding' in prompt.lower() or '[tensor' in prompt.lower()
    has_caption_in_prompt = 'caption' in prompt.lower() or 'Caption:' in prompt
    
    print(f"\nPrompt 分析:")
    print(f"  - 包含 'caption': {has_caption_in_prompt}")
    print(f"  - 包含 'embedding': {has_embedding_in_prompt}")
    
    # 步骤 3: Tokenization
    print("\n" + "=" * 80)
    print("[步骤 3] Tokenization 阶段")
    print("=" * 80)
    
    encoded = generator.tokenizer(prompt, return_tensors="pt")
    print(f"Input IDs shape: {encoded['input_ids'].shape}")
    print(f"Input IDs (前 20 个 tokens): {encoded['input_ids'][0][:20].tolist()}")
    
    # 解码前几个 tokens 看看内容
    decoded_preview = generator.tokenizer.decode(encoded['input_ids'][0][:30], skip_special_tokens=False)
    print(f"\n前 30 个 tokens 解码结果:")
    print(f"  {decoded_preview[:200]}...")
    
    # 最终结论
    print("\n" + "=" * 80)
    print("最终结论")
    print("=" * 80)
    
    if has_caption_in_prompt and not has_embedding_in_prompt:
        print("✅ 确认：LLM 接收的是 CAPTION（文本描述）")
        print("   - 检索结果中包含 caption 文本")
        print("   - Prompt 中包含 caption 文本")
        print("   - 没有 embedding 被传递给 LLM")
    elif has_embedding_in_prompt:
        print("⚠️  警告：发现 embedding 在 prompt 中！")
    else:
        print("❓ 无法确定")
    
    print("\n" + "=" * 80)
    print("详细分析")
    print("=" * 80)
    print("""
1. 检索阶段：
   - 使用 CLIP embedding 进行搜索（在 FAISS 索引中）
   - 返回的是元数据字典，包含 caption，但不包含 embedding

2. Prompt 构建：
   - 从检索结果中提取 caption 文本
   - 将 caption 转换为文本格式放入 prompt
   - 没有使用 embedding

3. LLM 生成：
   - 使用文本 tokenizer 编码 prompt（纯文本）
   - LLM 接收的是 token IDs，不是 embedding
   - 生成的是文本答案
    """)

if __name__ == "__main__":
    try:
        verify_caption_vs_embedding()
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
