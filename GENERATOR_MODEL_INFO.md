# 当前用于 Generation 的 LLM 模型

## 默认模型

**模型名称**: `Qwen/Qwen2.5-1.5B-Instruct`

**模型类型**: 
- 因果语言模型 (Causal Language Model)
- 指令微调模型 (Instruct-tuned)
- 纯文本模型（不支持图像输入）

**模型大小**: 1.5B 参数

**来源**: Hugging Face Model Hub

## 代码位置

### 默认配置 (`generator/rag_generator.py:12`)
```python
DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
```

### 模型加载 (`generator/rag_generator.py:31-43`)
```python
class RAGGenerator:
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()  # ← 使用默认配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载 tokenizer 和模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **model_kwargs).to(self.device)
        self.model.eval()
```

### 生成配置 (`generator/rag_generator.py:18-20`)
```python
@dataclass
class GeneratorConfig:
    model_name: str = DEFAULT_GENERATOR_MODEL  # ← 默认使用 Qwen2.5-1.5B-Instruct
    max_new_tokens: int = 256
    cache_dir: Optional[Path] = DEFAULT_CACHE_DIR
```

## 当前使用情况

在 `batch_test.py` 中：
```python
gen_config = GeneratorConfig()  # ← 使用默认配置，即 Qwen2.5-1.5B-Instruct
generator = RAGGenerator(gen_config)
```

## 模型特点

### ✅ 优点
- **轻量级**: 1.5B 参数，适合本地运行
- **指令遵循**: Instruct 版本，适合问答任务
- **快速推理**: 参数量小，生成速度快

### ⚠️ 限制
- **纯文本模型**: 不支持图像输入（只能处理文本 caption）
- **模型较小**: 1.5B 参数，可能不如更大的模型准确
- **中文支持**: Qwen 系列对中文支持较好，但主要针对英文任务

## 如何修改模型

### 方法 1: 修改默认配置
编辑 `generator/rag_generator.py`:
```python
DEFAULT_GENERATOR_MODEL = "your-model-name"
```

### 方法 2: 在代码中指定
```python
from generator.rag_generator import GeneratorConfig, RAGGenerator

# 使用不同的模型
gen_config = GeneratorConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # 更大的模型
    max_new_tokens=512,
)
generator = RAGGenerator(gen_config)
```

### 方法 3: 通过命令行参数（rag_pipeline.py）
```bash
python rag_pipeline.py --query "your query" --generator_model "Qwen/Qwen2.5-7B-Instruct"
```

## 其他可用的模型选项

### 更大的 Qwen 模型
- `Qwen/Qwen2.5-3B-Instruct` - 3B 参数
- `Qwen/Qwen2.5-7B-Instruct` - 7B 参数
- `Qwen/Qwen2.5-14B-Instruct` - 14B 参数

### 其他指令模型
- `meta-llama/Llama-2-7b-chat-hf` - Llama 2 (需要授权)
- `mistralai/Mistral-7B-Instruct-v0.2` - Mistral
- `microsoft/phi-2` - 2.7B 参数，轻量级

### 视觉-语言模型（如果要支持图像输入）
- `Qwen/Qwen-VL` - Qwen 的视觉版本
- `llava-hf/llava-1.5-7b-hf` - LLaVA
- `Salesforce/blip2-opt-2.7b` - BLIP-2

**注意**: 如果要使用视觉-语言模型，需要修改 `RAGGenerator` 类以支持图像输入。

## 当前模型在 RAG 中的使用

1. **输入**: 纯文本 prompt（包含查询和检索到的文本/caption）
2. **处理**: 文本 tokenization → 模型推理
3. **输出**: 生成的文本答案

**不支持**:
- ❌ 直接输入图像像素
- ❌ 图像 embedding
- ❌ 多模态输入

**支持**:
- ✅ 文本查询
- ✅ 检索到的文本块
- ✅ 图片的文本描述（caption）
