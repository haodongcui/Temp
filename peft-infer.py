from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from modelscope import snapshot_download
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# 模型加载
model_dir = snapshot_download('LLM-Research/Llama-3.2-1B-Instruct')
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# 模型推理
prompt = "解释一下大模型的推理过程"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# dataset = load_dataset('llamafactory/alpaca_zh', 'default', split='train')