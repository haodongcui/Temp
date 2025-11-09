from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download
import torch
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

############################## 模型加载 ##############################

model_dir = snapshot_download('LLM-Research/Llama-3.2-1B-Instruct')
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    dtype=torch.float16 if device.type == "cuda" else torch.float32,
    # 不设置 device_map，让 Trainer 自动处理设备分配
)

# 设置 padding token（如果不存在）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# 确保模型处于训练模式
model.train()

############################## 封装微调配置和模型 ##############################

# 检查模型结构，找到正确的 target_modules
def find_target_modules(model):
    """自动查找可用的线性层模块名"""
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 查找注意力相关的线性层
            if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
                target_modules.append(name.split('.')[-1])  # 只取最后一层名称
    # 去重并返回
    return list(set(target_modules))

# 尝试自动查找或使用默认值
try:
    available_modules = find_target_modules(model)
    print(f"找到的可用模块: {available_modules}")
    # Llama 3.2 通常使用这些模块
    if available_modules:
        target_modules = [m for m in ['q_proj', 'k_proj', 'v_proj', 'o_proj'] if m in available_modules]
    else:
        target_modules = ['q_proj', 'v_proj']  # 默认值
except:
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']  # Llama 3.2 标准模块

print(f"使用 LoRA target_modules: {target_modules}")

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=target_modules,  # 指定注入LoRA的线性层  
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,  # 明确指定任务类型
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# 确保 PEFT 模型处于训练模式
peft_model.train()

# 验证可训练参数
trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
print(f"可训练参数数量: {trainable_params}")
if trainable_params == 0:
    raise ValueError("错误：没有可训练的参数！请检查 LoRA 配置。")

# 验证模型配置
print(f"模型训练模式: {peft_model.training}")
print(f"模型设备: {next(peft_model.parameters()).device}")

# 重要：确保模型配置支持梯度检查点
if hasattr(peft_model, 'config'):
    if hasattr(peft_model.config, 'use_cache'):
        peft_model.config.use_cache = False  # 梯度检查点需要关闭 use_cache

# 验证 LoRA 层是否正确配置
print("\n检查 LoRA 层配置:")
for name, param in peft_model.named_parameters():
    if param.requires_grad and 'lora' in name.lower():
        print(f"  {name}: requires_grad={param.requires_grad}, device={param.device}")
        break  # 只打印第一个作为示例

############################## 加载微调数据集 ##############################

dataset = load_dataset('llamafactory/alpaca_zh', 'default', split='train')
print(dataset)

############################## 数据预处理 ##############################

def format_prompt(example):
    """将 instruction/input/output 格式化为模型输入"""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    # 构建输入文本：instruction + input（如果有）
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    # 完整文本 = prompt + output
    full_text = prompt + output
    
    return {"text": full_text, "prompt": prompt, "output": output}

def preprocess_function(examples):
    """对数据进行 tokenize 处理"""
    # 格式化文本
    texts = []
    for i in range(len(examples['instruction'])):
        example = {
            'instruction': examples['instruction'][i],
            'input': examples['input'][i],
            'output': examples['output'][i]
        }
        formatted = format_prompt(example)
        texts.append(formatted['text'])
    
    # Tokenize
    model_inputs = tokenizer(
        texts,
        truncation=True,
        max_length=512,  # 根据显存调整，8GB 建议 512-1024
        padding=False,
    )
    
    # 创建 labels（只对输出部分计算 loss）
    labels = []
    for i, text in enumerate(texts):
        # 找到 prompt 的长度
        example = {
            'instruction': examples['instruction'][i],
            'input': examples['input'][i],
            'output': examples['output'][i]
        }
        formatted = format_prompt(example)
        prompt_len = len(tokenizer(formatted['prompt'], add_special_tokens=False)['input_ids'])
        
        # 创建 label：prompt 部分设为 -100（忽略），output 部分保留
        label = [-100] * prompt_len + model_inputs['input_ids'][i][prompt_len:]
        # 确保长度一致
        if len(label) < len(model_inputs['input_ids'][i]):
            label = label + [-100] * (len(model_inputs['input_ids'][i]) - len(label))
        elif len(label) > len(model_inputs['input_ids'][i]):
            label = label[:len(model_inputs['input_ids'][i])]
        labels.append(label)
    
    model_inputs['labels'] = labels
    return model_inputs

# 预处理数据集
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,  # 移除原始列
)
print(f"预处理后的数据集: {tokenized_dataset}")

############################## 微调训练 ##############################
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-4,                # 学习率
    per_device_train_batch_size=1,      # 8GB 显存建议 batch_size=1，配合 gradient_accumulation_steps
    per_device_eval_batch_size=1,       # 评估批量大小
    gradient_accumulation_steps=8,      # 梯度累积，等效 batch_size = 1 * 8 = 8
    num_train_epochs=3,                 # 训练轮数
    save_steps=100,                     # 每多少步保存一次模型
    save_total_limit=2,                # 最多保存多少个模型
    logging_steps=10,                   # 每多少步打印一次日志
    logging_dir="./logs",               # 日志保存目录
    logging_first_step=True,            # 是否打印第一步的日志
    remove_unused_columns=False,        # 保留所有列（已预处理，不会有问题）
    fp16=True,                          # 混合精度训练，节省显存
    gradient_checkpointing=False,       # 暂时禁用梯度检查点（PEFT 模型可能不兼容）
    dataloader_pin_memory=False,        # 8GB 显存建议关闭
    ddp_find_unused_parameters=False,   # 加速训练
)

# 确保模型在正确的设备上（Trainer 会自动处理，但这里显式检查）
if torch.cuda.is_available():
    print(f"CUDA 可用，GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    # Trainer 会自动将模型移到 GPU，这里不需要手动移动

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,  # 使用预处理后的数据集
    # eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,  # 添加 tokenizer 用于保存
)

trainer.train()