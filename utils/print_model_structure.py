############################## 查看模型结构 ##############################
def print_model_structure(model, max_depth=3):
    """打印模型结构"""
    print("\n" + "="*60)
    print("模型基本信息")
    print("="*60)
    print(f"模型类型: {type(model).__name__}")
    print(f"模型配置: {model.config.model_type}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("\n" + "="*60)
    print("模型层级结构（前几层）")
    print("="*60)
    for i, (name, module) in enumerate(model.named_children()):
        if i < 10:  # 只显示前10层
            print(f"{name}: {type(module).__name__}")
        elif i == 10:
            print("... (更多层)")
            break
    
    print("\n" + "="*60)
    print("所有模块名称（Linear层）")
    print("="*60)
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append(name)
            if len(linear_layers) <= 20:  # 只显示前20个
                print(f"  {name}: {module}")
    if len(linear_layers) > 20:
        print(f"  ... 还有 {len(linear_layers) - 20} 个 Linear 层")
    
    print("\n" + "="*60)
    print("模型配置信息")
    print("="*60)
    print(f"vocab_size: {model.config.vocab_size}")
    print(f"hidden_size: {model.config.hidden_size}")
    print(f"num_hidden_layers: {model.config.num_hidden_layers}")
    print(f"num_attention_heads: {model.config.num_attention_heads}")
    print(f"intermediate_size: {model.config.intermediate_size}")
    print("="*60 + "\n")

# 打印模型结构
print_model_structure(model)

# ========== 其他查看模型结构的方法 ==========
# 
# 方法1: 直接打印模型（会显示完整结构，但可能很长）
# print(model)
# 
# 方法2: 查看模型配置
# print(model.config)
# 
# 方法3: 遍历所有模块
# for name, module in model.named_modules():
#     print(f"{name}: {type(module).__name__}")
# 
# 方法4: 查看所有参数
# for name, param in model.named_parameters():
#     print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")
# 
# 方法5: 查看特定层的结构
# print(model.layers[0])  # 查看第一层
# 
# 方法6: 使用 torchsummary（需要安装: pip install torchsummary）
# from torchsummary import summary
# summary(model, input_size=(1, 512))  # 需要根据模型调整
# 
# 方法7: 查看模型的前向传播路径
# print(model.forward.__code__.co_names)  # 查看前向传播使用的函数