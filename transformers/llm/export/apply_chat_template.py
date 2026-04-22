from transformers import AutoTokenizer

# 加载你本地的 Qwen3 tokenizer（假设你已下载到 ./qwen3-local）
tokenizer = AutoTokenizer.from_pretrained("/home/chensm22/Models/Qwen3-4B/", trust_remote_code=True)
# 或者如果你有本地路径：
# tokenizer = AutoTokenizer.from_pretrained("./qwen3-local", trust_remote_code=True)

def apply_qwen3_chat_template(user_message: str) -> str:
    # 构造对话历史（单轮）
    messages = [
        {"role": "system", "content": "You are Qwen, a large language model developed by Alibaba Cloud. You are helpful, harmless, and honest."},
        {"role": "user", "content": user_message}
    ]
    # 应用内置 chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # 返回字符串而非 token ID
        add_generation_prompt=True  # 添加生成提示（如 <|im_start|>assistant）
    )
    return input_text

# 示例使用
prompt = "你好，你是谁？"
formatted_input = apply_qwen3_chat_template(prompt)
print(formatted_input)