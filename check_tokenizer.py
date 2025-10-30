from transformers import PreTrainedTokenizerFast

# === 1. 加载你自己训练的 tokenizer ===
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer.json",  # ← 改成你的路径
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>",
)

# === 2. 打印基本信息 ===
print(f"Vocab size: {len(tokenizer)}")
print("=" * 20)

# === 3. 测试 tokenizer 行为 ===
# 1) 完整句子 encode->decode （判断整体行为）
text = "The quick brown fox jumps over the lazy dog."
ids = tokenizer.encode(text, add_special_tokens=False)
print("ids:", ids[:50])
print("decoded full:", repr(tokenizer.decode(ids)))

# 2) 单 token vs 多 token 的区别（示范）
single_id = ids[1]  # 取句子中第二个 token 的 id 做对比
print("single token id:", single_id)
print("convert token string:", tokenizer.convert_ids_to_tokens([single_id]))
print("decode([single_id]):", repr(tokenizer.decode([single_id])))