# train_tokenizer.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, normalizers, decoders

# 1) 加载文本语料（wikitext-2 train）
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = (t for t in ds["text"] if t and t.strip())  # generator: 跳过空行

# 2) 构建 tokenizer (BPE + ByteLevel pre-tokenizer)
vocab_size = 4000
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
special_tokens = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, min_frequency=2)

# 3) 训练（从 iterator 训练）
tokenizer.train_from_iterator(texts, trainer=trainer)

# 4) 用 ByteLevel 的 post-processor 设定 BOS/EOS 显示（可选）
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

tokenizer.decoder = decoders.ByteLevel()

# 5) 保存为 tokenizer.json（兼容 transformers.PreTrainedTokenizerFast）
tokenizer.save("tokenizer.json")
print("Saved tokenizer to tokenizer.json")
