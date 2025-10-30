# iterative_transformer_shakespeare.py
# Minimal iterative / looped transformer prototype for char-level Shakespeare-like data.
# Simplified for clarity and ease of running on a laptop/GPU.
# Python 3.8+, PyTorch

import math
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Data utilities (char-level)
# ---------------------------
class CharDataset(Dataset):
    def __init__(self, text, seq_len=128):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.data = [self.stoi[ch] for ch in text]
        self.seq_len = seq_len
        # make non-overlapping chunks for simplicity
        self.seqs = []
        for i in range(0, len(self.data) - seq_len, seq_len):
            self.seqs.append(self.data[i:i+seq_len])
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.long)

def load_shakespeare(path="shakespeare.txt", seq_len=128):
    text = Path(path).read_text(encoding="utf-8")
    ds = CharDataset(text, seq_len=seq_len)
    return ds

# ---------------------------
# Model: Iterative Transformer
# ---------------------------
class IterativeTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3, alpha=0.6):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.alpha = alpha

        # embedding for tokens (used when mapping distribution -> expected embedding)
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # a small Transformer encoder (we reuse same module across iterations)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # project transformer outputs back to token-logit space (delta logits)
        self.out_proj = nn.Linear(d_model, vocab_size)

        # small layernorm on delta for stability
        self.delta_norm = nn.LayerNorm(vocab_size)

        self.max_seq_len = max_seq_len = 1024  # 可在 __init__ 参数里加入默认值 max_seq_len=1024
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer('pos_enc', pe)  # 不参与训练，但随模型移动到 device

    def forward(self, L_init, known_mask=None, T=3, detach_every=0):
        """
        L_init: [B, N, V] initial logits (float)
        known_mask: [B, N] bool tensor, True where token is given (not masked)
        T: number of iterative refinement steps
        detach_every: if >0, detach L every `detach_every` steps (for truncated BPTT-like stability)
        returns final_logits [B, N, V], and optionally intermediate logits list
        """
        B, N, V = L_init.shape
        L = L_init  # working logits

        intermediates = []
        for t in range(T):
            # optionally detach for stability
            if detach_every and (t % detach_every == 0) and (t > 0):
                L = L.detach()

            # probs over vocab
            p = torch.softmax(L, dim=-1)  # [B, N, V]

            # expected embedding per position: sum_v p_{v} * emb_v
            # Efficient matmul: [B, N, V] @ [V, d] -> [B, N, d]
            expected_e = torch.matmul(p, self.token_emb.weight)  # [B, N, d]
            
            # 在将 expected_e 传入 transformer 之前加入：
            pos = self.pos_enc[:, :N, :].to(expected_e.dtype)     # [1, N, d]
            expected_e = expected_e + pos                         # 广播到 [B, N, d]

            # Transformer expects [seq_len, batch, d]
            x_in = expected_e.permute(1, 0, 2)  # [N, B, d]
            h = self.transformer(x_in)  # [N, B, d]
            h = h.permute(1, 0, 2)  # [B, N, d]

            # project to delta logits
            delta = self.out_proj(h)  # [B, N, V]
            # normalize delta for stability
            delta = self.delta_norm(delta)

            # residual update
            L = L + self.alpha * delta

            # enforce known tokens as (very confident) one-hot logits
            if known_mask is not None:
                # known_mask: bool, True where token given. We leave those logits as strong
                # rather than zeroing them, we set them to a large one-hot to ensure they remain fixed.
                # For efficiency, we assume a one-hot index tensor is available via attribute `known_idx`.
                # But here we expect L_init was already set accordingly; below we re-enforce:
                if hasattr(self, "_known_idx"):
                    idx = self._known_idx  # [B, N] long
                    # set to VERY_NEG for all
                    # for known positions, set huge positive on their index
                    # build one-hot large logits:
                    one_hot = torch.zeros_like(L)
                    one_hot.scatter_(-1, idx.unsqueeze(-1), 50.0)  # large positive
                    L = torch.where(known_mask.unsqueeze(-1), one_hot, L)

            intermediates.append(L)

        return L, intermediates

    def set_known_index_buffer(self, known_idx):
        # convenience: remember known indices to re-enforce during forward
        # known_idx: [B, N] long
        self._known_idx = known_idx

# ---------------------------
# Training utilities
# ---------------------------
def make_initial_logits(batch_tokens, mask_prob=0.15, device='cpu'):
    """
    batch_tokens: [B, N] long tensor of token indices (ground truth)
    Returns:
      L_init: [B, N, V] logits with known positions as strong one-hot and masked positions as zeros
      known_mask: [B, N] bool (True=known)
      known_idx: [B, N] long (index where known, filled with 0 for masked)
    """
    B, N = batch_tokens.shape
    V = vocab_size_global  # using global for simplicity
    L_init = torch.zeros(B, N, V, device=device)
    known_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    known_idx = batch_tokens.clone().to(device)

    # randomly mask some positions for denoising (do not mask special treatment)
    mask = torch.rand(B, N, device=device) < mask_prob
    # ensure at least one masked per sequence (optional)
    # apply mask: where mask == True we treat as unknown
    known_mask = ~mask
    # set known positions to strong one-hot logits
    L_init.scatter_(-1, batch_tokens.unsqueeze(-1).to(device), 50.0)  # large positive
    # masked positions: set uniform (zeros logits correspond to uniform after softmax)
    L_init[~known_mask] = 0.0
    # for masked positions replace known_idx with 0 to avoid invalid indices (we won't use them)
    known_idx = batch_tokens.clone()
    known_idx[~known_mask] = 0
    return L_init, known_mask, known_idx

# ---------------------------
# Simple training loop
# ---------------------------
def train_loop(model, dataloader, optim, device, epochs=3, T=3, mask_prob=0.15, detach_every=0):
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        total_tokens = 0
        for i, batch in enumerate(dataloader,1):
            batch = batch.to(device)  # [B, N]
            L_init, known_mask, known_idx = make_initial_logits(batch, mask_prob=mask_prob, device=device)
            model.set_known_index_buffer(known_idx)  # to let model re-enforce known positions each step

            # forward
            L_final, _ = model(L_init, known_mask=known_mask, T=T, detach_every=detach_every)
            # compute loss only on masked positions (we want model to recover them)
            p_final = torch.log_softmax(L_final, dim=-1)  # log-probs
            # gather target
            targets = batch  # [B,N]
            # compute negative log likelihood on masked positions:
            nll = -p_final.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B,N]
            loss_mask = (~known_mask).float()  # 1 where masked (we want to predict)
            # avoid all-zero masks
            if loss_mask.sum() == 0:
                # skip if nothing masked (rare)
                continue
            loss = (nll * loss_mask).sum() / loss_mask.sum()

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 20 == 0:  # 每20个batch打印一次
                print(f"Epoch {ep+1}  batch {i}/{len(dataloader)}  loss = {loss.item():.4f}", flush=True)

            total_loss += loss.item() * loss_mask.sum().item()
            total_tokens += loss_mask.sum().item()

        avg_loss = total_loss / (total_tokens + 1e-12)
        print(f"Epoch {ep+1}/{epochs}  avg_masked_nll = {avg_loss:.4f}")

# ---------------------------
# Quick run setup
# ---------------------------
def run_toy(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_shakespeare(args.data, seq_len=args.seq_len)
    global vocab_size_global
    vocab_size_global = ds.vocab_size
    print("vocab_size:", vocab_size_global)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = IterativeTransformer(vocab_size_global, d_model=args.d_model,
                                 nhead=args.nhead, num_layers=args.num_layers,
                                 alpha=args.alpha).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    train_loop(model, dl, optim, device, epochs=args.epochs, T=args.T, mask_prob=args.mask_prob, detach_every=args.detach_every)

    # Save model
    torch.save({
        'model_state': model.state_dict(),
        'vocab': ds.chars,
        'config': vars(args)
    }, "iterative_transformer_shakespeare.pt")
    print("Saved model to iterative_transformer_shakespeare.pt")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="shakespeare.txt")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--T", type=int, default=3, help="refinement steps per sample")
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--detach_every", type=int, default=0, help="if >0, detach L every this many steps")
    args = parser.parse_args()
    run_toy(args)
