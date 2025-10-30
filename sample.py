# Sampling helper (code fragment) — place this in your repo (e.g. sample.py) and run it where
# iterative_transformer_shakespeare.pt and the model/utility definitions are available.
# Assumes IterativeTransformer, and make_initial_logits (or similar) are defined/importable.

import torch
import torch.nn.functional as F
import random

def _make_init_logits_from_known(batch_tokens, known_mask, V, device, strong_val=50.0):
    """
    Minimal replacement for make_initial_logits but where known_mask & batch_tokens
    are provided (for generation). batch_tokens: [B,N] long, with arbitrary values
    in masked positions (we will ignore them). known_mask: [B,N] bool True=known.
    Returns L_init, known_mask, known_idx.
    """
    B, N = batch_tokens.shape
    L_init = torch.zeros(B, N, V, device=device)
    # set strong one-hot for known positions
    L_init.scatter_(-1, batch_tokens.unsqueeze(-1).to(device), strong_val)
    # masked positions remain zeros (uniform)
    L_init[~known_mask] = 0.0
    known_idx = batch_tokens.clone().to(device)
    known_idx[~known_mask] = 0
    return L_init, known_mask, known_idx

def top_k_filter(probs, top_k):
    if top_k is None or top_k <= 0:
        return probs
    v, _ = torch.topk(probs, top_k)
    min_topk = v[..., -1, None]
    filtered = torch.where(probs < min_topk, torch.zeros_like(probs), probs)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    return filtered

def sample_from_checkpoint(
    ckpt_path="iterative_transformer_shakespeare.pt",
    prompt="ROMEO: ",
    gen_len=200,
    seq_len=128,
    temperature=1.0,
    top_k=50,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = ckpt['vocab']                 # list of chars
    config = ckpt.get('config', {})
    # rebuild model using saved config (make sure IterativeTransformer signature accepts max_seq_len)
    model = IterativeTransformer(
        vocab_size=len(vocab),
        d_model=config.get('d_model', 128),
        nhead=config.get('nhead', 4),
        num_layers=config.get('num_layers', 3),
        alpha=config.get('alpha', 0.6),
        # if you added max_seq_len to ctor:
        max_seq_len=config.get('max_seq_len', max(seq_len, 1024))
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # build stoi
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    # tokenize prompt (map unknown chars to first vocab idx)
    prompt_ids = [stoi.get(ch, 0) for ch in prompt]

    generated_ids = list(prompt_ids)  # will grow
    total_to_gen = gen_len
    T_refine = config.get('T', 3)

    with torch.no_grad():
        while total_to_gen > 0:
            # window: take last seq_len tokens as context (prefix + masked tail)
            window_prefix = generated_ids[-seq_len:] if len(generated_ids) >= seq_len else generated_ids
            prefix_len = len(window_prefix)
            # we will generate up to min(remaining, seq_len - prefix_len) tokens in this window
            # but to keep logic simple, we generate one token at a time sequentially.
            # Build batch of size 1
            B = 1
            N = seq_len

            # prepare batch_tokens and known_mask: known for prefix positions; masked for others
            batch_tokens = torch.zeros((B, N), dtype=torch.long, device=device)
            known_mask = torch.zeros((B, N), dtype=torch.bool, device=device)

            # fill prefix into the rightmost positions of the window (left align works too)
            # here we left-align into positions [0:prefix_len)
            for i, tid in enumerate(window_prefix):
                batch_tokens[0, i] = tid
                known_mask[0, i] = True

            # set up initial logits where known positions are one-hot strong, others uniform (zeros)
            L_init, known_mask_batch, known_idx = _make_init_logits_from_known(batch_tokens, known_mask, len(vocab), device)

            # make model re-enforce buffer of known indices
            model.set_known_index_buffer(known_idx)

            # forward: run T_refine iterations to let model predict masked positions
            L_final, _ = model(L_init, known_mask=known_mask_batch, T=T_refine)

            # pick next position to sample: first masked index after prefix
            next_pos = prefix_len  # index in [0..N)
            if next_pos >= N:
                # if prefix already fills the window (rare if we generate one by one),
                # truncate window and continue (this case shouldn't happen given logic)
                continue

            logits = L_final[0, next_pos]  # [V]
            probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)  # temperature
            probs = top_k_filter(probs, top_k)
            probs = probs.cpu().numpy()

            # sample token id
            next_id = int(random.choices(range(len(probs)), weights=probs, k=1)[0])
            # append to generated
            generated_ids.append(next_id)
            total_to_gen -= 1

    # decode to string
    out = ''.join(itos[i] for i in generated_ids)
    return out

# Example usage:
# print(sample_from_checkpoint("iterative_transformer_shakespeare.pt", prompt="ROMEO: ", gen_len=300))
