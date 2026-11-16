"""Baseline latency/memory benchmark for a decoder-only LLM on a single GPU.

Measures, for several prompt lengths (batch = 1):
  * TTFT  - prefill time (one forward over the whole prompt)
  * decode tokens/s - step-by-step greedy decoding with KV cache
  * effective memory bandwidth during decode (weight bytes * tok/s)
  * effective prefill TFLOPS (~2 * params * prompt_tokens / TTFT)
  * peak GPU memory and KV-cache size
Then runs torch.profiler over one prefill and a few decode steps and
saves per-op tables plus Chrome traces.

Usage:
  python bench_llm.py --model Qwen/Qwen3-4B-Instruct-2507 --out results_bf16
"""
import argparse
import json
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer


def sync():
    torch.cuda.synchronize()


def make_prompt(vocab_size, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    # avoid special tokens at the top of the vocab
    return torch.randint(100, min(vocab_size, 150000), (1, n), generator=g)


@torch.inference_mode()
def prefill(model, ids):
    out = model(input_ids=ids, use_cache=True)
    return out.logits[:, -1:].argmax(-1), out.past_key_values


@torch.inference_mode()
def decode(model, tok, past, steps, per_step=None):
    for _ in range(steps):
        t0 = time.perf_counter() if per_step is not None else 0
        out = model(input_ids=tok, past_key_values=past, use_cache=True)
        tok = out.logits[:, -1:].argmax(-1)
        past = out.past_key_values
        if per_step is not None:
            sync()
            per_step.append(time.perf_counter() - t0)
    return tok, past


def kv_bytes(cfg, n_tokens, dtype_bytes):
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * n_tokens * dtype_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--prompt-lens", default="128,512,2048,8192")
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results_bf16")
    ap.add_argument("--peak-bw-gbs", type=float, default=360.0, help="RTX 3060 12GB")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    dev = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(dev).eval()
    sync()

    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # embeddings are gathered, not streamed, during decode; lm_head is streamed.
    emb = model.get_input_embeddings().weight
    tied = model.get_output_embeddings().weight.data_ptr() == emb.data_ptr()
    streamed_bytes = weight_bytes - (0 if tied else emb.numel() * emb.element_size())
    mem_after_load = torch.cuda.memory_allocated()

    info = {
        "model": args.model,
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "params_B": n_params / 1e9,
        "weight_GB": weight_bytes / 1e9,
        "streamed_per_token_GB": streamed_bytes / 1e9,
        "tied_embeddings": tied,
        "mem_after_load_GB": mem_after_load / 1e9,
        "layers": cfg.num_hidden_layers,
        "hidden": cfg.hidden_size,
        "intermediate": cfg.intermediate_size,
        "q_heads": cfg.num_attention_heads,
        "kv_heads": cfg.num_key_value_heads,
        "head_dim": getattr(cfg, "head_dim", None),
        "vocab": cfg.vocab_size,
        "kv_bytes_per_token_KB": kv_bytes(cfg, 1, emb.element_size()) / 1e3,
        "bw_ceiling_tok_s": args.peak_bw_gbs * 1e9 / streamed_bytes,
    }
    print(json.dumps(info, indent=2))

    # warm-up (cuBLAS heuristics, allocator)
    w = make_prompt(cfg.vocab_size, 64).to(dev)
    t, p = prefill(model, w)
    decode(model, t, p, 8)
    sync()

    rows = []
    for L in [int(x) for x in args.prompt_lens.split(",")]:
        ttfts, rates, steps_all = [], [], []
        for r in range(args.repeats):
            ids = make_prompt(cfg.vocab_size, L, seed=r).to(dev)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            sync()
            t0 = time.perf_counter()
            tok, past = prefill(model, ids)
            sync()
            ttft = time.perf_counter() - t0

            per_step = []
            sync()
            t1 = time.perf_counter()
            decode(model, tok, past, args.new_tokens, per_step)
            sync()
            dt = time.perf_counter() - t1
            ttfts.append(ttft)
            rates.append(args.new_tokens / dt)
            steps_all.append(per_step)
            peak = torch.cuda.max_memory_allocated()
            del past
        ttft = sorted(ttfts)[len(ttfts) // 2]
        rate = sorted(rates)[len(rates) // 2]
        step_ms = [1e3 * s for s in steps_all[0]]
        row = {
            "prompt_len": L,
            "ttft_ms": 1e3 * ttft,
            "prefill_tok_s": L / ttft,
            "prefill_TFLOPS": 2 * n_params * L / ttft / 1e12,
            "decode_tok_s": rate,
            "decode_ms_per_tok": 1e3 / rate,
            "first_step_ms": step_ms[0],
            "last_step_ms": step_ms[-1],
            "decode_eff_BW_GBs": streamed_bytes * rate / 1e9,
            "bw_utilization": streamed_bytes * rate / 1e9 / args.peak_bw_gbs,
            "kv_cache_MB_at_end": kv_bytes(cfg, L + args.new_tokens, emb.element_size()) / 1e6,
            "peak_mem_GB": peak / 1e9,
        }
        print(json.dumps(row))
        rows.append(row)

    with open(os.path.join(args.out, "bench.json"), "w") as f:
        json.dump({"info": info, "rows": rows}, f, indent=2)

    # ---------------- profiler ----------------
    L = 2048
    ids = make_prompt(cfg.vocab_size, L).to(dev)
    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=acts, record_shapes=True) as prof_pf:
        tok, past = prefill(model, ids)
        sync()
    with profile(activities=acts, record_shapes=True) as prof_dc:
        decode(model, tok, past, 16)
        sync()
    for name, prof in [("prefill", prof_pf), ("decode", prof_dc)]:
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
        with open(os.path.join(args.out, f"profile_{name}.txt"), "w") as f:
            f.write(table)
        prof.export_chrome_trace(os.path.join(args.out, f"trace_{name}.json"))
        print(f"\n===== {name} (prompt {L}) =====\n{table}")

    # CPU-side launch overhead vs GPU busy time for decode
    ev = prof_dc.key_averages()
    gpu_us = sum(e.self_device_time_total for e in ev)
    kernel_launches = sum(e.count for e in ev if e.key in ("cudaLaunchKernel", "cuLaunchKernel"))
    summary = {
        "decode_16_steps_gpu_busy_ms": gpu_us / 1e3,
        "kernel_launches_per_token": kernel_launches / 16,
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.out, "profile_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
