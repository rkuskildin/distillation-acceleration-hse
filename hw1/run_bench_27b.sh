#!/usr/bin/env bash
# Qwen3.8-27B on RTX 3060 12GB: llama.cpp measurements for HW1.
set -u
B=~/llama.cpp/build/bin
M=~/models/qwen38
R=~/hw1_llm/results
mkdir -p "$R"
cd "$R"

IQ3=$M/Qwen3.8-27B-UD-IQ3_XXS.gguf
Q2=$M/Qwen3.8-27B-UD-Q2_K_XL.gguf
Q4=$M/Qwen3.8-27B-UD-Q4_K_M.gguf

nvidia-smi > env_nvidia.txt
free -g > env_mem.txt
lscpu > env_cpu.txt

echo "== A: quants fully on GPU, prefill vs prompt length + decode"
for m in "$IQ3" "$Q2"; do
  $B/llama-bench -m "$m" -ngl 99 -p 512,2048,8192 -n 128 -r 3 -o jsonl 2>>bench_err.log
done > A_quants_gpu.jsonl

echo "== B: Q4_K_M partial offload sweep"
$B/llama-bench -m "$Q4" -ngl 0,16,24,32,36,40,44,48 -p 512 -n 64 -r 2 -o jsonl 2>>bench_err.log > B_q4_offload.jsonl
# same for IQ3 to show the cliff when a model that fits is pushed off the GPU
$B/llama-bench -m "$IQ3" -ngl 32,48,56 -p 512 -n 64 -r 2 -o jsonl 2>>bench_err.log > B_iq3_offload.jsonl

echo "== C: decode/prefill vs context depth (KV cache growth), IQ3 on GPU"
$B/llama-bench -m "$IQ3" -ngl 99 -p 512 -n 64 -d 0,2048,8192,16384 -r 2 -o jsonl 2>>bench_err.log > C_depth_f16kv.jsonl
$B/llama-bench -m "$IQ3" -ngl 99 -p 512 -n 64 -d 0,8192,16384 -ctk q8_0 -ctv q8_0 -fa on -r 2 -o jsonl 2>>bench_err.log > C_depth_q8kv.jsonl
# flash attention on/off at depth
$B/llama-bench -m "$IQ3" -ngl 99 -p 2048 -n 64 -d 8192 -fa on,off -r 2 -o jsonl 2>>bench_err.log > C_fa.jsonl

echo "== D: VRAM at different context sizes (server load log)"
for c in 4096 16384 32768 65536; do
  $B/llama-server -m "$IQ3" -ngl 99 -c $c -np 1 --port 18080 > D_load_ctx$c.log 2>&1 &
  pid=$!
  for i in $(seq 1 90); do grep -q "listening\|error\|failed" D_load_ctx$c.log && break; sleep 2; done
  nvidia-smi --query-gpu=memory.used --format=csv,noheader > D_vram_ctx$c.txt
  kill $pid; wait $pid 2>/dev/null
done

echo "== E: perplexity on wikitext-2 (first 40 chunks x 512)"
if [ ! -f wiki.test.raw ]; then
  wget -q https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip -O wt2.zip && \
  python3 -c "import zipfile;z=zipfile.ZipFile('wt2.zip');open('wiki.test.raw','wb').write(z.read('wikitext-2-raw/wiki.test.raw'))"
fi
for q in IQ3 Q2 Q4; do
  eval m=\$$q
  ngl=99; [ $q = Q4 ] && ngl=44
  $B/llama-perplexity -m "$m" -ngl $ngl -f wiki.test.raw -c 512 --chunks 40 > E_ppl_$q.log 2>&1
done

echo DONE
