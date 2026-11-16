#!/usr/bin/env bash
# Compare plain decoding vs built-in MTP speculative decoding on Qwen3.8-27B IQ3_XXS.
set -u
B=~/llama.cpp/build/bin
M=~/models/qwen38/Qwen3.8-27B-UD-IQ3_XXS.gguf
R=~/hw1_llm/results
cd "$R"

run() {  # name, extra args...
  name=$1; shift
  $B/llama-server -m "$M" -ngl 99 -c 4096 -np 1 --port 18081 "$@" > F_server_$name.log 2>&1 &
  pid=$!
  for i in $(seq 1 120); do curl -s localhost:18081/health | grep -q ok && break; sleep 2; done
  for task in code prose; do
    if [ $task = code ]; then
      prompt="Write a Python function that parses a CSV file of transactions (date, amount, category) and returns total spending per category per month. Include type hints and docstrings."
    else
      prompt="Explain in detail, for a university student, why autoregressive decoding of large language models on a GPU is limited by memory bandwidth rather than by compute."
    fi
    for rep in 1 2; do
      curl -s localhost:18081/v1/chat/completions -H 'Content-Type: application/json' -d "{
        \"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],
        \"max_tokens\":384, \"temperature\":0, \"chat_template_kwargs\":{\"enable_thinking\":false}
      }" > F_${name}_${task}_${rep}.json
    done
  done
  kill $pid; wait $pid 2>/dev/null
}

run plain
run mtp --spec-type draft-mtp                              # default draft length: CUDA OOM on 12 GB
run mtp1 --spec-type draft-mtp --spec-draft-n-max 1
run mtp2 --spec-type draft-mtp --spec-draft-n-max 2
run mtp3 --spec-type draft-mtp --spec-draft-n-max 3        # CUDA OOM
run mtp4 --spec-type draft-mtp --spec-draft-n-max 4        # CUDA OOM
echo DONE
