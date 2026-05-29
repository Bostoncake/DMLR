# DMLR Execution Walkthrough

This document traces how DMLR runs end-to-end, from the shell script invocation through every major code path.

---

## 1. Entry Point: `script/run.sh`

```bash
export HF_HOME=~/.huggingface_cache
export CUDA_VISIBLE_DEVICES=0,1

python main.py \
    --dataset data/mmvp.json \
    --model_name_or_path /WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct \
    --output_dir ./output_vis/0325_reproduce/mmvp/Qwen2.5-VL-7B-Instruct \
    --device cuda \
    --seed 42 \
    --max_new_tokens 2048 \
    --max_num_steps 15 \
    --num_thought_tokens 2 \
    --sigma 25.0 \
    --sigma_decay 0.95 \
    --lr 0.01 \
    --min_pixels 128 \
    --max_pixels 256 \
    --num_workers 2 \
    --worker_device_round_robin \
    --num_selected_patches 16 \
    --initial_patch_count 1 \
    --patch_increment 1 \
    --visual_insert_stride 1 \
    --visual_injection_start_step 0 \
    --visual_injection_interval 1 \
    --use_llm_verify
```

**What this configures:**
- **2 GPUs** (`CUDA_VISIBLE_DEVICES=0,1`) with **2 workers** (`--num_workers 2`), each worker assigned to one GPU via `--worker_device_round_robin`.
- **Dataset**: `data/mmvp.json` — a VQA-style visual benchmark.
- **Model**: `Qwen2.5-VL-7B-Instruct`, a vision-language model.
- **Optimization**: 15 RL steps (`--max_num_steps 15`) of latent thought token search.
- **Latent tokens**: 2 thought tokens (`--num_thought_tokens 2`) embedded into the input.
- **Visual injection**: starting at step 0, every step, insert top-16 attended image patches.
- **Patch budget**: starts at 1 patch per thought token, grows by 1 each time a new best reward is found, capped at 16.

---

## 2. `main.py` — Top-Level Dispatch (`__main__`)

```
main.py:__main__
  ├── parse_args()                      # collect all CLI flags
  ├── if num_workers > 1:
  │     pre-warm processor cache
  │     load dataset for length count
  │     spawn N worker processes → _worker_run()
  │     parent collects results via mp.Queue
  └── else:
        main(args)                      # single-process path
```

Because `--num_workers 2`, the **multiprocessing path** runs:

1. Parent pre-loads the processor to warm the HuggingFace cache (so workers don't fight over downloading).
2. Parent loads the dataset (lazy, no image loading) to get the total item count for the `tqdm` progress bar.
3. Parent splits the dataset index range `[start_data_idx, end)` evenly across workers via `_split_indices()`.
4. Each worker gets its own `(start, end)` slice and a `worker_device_round_robin` GPU assignment (worker 0 → `cuda:0`, worker 1 → `cuda:1`).
5. Workers are spawned with `mp.Process`, each running `_worker_run(wargs, worker_id, result_queue)`.
6. Parent reads from `result_queue`, aggregates results, and periodically writes `results.json` atomically.

---

## 3. `_worker_run()` — Per-Worker Lifecycle

Each worker independently:

### 3a. Load Model and Processor

```python
model = AutoModelForVision2Seq.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float32,
    attn_implementation="eager",   # disables torch.compile; needed for output_attentions=True
    trust_remote_code=True,
)
model.to(device)
model.eval()

processor = AutoProcessor.from_pretrained(
    model_name_or_path,
    padding_side='left',
    min_pixels=128 * 28 * 28,      # pixel range for Qwen's dynamic resolution tiling
    max_pixels=256 * 28 * 28,
)
```

`attn_implementation="eager"` is required because `output_attentions=True` is used during the RL loop, and flash attention does not return attention weights.

### 3b. Build RewardModel

```python
reward_model = RewardModel(
    model=model,
    tokenizer=processor.tokenizer,
    num_thought_tokens=2,
    device=device_str,
)
```

`RewardModel` is a wrapper around the **same model** used for generation. It is only activated when `--disable_conf_reward` is set (which is not the case in this run). In the default run it is instantiated but not called — the confidence-based reward from `get_confidence()` is used instead.

### 3c. Load Dataset Subset

```python
dataset = get_vl_dataset("data/mmvp.json", processor, prompt_idx=0,
                          start_idx=worker_start, end_idx=worker_end)
```

Inside `get_vl_dataset()` (`DMLR/data.py`):
1. Reads the JSON file: a list of `{prompt, solution, image_path, idx}` dicts.
2. Converts each to `{question, answer, image (path string), image_path, idx}`.
3. Applies `dataset.set_transform(add_messages_column)` — a **lazy** transform that fires when an item is accessed.
4. `add_messages_column` wraps the question in a CoT prompt (prompt_idx=0: "Please analyze the image carefully and solve step by step...") and builds the `messages` field in the Qwen chat format:
   ```python
   [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': <cot_question>}]}]
   ```

### 3d. Iterate Examples

For each example in the worker's slice:
```
for local_i in range(len(dataset)):
    example = dataset[local_i]          # triggers lazy transform
    question = example['question']
    image    = example['image']         # path string, e.g. "data/images/mmvp/001.jpg"
    true_answer = extract_true_answer(example['answer'], ...)
    → call generate_vl(...)
    → extract_answer(output)
    → verify correctness
    → result_queue.put(result_dict)
```

---

## 4. `generate_vl()` — The Core Optimization Loop (`DMLR/inference.py:414`)

This is the heart of DMLR. For each example it:

1. **Builds tokenized inputs with latent thought tokens** (`build_vl_inputs`)
2. **Runs an RL loop** for `max_num_steps=15` steps optimizing the latent token embeddings
3. **Generates a final answer** using the best-found latent state

### Step 4.1 — `build_vl_inputs()`

```
Input: question text + image path + num_thought_tokens=2

1. Append 2 × '<|endoftext|>' tokens to the question text as "internal thinking space"
2. Build chat messages:
   [system_prompt, {role:user, content:[{type:image}, {type:text, text:question+thought_tokens}]}]
3. Apply Qwen chat template → text string with image placeholder
4. processor(text, images) → {input_ids, attention_mask, pixel_values, image_grid_thw}
5. Locate the thought token positions in input_ids → thought_idx = [start, end]
6. Append '<|im_start|>assistant\n' generation-prompt tokens to input_ids

Returns: inputs dict, thought_idx = [start_idx, end_idx]
```

The two `<|endoftext|>` tokens at positions `[start_idx, end_idx]` are the **latent variables** being optimized.

### Step 4.2 — Initialize Latent Embeddings

```python
inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
# Extract initial embedding values for the 2 thought-token positions
base_init = inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone()

# Set up Adam optimizer over thought token embeddings
thought_hidden_states = nn.Parameter(base_init.detach().requires_grad_(True))
optimizer = Adam([thought_hidden_states], lr=0.01, maximize=True)
```

Also pre-compute image token positions with `compute_image_token_meta()`:
- Finds `<|vision_start|>` and `<|vision_end|>` boundaries in `input_ids`
- Returns absolute positions of all image patch tokens in the sequence

### Step 4.3 — RL Optimization Loop (15 steps)

Each step does:

#### A. Exploration noise (Evolution-Strategies style)
```python
epsilon = Normal(0, sigma=25.0 * 0.95^step)  # noise decays each step
candidate_latent = thought_hidden_states.detach() + epsilon
```

#### B. Forward pass with candidate latents
```python
inputs_embeds_step = inputs_embeds.clone()
inputs_embeds_step[0, thought_idx[0]:thought_idx[1]] = candidate_latent

outputs = model(
    inputs_embeds=inputs_embeds_step,
    attention_mask=...,
    pixel_values=...,           # raw pixel data for vision tower
    image_grid_thw=...,         # tiling metadata
    output_hidden_states=True,
    output_attentions=True,     # CRITICAL: needed for visual token selection
    use_cache=False,
)
# attentions: list of (1, num_heads, seq_len, seq_len) per layer
# hidden_states[-1]: last-layer hidden states (1, seq_len, hidden_dim)
```

#### C. Attention-based visual token injection

For each thought token (stride=1 means both tokens):
```python
# Average attention across all layers and heads
avg_attention = cat(attentions, dim=1).mean(dim=1)  # (1, seq_len, seq_len)

# For thought token at position `current_thought_idx`:
att_to_images = avg_attention[0, current_thought_idx, image_start:image_end]

# Select top-k image patches by attention weight
# k starts at initial_patch_count=1, grows by patch_increment=1 on each new best
k_limit = min(current_patch_budget, total_image_tokens)
sorted_rel_indices = argsort(att_to_images, descending=True)
chosen_abs_ids = top-k absolute image token positions
```

Then rebuild the embedding sequence by **interleaving** thought tokens with their selected visual tokens:
```
[prefix ... ] [think_token_0] [top-k visual patches for token_0]
              [think_token_1] [top-k visual patches for token_1]
[... suffix]
```

This injects the most-attended image regions directly adjacent to each latent thought token.

#### D. Compute reward (confidence)
```python
reward = get_confidence(model, inputs_step, thought_idx, candidate_latent, k=10)
```

Inside `get_confidence()`:
- Runs another forward pass with the new interleaved embeddings
- Gets logits at the thought-token positions + 1
- Computes **negative mean log top-k probability**: `reward = -mean(log(top_k_probs))`
- Higher reward = more peaked/confident distribution = the model is "ready to answer"

```python
reward.backward()   # gradients flow back to thought_hidden_states
optimizer.step()    # Adam gradient ascent on thought embeddings
sigma *= 0.95       # decay exploration noise
```

#### E. Track best state
```python
if reward_value > best_reward:
    best_reward = reward_value
    best_thought_hidden_states = thought_hidden_states.clone()
    locked_patch_ids = current_step_patch_ids   # lock in this visual selection
    current_patch_budget = min(max_patch_limit, current_patch_budget + patch_increment)
    # budget: 1 → 2 → 3 → ... capped at 16
```

Per-step reward, sigma, patch counts are logged to `reward_logs/reward_steps_XXXXXX.csv`.

### Step 4.4 — Final Generation

After the loop, apply the best latent embeddings and run greedy decoding:

```python
inputs_embeds[0, thought_idx[0]:thought_idx[1]] = best_thought_hidden_states
inputs['inputs_embeds'] = inputs_embeds

outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    do_sample=False,
    num_beams=1,
)
response = processor.decode(outputs[0], skip_special_tokens=True)
```

The model now generates its answer with the optimized "thinking space" conditioning it.

---

## 5. Answer Extraction and Verification

Back in `_worker_run()`:

```python
answer = extract_answer(output)
# Looks for \boxed{...} in the response; falls back to heuristics
```

Correctness check (with `--use_llm_verify`):
```python
is_correct = verify_solution_equivalence(answer, ground_truth)
```

Inside `verify_solution_equivalence()` (`DMLR/verifier.py`):
- Calls an OpenAI-compatible API (configured via `.env`: `OPENAI_API_KEY`, `OPENAI_API_BASE_URL`, `MODEL_TYPE`)
- Uses structured output (`pydantic` `EquivalenceResult`) to get a boolean `equivalent` field
- Fallback to rule-based `judge_answer()` if the API call fails

---

## 6. Output

Each result is sent via `result_queue` to the parent process, which aggregates and writes:

```
output_vis/0325_reproduce/mmvp/Qwen2.5-VL-7B-Instruct/
├── results.json           # all entries + accuracy summary
│   {
│     "accuracy": 0.xx,
│     "correct": N,
│     "total": M,
│     "args": {...},
│     "entries": [
│       {
│         "data_idx": 0,
│         "question": "...",
│         "image_path": "...",
│         "model_output": "...",
│         "answer": "A",
│         "ground_truth": "A",
│         "is_correct": true,
│         "best_reward": 1.234,
│         "best_reward_step": 7,
│         "stop_reason": "eos_token"
│       }, ...
│     ]
│   }
└── reward_logs/
    ├── reward_steps_000000.csv   # step, reward, sigma, is_new_best, patch_count, patch_budget
    ├── reward_steps_000001.csv
    └── ...
```

---

## 7. Data Flow Summary

```
run.sh
  └─► python main.py
        ├─ parse_args → config
        ├─ [MP] spawn 2 workers (cuda:0, cuda:1)
        │
        └─ _worker_run (×2, each on separate GPU)
              ├─ load Qwen2.5-VL-7B  (float32, eager attn)
              ├─ load processor      (min/max_pixels for tiling)
              ├─ build RewardModel   (same model, for fallback only)
              ├─ load dataset slice  (lazy JSON→messages transform)
              │
              └─ for each example:
                    ├─ build_vl_inputs()
                    │    └─ tokenize [system | image | question + 2×<|endoftext|> | <|im_start|>assistant]
                    │
                    ├─ initialize thought embeddings (Adam param)
                    │
                    ├─ for step in range(15):           ← RL LOOP
                    │    ├─ add Gaussian noise → candidate_latent
                    │    ├─ forward pass (output_attentions=True)
                    │    ├─ select top-k attended image patches per thought token
                    │    ├─ rebuild embeddings: interleave thought+visual tokens
                    │    ├─ get_confidence() → reward
                    │    ├─ reward.backward() + Adam.step()
                    │    ├─ sigma *= 0.95
                    │    └─ track best (reward, latent, patch_ids)
                    │
                    ├─ apply best_thought_hidden_states
                    ├─ model.generate() → response text
                    ├─ extract_answer() → boxed answer
                    ├─ verify_solution_equivalence() (LLM judge)
                    └─ result_queue.put(result_dict)
```

---

## 8. Key Design Choices

| Aspect | Choice | Reason |
|--------|--------|--------|
| Thought token init | `<\|endoftext\|>` embeddings | Stable, model-native start point |
| Reward signal | Negative mean log top-k probability | Measures model "readiness" without needing ground truth |
| Optimizer | Adam (maximize=True) | Gradient ascent on confidence reward |
| Visual injection | Attention-selected top-k patches interleaved with thought tokens | Grounds latent reasoning in salient image regions |
| Patch budget growth | Start small (1), increment on new best | Curriculum: start focused, expand when improvement found |
| Noise schedule | Gaussian with exponential decay (`σ × 0.95^t`) | Broad exploration early, fine-tuning later |
| `attn_implementation="eager"` | Required | Flash attention does not return attention weights needed for patch selection |
| `torch_dtype=float32` | Required | BFloat16 causes precision issues with embedding gradients |
