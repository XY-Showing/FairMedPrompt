# FairMedPrompt

Mitigating Medical Bias in Large Language Models by Prompt Engineering — replication package with code, data, and results for an empirical study on fairness, accuracy, and inference overhead in medical QA.

This repository evaluates whether prompt engineering can reduce demographic bias in medical question answering (Medical QA) without model retraining. Using the AMQA benchmark (counterfactual vignette pairs differing only in a sensitive attribute), we test multiple prompt strategies across several influential LLMs, then compute diagnostic accuracy, accuracy gaps between privileged vs. unprivileged groups, statistical significance (McNemar), and cost/latency trade-offs. The accompanying paper is included as `FairMedPrompt.pdf`. Raw outputs used in the study are provided in `Raw_Results.zip`.

## Repository layout:

    FairMedPrompt/
      Script/
        AMQA_Dataset_OpenAI.jsonl
        AMQA_FairMP_Benchmark_LLMs.py
        AMQA_FairMP_Benchmark_LLMs_batch.py
        AMQA_FairMP_Benchmark_LLMs_sys_user.py
        AMQA_FairMP_Benchmark_LLMs_debugging.py
        AMQA_FairMP_Benchmark_LLMs_backup.py
        analysis.py
        analysis_debugging.py
      outputs/                       # put your generated jsonl/csv/plots here
      FairMedPrompt.pdf              # the paper
      Raw_Results.zip                # raw results package
      LICENSE                        # MIT
      README.md

## Models and prompts:

- Model keys used by scripts: `gpt`, `claude`, `gemini`, `qwen`, `deepseek`. The analysis script maps them to display labels (e.g., GPT-5-Mini, Claude-Sonnet-4, Gemini-2.5-Flash, Qwen-3, DeepSeek-V3.1).
- Prompt types: `naive` (baseline), `role` (role-playing), `aware` (bias-aware), `cot` (chain-of-thought). Few-shot is analyzed in the downstream analysis; you can extend the runner to add online few-shot if needed.

<p align="center">
  <img src="outputs/charts/Prompts.pdf" alt="Prompts used in this project." width="720">
  <br><em>Figure: Overview of FairMedPrompt</em>
</p>


Install:

    pip install openai anthropic google-generativeai dashscope tqdm pandas numpy matplotlib seaborn scipy

## API keys (edit the script to read from env or paste keys at the top):

    # recommended: environment variables
    export OPENAI_API_KEY=xxx
    export ANTHROPIC_API_KEY=xxx
    export GOOGLE_API_KEY=xxx
    export DASHSCOPE_API_KEY=xxx        # Qwen (DashScope)
    export DEEPSEEK_API_KEY=xxx         # DeepSeek (OpenAI-compatible base_url)

    # notes:
    # - deepseek uses an OpenAI-compatible endpoint; see the script for base_url.
    # - provider access, quotas, and billing constraints apply.

## Run benchmarks (from the `Script/` directory). The runner emits two files per (model × prompt):
- answers: `FairMP_Answer_{model}_{prompt}.jsonl`
- summary: `FairMP_Summary_{model}_{prompt}.jsonl`

Examples:

    cd Script

    # single run
    python AMQA_FairMP_Benchmark_LLMs.py --model qwen --prompt cot
    python AMQA_FairMP_Benchmark_LLMs.py --model deepseek --prompt aware

    # sweep
    for m in gpt claude gemini qwen deepseek; do
      for p in naive role aware cot; do
        python AMQA_FairMP_Benchmark_LLMs.py --model $m --prompt $p
      done
    done

## Controlling workload and costs (edit top of `AMQA_FairMP_Benchmark_LLMs.py`):

    MODE = "test"      # "test" runs a subset; set "full" for the entire dataset
    START_INDEX = 0
    PROCESS_NUM  = 20  # number of items to process in "test" mode
    LIMIT = START_INDEX + PROCESS_NUM

## Analysis (aggregations, significance tests, and plots):

    # still in Script/
    python analysis.py

The analysis script will scan for `FairMP_Answer_*.jsonl`, compute per-variant accuracy, privileged–unprivileged accuracy gaps, absolute/relative gap change vs. baseline, answer-change ratios, McNemar p-values, and render several charts (bar grids, heatmaps, radar, Pareto scatter). CSVs and images are saved to the working directory (recommend moving them into `../outputs/`).

## AMQA JSONL schema (one item per line):

- `question_id`: unique ID
- `options`: dict of options, e.g. `{"A": "...", "B": "...", "C": "...", "D": "..."}`  
- `answer_idx`: the correct option key (e.g., `"B"`)
- `original_question`, `desensitized_question`
- `adv_question_white`, `adv_question_black`
- `adv_question_high_income`, `adv_question_low_income`
- `adv_question_male`, `adv_question_female`

The runner queries each of these fields (when present) and records normalized model choices as `test_model_answer_*` keys in the output JSONL.

## Metrics and statistics:

- Accuracy gap: `AG = Acc(privileged) − Acc(unprivileged)` (group accuracies computed over matched counterfactual items).
- Absolute gap change: `ΔAG = AG_strategy − AG_baseline`.
- Relative gap change: `ΔrelAG = (AG_strategy − AG_baseline) / AG_baseline`. Negative values mean bias reduction.
- Significance: paired McNemar test (`p < 0.05`) on inequity outcomes (baseline vs. strategy).

## Key findings from the paper (summarized):

- No single strategy is universally effective across models/attributes; effects are strongly model-dependent, and some prompts can worsen bias.
- Chain-of-thought achieves the largest average bias reduction, decreasing the overall mean gap by about 2.4 percentage points; it also yields the most significant cases (8/15), largely by improving unprivileged-group accuracy. The largest single reduction observed is 6.2 percentage points (DeepSeek-V3.1–Sex).
- Improvements are generally Pareto-favorable: privileged-group performance is preserved while unprivileged-group performance increases (e.g., DeepSeek-V3.1–Female +9 percentage points).
- Overhead varies: role-playing and bias-aware add little cost; few-shot can reduce overhead in “thinking” models; chain-of-thought reliably increases output tokens and wall-time (e.g., on Qwen-3, single-inference output tokens increased from ~2 to ~72 and total script time increased by ~507% in the study).

## Practical notes:

- Start with `MODE="test"` to validate setup and control spend; then switch to `full` if needed.
- Fix temperatures and model versions for reproducibility.
- This repository is for research replication only; it must not be used for clinical decision making.



License: MIT (see `LICENSE`).

Contact: Ying Xiao — ying.1.xiao@kcl.ac.uk. Issues and PRs are welcome.

