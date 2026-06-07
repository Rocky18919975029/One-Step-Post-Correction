# Legacy Experiment Utilities

This folder contains archived scripts from the older single-shot sampling pipeline:

- `power_samp_*.py`
- `eval_*.py`
- `passk_*.py`
- `response_length_stats.py`
- `scripts/power_samp_*.sh`

They are kept for historical reference and occasional reproduction, but they are not part of the maintained training path.

The maintained workflow lives in the parent directory and centers on:

- `blockwise_vllm_sample_buffer.py`
- `blockwise_power_tb_buffer_train.py`
- `blockwise_power_tb_train.py`

When running a legacy Python script directly, use the repository root or `llm_experiments` on `PYTHONPATH`.
