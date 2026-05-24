"""Generate LLM-paraphrased oversampled datasets for the llm_paraphrasing strategy.

Run once to populate DATA_DIR/tokenized/llm_paraphrasing/. Supports checkpoint/resume
via per-pass JSONL files — safe to interrupt and re-run.

    python scripts/paraphrasing.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

_MIN_LENGTH_RATIO   = 0.25   # paraphrase must be at least 25% as long as the original
_MIN_WORD_OVERLAP   = 0.10   # at least 10% of content word types must appear in paraphrase
BERTSCORE_THRESHOLD = 0.85   # semantic similarity threshold (distilbert F1)
MAX_PASSES          = 15     # cap on LLM passes to prevent repetitiveness on high-imbalance datasets

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "that", "this",
    "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "she", "his", "her", "they", "their", "them", "what",
    "which", "who", "not", "no", "so", "if", "as", "up", "out", "about",
    "into", "than", "then", "just", "also", "more", "very", "only", "s",
}


def _content_words(text: str) -> set[str]:
    return {w for w in text.lower().split() if w not in _STOPWORDS and len(w) > 1}


def _is_valid_paraphrase(paraphrase: str, original: str) -> bool:
    orig_tokens = original.lower().split()
    para_tokens = paraphrase.lower().split()
    if not orig_tokens:
        return True
    if len(para_tokens) / len(orig_tokens) < _MIN_LENGTH_RATIO:
        return False
    orig_content = _content_words(original)
    if not orig_content:
        return True
    overlap = len(_content_words(paraphrase) & orig_content) / len(orig_content)
    return overlap >= _MIN_WORD_OVERLAP

# ── Hardware & Model ─────────────────────────────────────────────────────────
GPU_INDEX    = 0
MODEL_ID     = "google/gemma-4-e2b-it"
QUANTIZATION = None   # None = BF16 native; "awq" for 26B variant

# ── Inference ────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 256
CHUNK_SIZE     = 10_000

os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(GPU_INDEX))
# Point gcc to uv's Python 3.12 headers (python3.12-dev not installed system-wide)
os.environ.setdefault(
    "CPATH",
    "/home/phmgc/.local/share/uv/python/"
    "cpython-3.12.11-linux-x86_64-gnu/include/python3.12",
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)

_PROMPT_PREFIX = (
    "<|turn>user\nParaphrase the following review while preserving its "
    "sentiment and meaning. Provide only the paraphrased text.\n"
    "Review: \""
)
_PROMPT_SUFFIX  = "\"<turn|>\n<|turn>model\n"
_MAX_INPUT_TOKENS = 4096 - MAX_NEW_TOKENS - 10


def load_processed_ids(cache_path: Path) -> set:
    if not cache_path.exists():
        return set()
    processed = set()
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                processed.add(json.loads(line)["id"])
    return processed


def _build_prompt(text: str, tokenizer) -> str:
    ids        = tokenizer.encode(text, add_special_tokens=False)
    prefix_ids = tokenizer.encode(_PROMPT_PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(_PROMPT_SUFFIX, add_special_tokens=False)
    budget     = _MAX_INPUT_TOKENS - len(prefix_ids) - len(suffix_ids)
    if len(ids) > budget:
        ids  = ids[:budget]
        text = tokenizer.decode(ids, skip_special_tokens=True)
    return _PROMPT_PREFIX + text + _PROMPT_SUFFIX


def _read_cache(cache_path: Path) -> list:
    if not cache_path.exists():
        return []
    results = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def paraphrase_stream(samples, llm, sampling_params, cache_path):
    tok           = llm.get_tokenizer()
    processed_ids = {r["id"] for r in _read_cache(cache_path)}
    pending       = [s for s in samples if s["id"] not in processed_ids]

    existing_results = _read_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    new_results = []
    for i in range(0, len(pending), CHUNK_SIZE):
        chunk   = pending[i : i + CHUNK_SIZE]
        prompts = [_build_prompt(s["text"], tok) for s in chunk]
        outputs = llm.generate(prompts, sampling_params)

        with open(cache_path, "a", encoding="utf-8") as f:
            for s, out in zip(chunk, outputs):
                r = {"id": s["id"], "text": out.outputs[0].text.strip(), "label": s["label"]}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                new_results.append(r)

        logging.info("  chunk %d: %d samples → %s",
                     i // CHUNK_SIZE + 1, len(chunk), cache_path.name)

    return existing_results + new_results


def paraphrase_minority_class(raw_train, get_llm, cache_dir, seed):
    import evaluate
    from datasets import Dataset, concatenate_datasets

    labels         = [int(x) for x in raw_train["label"]]
    class_counts   = Counter(labels)
    minority_label = min(class_counts, key=class_counts.get)
    majority_count = max(class_counts.values())
    needed         = majority_count - class_counts[minority_label]

    logging.info("  class distribution: %s", dict(class_counts))
    logging.info("  paraphrases needed: %d", needed)

    minority_samples = [
        {"id": f"orig_{i}", "text": raw_train[i]["text"], "label": minority_label}
        for i, lbl in enumerate(labels) if lbl == minority_label
    ]

    bs_metric       = evaluate.load("bertscore")
    all_paraphrases = []
    pass_idx        = 0

    while len(all_paraphrases) < needed:
        if pass_idx >= MAX_PASSES:
            logging.warning(
                "  MAX_PASSES (%d) reached — partial balance: %d/%d paraphrases generated. "
                "Proceeding with available samples.",
                MAX_PASSES, len(all_paraphrases), needed,
            )
            break

        remaining  = needed - len(all_paraphrases)
        cache_path = cache_dir / f"pass_{pass_idx:03d}.jsonl"
        samples_with_ids = [{**s, "id": f"p{pass_idx}_{s['id']}"} for s in minority_samples]
        logging.info("  pass %d | remaining: %d | cache: %s",
                     pass_idx, remaining, cache_path.name)
        cached_ids = {r["id"] for r in _read_cache(cache_path)}
        pending    = [s for s in samples_with_ids if s["id"] not in cached_ids]
        if pending:
            llm, sampling_params = get_llm()
            new = paraphrase_stream(samples_with_ids, llm, sampling_params, cache_path)
        else:
            new = _read_cache(cache_path)

        orig_by_id = {s["id"]: s["text"] for s in samples_with_ids}
        valid = [
            p for p in new
            if _is_valid_paraphrase(p["text"], orig_by_id[p["id"]])
        ]
        heuristic_discarded = len(new) - len(valid)
        if heuristic_discarded:
            logging.info("  pass %d: heuristic filter discarded %d/%d paraphrases",
                         pass_idx, heuristic_discarded, len(new))

        if valid:
            scores = bs_metric.compute(
                predictions=[p["text"] for p in valid],
                references=[orig_by_id[p["id"]] for p in valid],
                model_type="distilbert-base-uncased",
                lang="en",
            )
            before_bert = len(valid)
            valid = [p for p, s in zip(valid, scores["f1"]) if s >= BERTSCORE_THRESHOLD]
            bert_discarded = before_bert - len(valid)
            if bert_discarded:
                logging.info("  pass %d: BERTScore filter discarded %d/%d paraphrases",
                             pass_idx, bert_discarded, before_bert)

        all_paraphrases.extend(valid[:remaining])
        pass_idx += 1

    synthetic = Dataset.from_dict({
        "text":  [p["text"]  for p in all_paraphrases],
        "label": [p["label"] for p in all_paraphrases],
    })
    combined     = concatenate_datasets([raw_train, synthetic]).shuffle(seed=seed)
    after_counts = Counter(int(x) for x in combined["label"])
    return combined, class_counts, after_counts


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from src.config import DATASETS, MODEL_NAME, SEED, TRAIN_RATIO
    from src.data_utils import _format_and_save, _load_raw_splits
    from src.paths import DATA_DIR

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    _llm_cache: list = []

    def get_llm():
        if not _llm_cache:
            logging.info("Loading vLLM engine: %s", MODEL_ID)
            llm = LLM(model=MODEL_ID, quantization=QUANTIZATION,
                      tensor_parallel_size=1, max_model_len=4096)
            sp  = SamplingParams(
                temperature=0.4, top_p=0.9, top_k=64,
                max_tokens=MAX_NEW_TOKENS, stop=["<turn|>", "\n\n"],
            )
            _llm_cache.extend([llm, sp])
        return _llm_cache[0], _llm_cache[1]

    for author, dataset_names in DATASETS.items():
        for name in dataset_names:
            output_path = DATA_DIR / "tokenized" / "llm_paraphrasing" / author / name / "train"
            if output_path.exists():
                logging.info("skip (already tokenized): %s/%s", author, name)
                continue

            logging.info("\n%s\nDataset: %s/%s\n%s", "="*60, author, name, "="*60)

            raw_train, _ = _load_raw_splits(author, name, TRAIN_RATIO, SEED)
            cache_dir    = DATA_DIR / "paraphrase_cache" / author / name

            combined_raw, before_counts, after_counts = paraphrase_minority_class(
                raw_train, get_llm, cache_dir, seed=SEED,
            )
            _format_and_save(combined_raw, tokenizer, output_path, max_length=256)
            logging.info("done: %s -> %s | saved: %s",
                         dict(before_counts), dict(after_counts), output_path)


if __name__ == "__main__":
    main()
