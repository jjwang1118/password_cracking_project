import torch
import json
import yaml
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

from util.tokenize import get_alpha_vocab
from util.prompt_template import _get_indice
from util.search import contrastive_search


def build_vocab_list(vocab_dict: dict) -> list:
    """將 get_alpha_vocab 返回的 {char: token_id} 轉成 list，EOS 放最後"""
    eos_id = vocab_dict["\t"]
    seen = set()
    char_ids = []
    for k, v in vocab_dict.items():
        if k in ("\t", "</s>"):
            continue
        if v not in seen:
            seen.add(v)
            char_ids.append(v)
    char_ids.append(eos_id)
    return char_ids


def decode_password(seq_tensor: torch.Tensor, inv_vocab: dict, eos_id: int) -> str:
    """將 token id sequence 解碼成密碼字串（遇到 EOS 停止）"""
    pw_chars = []
    for tid in seq_tensor.tolist():
        if tid == eos_id:
            break
        pw_chars.append(inv_vocab.get(tid, "?"))
    return "".join(pw_chars)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    eval_config      = config["eval"]["config"]
    model_name       = eval_config["model_name"]
    batch_size       = eval_config["batch_size"]
    prompt_tmpl_id   = eval_config["prompmt_template_id"]
    precision        = eval_config["precistion"]
    eos_threshold    = eval_config["eos_threshold"]
    max_guess_number = eval_config["max_guess_number"]

    project_root    = Path(__file__).parent
    output_path     = project_root / "gen"
    output_path.mkdir(parents=True, exist_ok=True)

    test_path        = project_root / config["dataset_path"] / "split" / "test_data.jsonl"
    local_model_path = project_root / "model" / model_name
    lora_final_path  = project_root / "checkpoints" / model_name / "lora_final"
    max_pw_length   = 30
    beam_width      =  [95,1000]+[1000]*14
    search_width    = 200   
    contrastive_alpha = 0.6

    beam_width_list   = [beam_width]  * max_pw_length
    search_width_list = [search_width] * max_pw_length

    # ========== 載入模型 ==========
    print(f"[INFO] Loading model from {local_model_path}")
    model_source = str(local_model_path) if local_model_path.exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"half": torch.float16, "bf16": torch.bfloat16, "full": torch.float32}
    dtype = dtype_map.get(precision, torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=dtype,
        device_map="auto",
    )

    if lora_final_path.exists():
        print(f"[INFO] Loading LoRA weights from {lora_final_path}")
        model = PeftModel.from_pretrained(base_model, str(lora_final_path))
        model = model.merge_and_unload()
    else:
        print("[WARN] lora_final not found, using base model")
        model = base_model

    model.eval()
    device = next(model.parameters()).device

    # ========== 建立詞彙表 ==========
    vocab_dict = get_alpha_vocab(tokenizer)
    vocab_list = build_vocab_list(vocab_dict)
    eos_id     = vocab_dict["\t"]
    inv_vocab  = {v: k for k, v in vocab_dict.items() if k not in ("\t", "</s>")}
    print(f"[INFO] Vocab size: {len(vocab_list)} (including EOS)")

    prompt_template = _get_indice(prompt_tmpl_id)

    print(f"\n[INFO] Search Configuration:")
    print(f"  beam_width      : {beam_width}")
    print(f"  search_width    : {search_width}")
    print(f"  max_pw_length   : {max_pw_length}")
    print(f"  eos_threshold   : {eos_threshold}")
    print(f"  batch_size      : {batch_size}")
    print(f"  max_guess_number: {max_guess_number}\n")

    # ========== 載入測試資料並評估 ==========
    dataset = load_dataset("json", data_files={"test": str(test_path)})["test"]
    hit     = 0
    total   = len(dataset)
    results = []

    pbar = tqdm(dataset, desc="Evaluating", total=total)
    for example in pbar:
        account     = str(example.get("account", ""))
        gt_password = str(example.get("password") or example.get("passwords", ""))

        prompt_ids     = tokenizer(prompt_template, add_special_tokens=False)["input_ids"]
        if tokenizer.bos_token_id is not None:
            prompt_ids = [tokenizer.bos_token_id] + prompt_ids
        knowledge_text = json.dumps({"Username": account}, ensure_ascii=False)
        knowledge_ids  = tokenizer(knowledge_text, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor(
            [prompt_ids + knowledge_ids], dtype=torch.long, device=device
        )

        candidates = contrastive_search(
            model=model,
            input_ids=input_ids,
            batch_size=batch_size,
            beam_width_list=beam_width_list.copy(),
            vocab=vocab_list,
            eos_threshold=eos_threshold,
            max_length=max_pw_length,
            search_width_list=search_width_list.copy(),
            sorted=True,
            use_contrastive=True,
            contrastive_alpha=contrastive_alpha,
        )

        guesses = [
            decode_password(seq, inv_vocab, eos_id)
            for seq, _ in candidates[:max_guess_number]
        ]

        is_hit = gt_password in guesses
        if is_hit:
            hit += 1
        pbar.set_postfix({"hit": hit, "rate": f"{hit/max(len(results)+1,1):.2%}"})
        results.append({
            "account":      account,
            "gt_password":  gt_password,
            "top1_guess":   guesses[0] if guesses else "",
            "top5_guesses": guesses[:5],
            "hit":          is_hit,
            "rank":         guesses.index(gt_password) + 1 if is_hit else -1,
        })

    hit_rate = hit / total if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"[INFO] Evaluation Results:")
    print(f"  Total    : {total}")
    print(f"  Hit      : {hit}")
    print(f"  Hit rate : {hit_rate:.4f} ({hit_rate*100:.2f}%)")
    print(f"{'='*50}\n")

    output_file = output_path / f"{model_name}_eval_results.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] Results saved to {output_file}")

