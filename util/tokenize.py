import json


def get_alpha_vocab(tokenizer):
    """提取 95 個可打印字符的 token 映射"""
    PW_WORD = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./;<=>?@[\\]^_`{|}~ "
    vocab = {}
    for w in PW_WORD:
        vocab[w] = tokenizer(w)["input_ids"][-1]
    vocab[tokenizer.eos_token] = tokenizer.eos_token_id
    vocab["\t"] = tokenizer.eos_token_id  # \t 作為 EOS marker
    return vocab

def encode_limit(input_str, vocab):
    new_str = input_str.replace("</s>", "\t")  # 處理 EOS token 在密碼中
    ret = [0 for i in range(len(new_str))]
    for i in range(len(new_str)):
        if new_str[i] == "\t":
            ret[i] = vocab["\t"]
        elif new_str[i] not in vocab.keys():
            ret[i] = 0  # <unk>
        else:
            ret[i] = vocab[new_str[i]]  
    return {
        "input_ids": ret,
        "attention_mask": [1 for i in range(len(ret))]
    }

def process_train_targeted(example, tokenizer, prompt_template, vocab, max_length=512):
    account = str(example.get("account", ""))
    password = example.get("password")
    if password is None:
        password = example.get("passwords", "")
    password = str(password)

    prompt_ids = tokenizer(prompt_template, add_special_tokens=False)["input_ids"]
    if tokenizer.bos_token_id is not None:
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids

    knowledge_text = json.dumps({"Username": account}, ensure_ascii=False)
    knowledge_ids = tokenizer(knowledge_text, add_special_tokens=False)["input_ids"]
    password_ids = encode_limit(password, vocab)["input_ids"]
    input_ids = prompt_ids + knowledge_ids + password_ids
    if tokenizer.eos_token_id is not None:
        input_ids = input_ids + [tokenizer.eos_token_id]

    input_ids = input_ids[:max_length]
    attention_mask = [1] * len(input_ids)

    labels = input_ids.copy()
    mask_upto = min(len(prompt_ids) + len(knowledge_ids), len(labels))
    labels[:mask_upto] = [-100] * mask_upto



    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
