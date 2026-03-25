## 實作constrative search
## 獨立kv cache

import numpy as np
import torch
import math
import random
import time
import os
from queue import PriorityQueue
from collections import defaultdict,deque
from dataclasses import dataclass, field

import torch
from tqdm import tqdm


from transformers import (
    MistralForCausalLM
)

from src.utils.tokenize import process_test_trawling



def _reorder_cache(past_key_values, beam_idx):#copy from transformers
    """
    Reorder past key values according to beam indices.
    """
    if not past_key_values:
        return past_key_values
    reordered_past = ()
    for layer_past in past_key_values:
        reordered_past += (
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
        )
    return reordered_past


def _merge_cache(past_key_values, cache_2):
    """
    Merge two sets of past key values along the batch dimension.
    """
    if not past_key_values:
        return cache_2
    merged_cache = ()
    for layer_idx in range(len(past_key_values)):
        layer_cache = tuple(
            torch.cat([past_state_1, past_state_2], dim=0)
            for past_state_1, past_state_2 in zip(past_key_values[layer_idx], cache_2[layer_idx])
        )
        merged_cache += (layer_cache,)
    return merged_cache


def _cache_slice(past_key_values, prompt):
    """
    Slice past key values using the given prompt indices.
    """
    sliced_cache = ()
    for layer_idx in range(len(past_key_values)):
        layer_slice = tuple(
            past_state[:, :, prompt, :] for past_state in past_key_values[layer_idx]
        )
        sliced_cache += (layer_slice,)
    return sliced_cache


def _cache_concat(past_key_values_head, past_key_values_tail):
    """
    Concatenate two sets of past key values along the sequence length dimension.
    """
    if not past_key_values_tail:
        return past_key_values_head
    concatenated_cache = ()
    for layer_idx in range(len(past_key_values_head)):
        layer_cache = tuple(
            torch.cat([past_state_1, past_state_2], dim=2)
            for past_state_1, past_state_2 in zip(
                past_key_values_head[layer_idx], past_key_values_tail[layer_idx]
            )
        )
        concatenated_cache += (layer_cache,)
    return concatenated_cache

def remap_logits(vocab, logits: torch.Tensor):
    '''
    vocab: list, custom vocabulary
    logits: tensor, raw model output logits of shape [batch_size, seq_length, primary_vocab_size]
    output: tensor, logits corresponding only to tokens in the provided vocabulary
    '''
    dtype, device = logits.dtype, logits.device
    # Keep only the logits corresponding to characters in the custom vocabulary, ignoring others
    filtered_logits = logits[:, :, vocab].to(device)
    return filtered_logits


@torch.no_grad()
def predict_next(model, vocab, input_ids, past_key_values=None):
    #predict and vocab limit
    #
    outputs = model.forward(
        input_ids=input_ids,
        past_key_values=past_key_values,
        output_attentions=False,
        output_hidden_states=True,
        use_cache=True
    )
    logits = remap_logits(vocab, outputs.logits)[:, -1, :]
    word_prob = logits.log_softmax(dim=1)

    return word_prob, outputs.past_key_values , outputs.hidden_states




class DBS_beam_contrastive_search:
    def __init__(self, info_cache: tuple, vocab_tensor: torch.Tensor, device, initial_hidden: torch.Tensor):
        """
        Args:
            info_cache: prompt 的 KV cache
            vocab_tensor: 詞彙表 tensor
            device: 設備
            initial_hidden: 初始 prompt 的 hidden state [1, hidden_dim]
        """
        self.info_cache = info_cache
        self.pw_cache = None
        self.pw_idx = torch.empty(1, 0, device=device, dtype=torch.int)
        self.beam_prob = torch.zeros(1, 1, device=device, dtype=torch.double)
        self.search_prob = torch.zeros(1, 1, device=device, dtype=torch.double)
        self.vocab_tensor = vocab_tensor
        self.device = device
        
        # 每個 beam 獨立的歷史: [beam_width, history_len, hidden_dim]
        # 初始時只有 1 個 beam，歷史長度為 1
        self.acc_hidden_state = initial_hidden.unsqueeze(1)  # [1, 1, hidden_dim]

    def return_beam_width(self):
        return self.pw_idx.shape[0]
    
    def accumulate_hidden(self, hidden_states: torch.Tensor):
        """
        為每個 beam 累積其對應的隱藏狀態
        
        Args:
            hidden_states: 當前步驟所有 beam 的隱藏狀態 [beam_width, hidden_dim]
        """
        # hidden_states: [beam_width, hidden_dim] -> [beam_width, 1, hidden_dim]
        new_hidden = hidden_states.unsqueeze(1)
        # 沿 history 維度拼接: [beam_width, history_len+1, hidden_dim]
        #水平拼接，沿著SEQ增加
        #沿著一為增加，對應自己的歷史紀錄
        self.acc_hidden_state = torch.cat([self.acc_hidden_state, new_hidden], dim=1)
        
    def compute_contrastive_penalty(self, current_hidden: torch.Tensor, beam_indices: torch.Tensor, alpha: float = 0.6):
        """
        基於每個 beam 自己的歷史隱藏狀態計算對比懲罰
        
        Args:
            current_hidden: 當前候選的隱藏狀態 [batch_size, hidden_dim]
            beam_indices: 每個候選對應的父 beam 索引 [batch_size]
            alpha: 對比懲罰權重
            
        Returns:
            penalty: 對比懲罰分數 [batch_size]
        """
        batch_size = current_hidden.shape[0]
        
        if self.acc_hidden_state.shape[1] == 0:
            # 沒有歷史記錄，返回零懲罰
            return torch.zeros(batch_size, device=current_hidden.device)
        
        # 取出每個候選對應的父 beam 的歷史 [batch_size, history_len, hidden_dim]
        beam_histories = self.acc_hidden_state[beam_indices]
        
        # 正規化當前隱藏狀態 [batch_size, hidden_dim]
        current_norm = torch.nn.functional.normalize(current_hidden, p=2, dim=-1)
        
        # 正規化歷史隱藏狀態 [batch_size, history_len, hidden_dim]
        history_norm = torch.nn.functional.normalize(beam_histories, p=2, dim=-1)
        
        # 計算餘弦相似度: [batch_size, hidden_dim] @ [batch_size, hidden_dim, history_len]
        # -> [batch_size, history_len]
        similarity = torch.bmm(
            current_norm.unsqueeze(1),  # [batch_size, 1, hidden_dim]
            history_norm.transpose(1, 2)  # [batch_size, hidden_dim, history_len]
        ).squeeze(1)  # [batch_size, history_len]
        
        # 對每個候選取最大相似度 [batch_size]
        max_similarity = similarity.max(dim=-1)[0]
        
        # 計算懲罰：penalty = alpha * max_similarity
        penalty = alpha * max_similarity
        
        return penalty

    def update_by_prob(self, beam_width, search_width: int, probs: torch.Tensor, pw_past_key_values, hidden_states: torch.Tensor = None):
        """
        Update beam state based on probability
        
        Args:
            beam_width: 保留的 beam 數量
            search_width: 搜索寬度
            probs: 機率分數 [current_beam_width, vocab_size-1]
            pw_past_key_values: KV cache
            hidden_states: 當前步驟的 hidden states [current_beam_width, hidden_dim]
        """

        tot_probs = (self.beam_prob.reshape(-1, 1) + probs).reshape(-1)


        self.search_prob, self.search_idx = torch.topk(tot_probs, search_width, largest=True)
 
        self.beam_prob, self.beam_idx = torch.topk(tot_probs, beam_width, largest=True)
        self.search_prob = self.search_prob.reshape(-1, 1)
        self.beam_prob = self.beam_prob.reshape(-1, 1)

        self.last_beam_index = self.search_idx // (self.vocab_tensor.shape[0] - 1)  # Subtract 1 to remove the EOS token
        word_index = torch.remainder(self.search_idx, self.vocab_tensor.shape[0] - 1)
        
        self.beam_parent_index = self.beam_idx // (self.vocab_tensor.shape[0] - 1)


        self.pw_idx = torch.cat([
            self.pw_idx[self.last_beam_index],
            self.vocab_tensor[word_index].reshape(-1, 1)
        ], dim=1)

        self.pw_cache = pw_past_key_values
        
  
        if hidden_states is not None:

            self.acc_hidden_state = self.acc_hidden_state[self.beam_parent_index]
            
            # 取出被選中的 beam 對應的 hidden states 並累積
            # self.beam_idx 對應到的父節點 
            selected_hidden = hidden_states[self.beam_parent_index]  # [beam_width, hidden_dim]
            self.accumulate_hidden(selected_hidden)
        else:

            self.acc_hidden_state = self.acc_hidden_state.expand(beam_width, -1, -1).clone()
        
@torch.no_grad()
def contrastive_search(
    model: MistralForCausalLM,
    input_ids: torch.Tensor,
    batch_size: int,
    beam_width_list: list = None,       
    vocab: list = None,
    eos_threshold: float = 0.001,
    threshold: float = None,            
    max_length: int = None,               
    search_width_list: list = [],
    sorted: bool = True,
    use_contrastive: bool = True,
    contrastive_alpha: float = 0.6,
    top_k: int = None,
):
    """
    Contrastive Search: 使用 threshold-based 搜索生成大量候選，再用 contrastive penalty 重新排序。
    
    Args:
        model: 語言模型
        input_ids: prompt 的 input_ids
        batch_size: 批次大小
        beam_width_list: beam 寬度列表 (如果提供，使用 dynamic_beam_search)
        vocab: 詞彙表
        eos_threshold: EOS 閾值
        threshold: 機率閾值 (用於 _width_search，可生成百萬級密碼)
        max_length: 最大長度
        search_width_list: search 寬度列表
        sorted: 是否排序
        use_contrastive: 是否使用 contrastive reranking
        contrastive_alpha: contrastive penalty 強度
        top_k: 只返回前 k 個結果
        
    Returns:
        candidates: 排序後的候選列表
    """
    if not search_width_list:
        search_width_list = beam_width_list

    device = model.device
    eos_threshold = torch.tensor(math.log(eos_threshold), device=device)

    input_ids = input_ids.reshape(1, -1).to(device=device)
    reorder_info_cache_index = torch.zeros(batch_size, device=device, dtype=torch.int)

    # 如果 beam_width_list 為 None，使用 max_length 和預設 beam_width 自動生成
    if beam_width_list is None:
        if max_length is None:
            raise ValueError("beam_width_list 和 max_length 不能同時為 None")
        default_beam_width = 100  # 預設 beam 寬度
        beam_width_list = [default_beam_width] * max_length
        if not search_width_list:
            search_width_list = beam_width_list
    
    max_length = len(beam_width_list)
    pw_silce_index = torch.arange(
        input_ids.shape[1], input_ids.shape[1] + max_length, device=device, dtype=torch.int
    )
    vocab_tensor = torch.tensor(vocab, device=device, dtype=torch.int)
    vocab_size = len(vocab)
    
    # 調整 beam_width_list：第一層受詞彙表限制，後續層受前一層 beam 數量限制
    # 第0層：最多 vocab_size-1 個候選（去掉 EOS）
    beam_width_list[0] = min(beam_width_list[0], vocab_size - 1)
    
    # 第i層：最多 beam_width[i-1] × (vocab_size-1) 個候選
    for i in range(1, len(beam_width_list)):
        max_candidates = beam_width_list[i-1] * (vocab_size - 1)
        beam_width_list[i] = min(beam_width_list[i], max_candidates)
    
    # 同樣調整 search_width_list
    if search_width_list:
        search_width_list[0] = min(search_width_list[0], vocab_size - 1)
        for i in range(1, len(search_width_list)):
            max_candidates = beam_width_list[i-1] * (vocab_size - 1)
            search_width_list[i] = min(search_width_list[i], max_candidates)

    #Get the auxiliary information  cache
    outputs = model.forward(
        input_ids=input_ids,
        past_key_values=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=True,
    )
    logits = remap_logits(vocab_tensor, outputs.logits)[:, -1, :]
    info_cache = outputs.past_key_values
    
    # 儲存初始 prompt 的 hidden state 用於對比搜索
    initial_hidden = outputs.hidden_states[-1][:, -1, :]
    
    del outputs

    word_probs = torch.nn.functional.log_softmax(logits, dim=1)
    eos_list = []
    word_probs = word_probs[:, :-1]
    pw_past_key_values = ()
    
    # 用初始 hidden state 創建 beam
    beam = DBS_beam_contrastive_search(info_cache, vocab_tensor, device, initial_hidden)
    
    # 第一層：使用初始 word_probs 更新 beam
    reserve_width = max(beam_width_list[0], search_width_list[0])
    beam.update_by_prob(beam_width_list[0], reserve_width, word_probs, pw_past_key_values)
    
    for l in range(max_length):
        #Divide the beam width into multiple batches and perform forward passes sequentially.
        reserve_width = max(beam_width_list[l], search_width_list[l])
        #計算總共需要多少個batch
        forward_num = math.ceil(reserve_width / batch_size)
        # 計算需要多少次forward
        beam_forward_num = math.ceil(beam_width_list[l] / batch_size)
        
        word_probs = torch.empty(0, vocab_size - 1, device=device)
        pw_past_key_values = ()
        hidden_states_batch = []  # 收集當前步驟的 hidden states
        
        for i in range(forward_num):
            if i < beam_forward_num:
                start, end = i * batch_size, min((i + 1) * batch_size, beam_width_list[l])
            else:
                start = beam_width_list[l] + (i - beam_forward_num) * batch_size
                end = min(beam_width_list[l] + (i - beam_forward_num + 1) * batch_size, reserve_width)

            input_seqs = beam.pw_idx[start:end, :]
            input_ids = beam.pw_idx[start:end, -1:]
            
            # 取得當前 batch 對應的父 beam 索引（用於 cache 重排）
            cache_beam_indices = beam.last_beam_index[start:end]
            
            # 計算對比懲罰
            if i < beam_forward_num:

                current_beam_indices = torch.arange(start, end, device=device)

            #Concatenate info_cache and pw_cache before feeding them into the model.
            cache = _cache_concat(
                _reorder_cache(beam.info_cache, reorder_info_cache_index[:end - start]),
                _reorder_cache(beam.pw_cache, cache_beam_indices)
            )

            outputs = model.forward(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=True,  # 啟用 hidden states 輸出
            )
            del cache
  
            # 提取當前 batch 的 hidden states [batch_size, hidden_dim]
            current_hidden = outputs.hidden_states[-1][:, -1, :]
            
            # 計算對比懲罰 [batch_size]
            # 只有 beam_width 內的 beam 需要 penalty（search_width 中額外的部分不需要）
            if i < beam_forward_num:
                penalty = beam.compute_contrastive_penalty(current_hidden, current_beam_indices, contrastive_alpha)
            else:
                # search_width 中額外的 beam 不需要計算 penalty（不會保存到 word_probs）
                penalty = torch.zeros(current_hidden.shape[0], device=device)
            
            # 詞彙過濾與機率計算
            logits = remap_logits(vocab_tensor, outputs.logits)[:, -1, :]
            batch_word_probs = torch.nn.functional.log_softmax(logits, dim=1)
            batch_word_probs[:, :-1] = (1 - contrastive_alpha) * batch_word_probs[:, :-1] - penalty.unsqueeze(1)

            if i < beam_forward_num:
                #將PROMPT 從PAST_KEY _VALUE切分
                batch_pw_past_key_values = _cache_slice(outputs.past_key_values, pw_silce_index[:l + 1])
                word_probs = torch.cat([word_probs, batch_word_probs[:, :-1]], dim=0)
                pw_past_key_values = _merge_cache(pw_past_key_values, batch_pw_past_key_values)
                # 收集 beam_width 內所有current hidden states (未篩選)
                hidden_states_batch.append(current_hidden)

            del outputs

            batch_eos_over_threshold_index = torch.where(batch_word_probs[:, -1] >= eos_threshold)[0]
            if batch_eos_over_threshold_index.shape[0] != 0:
                eos_seqs = torch.cat([
                    input_seqs[batch_eos_over_threshold_index, :],
                    vocab_tensor[-1].repeat(batch_eos_over_threshold_index.shape[0], 1)
                ], dim=1)

                batch_eos_probs = (
                    beam.search_prob[start:end, :] + batch_word_probs[:, -1:]
                )[batch_eos_over_threshold_index]

                eos_list.extend(zip(eos_seqs,batch_eos_probs))

        next_l = l + 1
        if next_l < max_length:
            next_beam_width = beam_width_list[next_l]
            next_reserve_width = max(beam_width_list[next_l], search_width_list[next_l])
            if len(hidden_states_batch) > 0:
                all_hidden = torch.cat(hidden_states_batch, dim=0)  # [beam_width, hidden_dim]
                beam.update_by_prob(next_beam_width, next_reserve_width, word_probs, pw_past_key_values, all_hidden)
            else:
                beam.update_by_prob(next_beam_width, next_reserve_width, word_probs, pw_past_key_values)

    if sorted:
        eos_list.sort(key=lambda x: x[1], reverse=True)

    return eos_list