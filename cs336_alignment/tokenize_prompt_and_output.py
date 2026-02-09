import torch
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer,PreTrainedModel,get_linear_schedule_with_warmup

def tokenize_prompt_and_output(prompt_strs: List[str],
                               output_strs: List[str],
                               tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings separately, concatenate them, 
    and construct input_ids, shifted labels and response_mask for model training.
    
    Args:
        prompt_strs: list[str] List of prompt strings (one per sample).
        output_strs: list[str] List of output strings (one per sample, aligned with prompts).
        tokenizer: PreTrainedTokenizer Hugging Face tokenizer for tokenization.

    Returns:
        dict[str, torch.Tensor] Containing input_ids, labels, response_mask (all shape: [batch_size, max_len-1])
    """
    assert len(prompt_strs) == len(output_strs), \
        f"Prompt and output list length mismatch: {len(prompt_strs)} vs {len(output_strs)}"
    batch_size = len(prompt_strs)
    if batch_size == 0:
        raise ValueError("Empty prompt/output lists are not allowed")

    # 兼容处理：设置pad_token（部分tokenizer如GPT2默认无pad_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    # 步骤1：单独分词，记录【prompt真实长度、output真实长度、完整拼接序列、总长度】
    prompt_lens = []  # 每个样本prompt的原始token长度（未padding）
    output_lens = []  # 每个样本output的原始token长度（未padding，新增核心）
    full_token_ids = []  # 每个样本prompt+output的原始拼接token id
    prompt_and_output_lens = []  # 每个样本拼接后的原始总token长度
    for prompt, output in zip(prompt_strs, output_strs):
        # 单独分词，不添加特殊token，保证拼接边界清晰
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        output_tokens = tokenizer.encode(output, add_special_tokens=False)
        full_tokens = prompt_tokens + output_tokens
        # 记录关键信息：新增output_lens存储原始output长度
        prompt_lens.append(len(prompt_tokens))
        output_lens.append(len(output_tokens))
        full_token_ids.append(full_tokens)
        prompt_and_output_lens.append(len(full_tokens))

    # 步骤2：计算批次最大总长度，确定统一padding目标
    max_total_len = max(prompt_and_output_lens)
    target_seq_len = max_total_len - 1  # 最终所有张量的第二维度

    # 步骤3：先对所有完整序列统一padding到max_total_len（等长），再切分
    padded_full_tokens = []
    for tokens in full_token_ids:
        pad_len = max_total_len - len(tokens)
        padded_tokens = tokens + [pad_token_id] * pad_len  # 右侧padding，匹配快照规则
        padded_full_tokens.append(padded_tokens)

    # 步骤4：基于等长完整序列，切分生成input_ids和labels（无需再padding）
    batch_input_ids = [tokens[:-1] for tokens in padded_full_tokens]
    batch_labels = [tokens[1:] for tokens in padded_full_tokens]

    # 步骤5：构建response_mask（核心修正：按原始output长度限制1的区间）
    batch_response_mask = []
    for i in range(batch_size):
        pl = prompt_lens[i]  # 当前样本prompt原始长度
        ol = output_lens[i]  # 当前样本output原始长度
        mask = [False] * target_seq_len  # 先初始化全False的等长mask
        
        # 核心修正1：正确的output起始位置 = pl - 1
        output_start = pl - 1
        # 核心修正2：计算正确的output结束位置，增加溢出保护（不超过target_seq_len）
        output_end = output_start + ol
        output_end = min(output_end, target_seq_len)  # 防止超出序列长度
        
        # 仅在有效区间内标True，其余保持False（自动覆盖padding区）
        if output_start < output_end:  # 避免start >= end的无效情况（如pl=0/ol=0）
            mask[output_start:output_end] = [True] * (output_end - output_start)
        batch_response_mask.append(mask)

    # 步骤6：转换为torch张量，设置标准数据类型
    input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long)
    labels_tensor = torch.tensor(batch_labels, dtype=torch.long)
    response_mask_tensor = torch.tensor(batch_response_mask, dtype=torch.bool)

    # 校验张量形状，确保符合需求
    assert all(tensor.shape == (batch_size, target_seq_len) for tensor in 
               [input_ids_tensor, labels_tensor, response_mask_tensor]), \
        f"Tensor shape mismatch: expected ({batch_size}, {target_seq_len})"

    # 返回结果字典
    return {
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "response_mask": response_mask_tensor
    }
