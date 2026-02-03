import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算逐Token的下一个Token预测熵（基于词汇表维度），采用数值稳定方法避免溢出。
    Args:
        logits: torch.Tensor，形状为 (batch_size, sequence_length, vocab_size)，
                包含未归一化的模型输出logits。
    Returns:
        torch.Tensor，形状为 (batch_size, sequence_length)，每个位置的下一个Token预测熵。
    """
    # 步骤1：计算词汇表维度的logsumexp，得到log(Σexp(logits))，数值稳定避免exp溢出
    # dim=-1：沿词汇表维度计算，keepdim=False：输出形状(batch_size, sequence_length)
    log_Z = torch.logsumexp(logits, dim=-1)  # shape: (B, S)
    
    # 步骤2：计算数值稳定的对数概率 log(p_i) = logits_i - log_Z
    # unsqueeze(-1)恢复词汇表维度，实现广播计算，shape保持(B, S, V)
    log_probs = logits - log_Z.unsqueeze(dim=-1)  # shape: (B, S, V)
    
    # 步骤3：计算熵 H = -Σ(p_i * log(p_i))，p_i = exp(log_probs)
    # 沿词汇表维度求和后取负，最终形状(B, S)
    entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)  # shape: (B, S)
    
    return entropy
    
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    计算自回归模型的条件对数概率 log pθ(xt | x<t)，可选返回逐Token熵。
    适配Hugging Face PreTrainedModel，自动处理labels中-100的掩码标记，采用数值稳定计算避免溢出。

    Args:
        model: PreTrainedModel - Hugging Face预训练模型（已放置在正确设备，推理模式由用户保证）。
        input_ids: torch.Tensor - 形状(batch_size, sequence_length)，拼接后的prompt+response的token id序列。
        labels: torch.Tensor - 形状(batch_size, sequence_length)，与tokenization生成的labels一致，-100为掩码忽略位。
        return_token_entropy: bool - 若为True，额外返回每个位置的逐Token熵（调用compute_entropy）。

    Returns:
        dict[str, torch.Tensor]:
            - log_probs: 形状(batch_size, sequence_length)，每个位置的条件对数概率log pθ(xt | x<t)，掩码位(-100)置0。
            - token_entropy: 可选键，形状(batch_size, sequence_length)，逐Token熵，仅当return_token_entropy=True时存在。
    """
    # 确保输入张量在模型同一设备
    input_ids = input_ids.to(model.device)
    labels = labels.to(model.device)
    batch_size, seq_len = input_ids.shape

    model_outputs = model(input_ids=input_ids)
    logits = model_outputs.logits  # 形状: (batch_size, sequence_length, vocab_size)

    # 步骤1：数值稳定计算所有词汇的对数概率 log(p_i) = logits_i - logsumexp(logits, dim=-1)
    log_Z = torch.logsumexp(logits, dim=-1)  # 归一化常数，形状: (B, S)
    vocab_log_probs = logits - log_Z.unsqueeze(dim=-1)  # 所有vocab的对数概率，形状: (B, S, V)

    # 步骤2：提取labels对应位置的条件对数概率（核心：gather按索引取值）
    # labels.unsqueeze(-1)扩展为(B, S, 1)，gather在词汇表维度取对应token的对数概率
    token_log_probs = vocab_log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1)  # 索引与labels一致，取对应vocab的log prob
    ).squeeze(-1)  # 挤压最后一维，恢复为(B, S)，与labels形状一致

    # 步骤3：处理labels中的-100掩码位（将忽略位的对数概率置0，不影响后续求和/统计）
    mask = labels == -100  # 形状(B, S)，掩码位为True
    token_log_probs[mask] = 0.0

    # 构建返回字典
    result = {"log_probs": token_log_probs}

    # 步骤4：若需要，计算并添加逐Token熵
    if return_token_entropy:
        token_entropy = compute_entropy(logits)  # 复用数值稳定版计算
        result["token_entropy"] = token_entropy

    return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    带布尔掩码的张量求和与常数归一化：仅mask==1的元素参与求和，求和后除以归一化常数。
    支持指定维度求和或全维度求和，保证mask无效元素不贡献任何计算。

    Args:
        tensor: torch.Tensor - 待求和、归一化的目标张量，任意维度。
        mask: torch.Tensor - 掩码张量，与tensor形状完全一致；mask==1/True为有效元素，参与求和。
        normalize_constant: float - 归一化常数，求和结果将除以该值。
        dim: int | None - 求和维度；None表示对所有维度求和，支持正/负索引。

    Returns:
        torch.Tensor - 归一化后的求和结果，形状规则：
                       - dim=None → 返回标量（0维张量）；
                       - 指定dim → 返回与原张量维度数相同的张量，求和维度被压缩为1。
    """
    # 关键校验：mask与tensor形状必须一致，避免维度不匹配
    if tensor.shape != mask.shape:
        raise ValueError(f"Tensor shape ({tensor.shape}) must match mask shape ({mask.shape})")
    
    # 掩码过滤——将mask=0的位置置0，仅保留有效元素
    # 转换为同类型张量，避免数据类型误差（如float16张量与uint8掩码相乘）
    masked_tensor = tensor * mask.to(dtype=tensor.dtype)
    
    if dim is not None:
        # 指定维度求和：keepdim=False，直接删除该维度，无冗余
        sum_result = torch.sum(masked_tensor, dim=dim, keepdim=False)
    else:
        # 全维度求和：先保留维度，再挤压为标量
        sum_result = torch.sum(masked_tensor, dim=dim, keepdim=True)
        sum_result = sum_result.squeeze()
    
    # 步骤3：常数归一化——除以归一化常数，完成最终计算
    normalized_result = sum_result / normalize_constant
    
    return normalized_result