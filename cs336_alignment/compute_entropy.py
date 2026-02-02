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

    # 前向传播获取logits，不计算梯度（评分任务无需梯度，节省显存）
    with torch.no_grad():
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