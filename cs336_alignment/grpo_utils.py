import torch
from typing import Callable, List, Literal

def compute_group_normalized_rewards(
    reward_fn:Callable[[str, str], dict[str, float]],
    rollout_responses:List[str],
    repeated_ground_truths:List[str] ,
    group_size:int,
    advantage_eps:float,
    normalize_by_std:bool,
    )->tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    按组计算归一化奖励和优势值
    """
    #原始奖励
    raw_rewards=[]
    for response,gt in zip(rollout_responses,repeated_ground_truths):
        reward_dict=reward_fn(response,gt)
        raw_reward=reward_dict["reward"]
        raw_rewards.append(raw_reward)
    raw_rewards_tensor=torch.tensor(raw_rewards,dtype=torch.float32)
    #按组重塑
    num_groups = len(raw_rewards) // group_size
    rewards_grouped = raw_rewards_tensor.reshape(num_groups, group_size)
    #均值 标准差
    group_means=rewards_grouped.mean(dim=1,keepdim=True)
    group_stds=rewards_grouped.std(dim=1,keepdim=True)
    #组内归一化奖励
    if normalize_by_std:
        normalized_rewards_grouped=(rewards_grouped-group_means)/(group_stds+advantage_eps)
    else:
        normalized_rewards_grouped=rewards_grouped-group_means
    #展平为原始形状
    normalized_rewards=normalized_rewards_grouped.reshape(-1)
    #计算优势,这里组内均值已减为0
    advantages=normalized_rewards.clone()
    reward_stats = {
        "raw_reward_mean": float(raw_rewards_tensor.mean().item()),
        "raw_reward_std": float(raw_rewards_tensor.std().item()),
        "normalized_reward_mean": float(normalized_rewards.mean().item()),
        "normalized_reward_std": float(normalized_rewards.std().item()),
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std().item())
    }
    return advantages,raw_rewards_tensor,reward_stats

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    ) -> torch.Tensor:
    """
    计算每个token的损失
    """
    per_token_pg_loss=-raw_rewards_or_advantages*policy_log_probs
    return per_token_pg_loss

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算每个token的grpo clip的损失
    """
    ratio = torch.exp(policy_log_probs - old_log_probs) #输入的概率是对数概率,log_prob相减 → exp后是概率相除
    # 计算裁剪后的ratio
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    
    # 计算两个替代损失（未裁剪/裁剪）
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    
    per_token_grpo_loss = -torch.min(surrogate1, surrogate2)
    
    # 计算统计信息（逐token+聚合指标）
    # 逐token标记是否被裁剪（ratio超出[1-ε,1+ε]范围）
    is_clipped = torch.logical_or(ratio < (1 - cliprange), ratio > (1 + cliprange))
    stats = {
        # 逐token统计
        "is_clipped": is_clipped,  # (batch_size, seq_len) 布尔张量
        "ratio": ratio,            # (batch_size, seq_len) 原始ratio
        "clipped_ratio": clipped_ratio,  # 裁剪后的ratio
        # 聚合统计（标量，方便训练监控）
        "clip_fraction": is_clipped.float().mean(),  # 被裁剪的token比例
        "avg_ppo_loss": per_token_grpo_loss.mean(),   # 平均逐token损失
        "avg_ratio": ratio.mean(),                   # 平均ratio
    }
    
    return per_token_grpo_loss, stats

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.
    """
    if loss_type=="no_baseline":
        loss=compute_naive_policy_gradient_loss(raw_rewards_or_advantages=raw_rewards,policy_log_probs=policy_log_probs)
        metadata = {
            "loss_type": "no_baseline",
            "avg_pg_loss": loss.mean(),
            "raw_rewards_mean": raw_rewards.mean(),
        }
    elif loss_type=="grpo_clip":
        loss,stats= compute_grpo_clip_loss(
            advantages=advantages,policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,cliprange=cliprange
        )
        metadata = {
            "loss_type": "grpo_clip", 
            "cliprange": torch.tensor(cliprange),
            **stats,  # 合并grpo clip的统计信息
        }
    else:
        #reinforce_with_baseline
        loss=compute_naive_policy_gradient_loss(raw_rewards_or_advantages=advantages,policy_log_probs=policy_log_probs)
        metadata = {
            "loss_type": "reinforce_with_baseline",
            "avg_pg_loss": loss.mean(),
            "advantages_mean": advantages.mean(),
        }
    return loss,metadata

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    ) -> torch.Tensor:
    """
    To reduce our per-token loss tensors of shape (batch_size, sequence_length) to a vector of losses (one scalar for each example),
    """
    # 计算有效元素的和（仅mask=1的元素）
    masked_sum = (tensor * mask).sum(dim=dim, keepdim=False)
    # 计算有效元素的数量
    mask_count = mask.sum(dim=dim, keepdim=False)
    masked_mean_val = masked_sum / mask_count
    return masked_mean_val
    