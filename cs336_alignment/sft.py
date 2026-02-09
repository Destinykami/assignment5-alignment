from transformers import AutoModelForCausalLM, AutoTokenizer,PreTrainedModel,get_linear_schedule_with_warmup
import torch
from typing import Dict, List, Tuple
import json
from cs336_alignment.sft_utils import get_response_log_probs,masked_normalize
from torch.utils.data import DataLoader
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
        Execute a forward-and-backward pass on a microbatch.
    Args:
        policy_log_probs :(batch_size, sequence_length), per-token log-probabilities from the SFT policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps :Number of microbatches per optimizer step.
        normalize_constant :The constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss :scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return this so we can log it.
        metadata : Dict with metadata from the underlying loss call, and any other statistics you might want to log
    """
    # 步骤 1：计算 SFT 微批次的掩码损失
    #掩码过滤  只保留response部分的对数概率
    masked_log_probs=policy_log_probs*response_mask.to(dtype=policy_log_probs.dtype)
    #求和 归一化 取负得到损失
    sum_valid_log_probs=torch.sum(masked_log_probs,dim=None) #全维度求和有效区
    raw_loss=-(sum_valid_log_probs/normalize_constant) #训练时要最小化损失，而目标是最大化有效区对数概率，取负后两者等价（对数概率越大，损失越小）
    
    # 步骤 2：调整损失，适配梯度累积
    adjusted_loss=raw_loss/gradient_accumulation_steps

    # 步骤 3：反向传播，累积梯度到模型参数
    adjusted_loss.backward()

    # 步骤 4：计算训练监控元数据
    # 统计有效Token数（response部分的Token数量）
    valid_token_num = torch.sum(response_mask).item()
    # 统计response部分的平均条件对数概率（收敛核心指标：越高越接近0，模型越确定）
    avg_log_prob = (sum_valid_log_probs / valid_token_num).item() if valid_token_num > 0 else 0.0

    # 封装元数据：包含原始损失、调整后损失、有效Token数、平均对数概率等
    metadata = {
        "raw_microbatch_loss": raw_loss.item(),  # 未调整的原始损失
        "adjusted_microbatch_loss": adjusted_loss.item(),  # 梯度累积调整后的损失
        "valid_response_token_num": valid_token_num,  # 有效训练Token数
        "avg_response_log_prob": avg_log_prob,  # response部分平均对数概率（收敛监控）
        "normalize_constant": normalize_constant  # 归一化常数（日志回溯）
    }

    return adjusted_loss,metadata

# ===== 补充2：轻量Dataset类（仅存储数据，分词逻辑抽离到collate_fn） =====
# 该Dataset仅做数据容器，不包含分词逻辑，配合collate_fn调用你的tokenize_prompt_and_output
class SimpleSFTDataset(torch.utils.data.Dataset):
    def __init__(self, prompt_list: List[str], output_list: List[str]):
        super().__init__()
        self.prompt_list = prompt_list
        self.output_list = output_list
        assert len(self.prompt_list) == len(self.output_list), "数据长度不匹配"

    def __len__(self) -> int:
        return len(self.prompt_list)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        # 仅返回单条prompt和output，分词由collate_fn统一处理
        return self.prompt_list[idx], self.output_list[idx]

# ===== 补充3：定义collate_fn（核心衔接：DataLoader批次 → 你的分词函数） =====
# DataLoader加载批次数据后，通过该函数调用你的tokenize_prompt_and_output，生成模型所需张量
def sft_collate_fn(
        batch: List[Tuple[str, str]],
        tokenizer: AutoTokenizer
    ) -> Dict[str, torch.Tensor]:
    # batch是[(prompt1, output1), (prompt2, output2), ...]，拆分出批次的prompt和output
    prompt_strs = [item[0] for item in batch]
    output_strs = [item[1] for item in batch]
    # 调用你已实现的分词函数，生成input_ids/labels/response_mask
    tokenized_dict = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    return tokenized_dict


# SFT
def sft(data_path="../data/MATH/sft.jsonl",model_name="Qwen/Qwen2.5-Math-1.5B",save_path="../output/model/sft_qwen_math_1.5b"):
    DEVICE='cuda'       # [CONFIG]
    BATCH_SIZE=2        #[CONFIG]
    LEARNING_RATE=5e-5  #[CONFIG]
    GRAD_ACCUM_STEPS = 4  # 显存不足增大至8/16
    EPOCHS = 3  # 可根据收敛情况调整
    SAVE_MODEL_PATH = save_path  # 模型保存路径
    MAX_TRAIN_SAMPLES = None  # 设为None使用全部数据，如需限制设具体数值（如500）
    #加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map='cuda'  # 强制模型在GPU上初始化，FA2组件同步构建在GPU
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #构建Dataset和DataLoader
    train_prompts=[]
    train_outputs=[]
    with open(data_path,'r') as f:
        for line in f:
            item=json.loads(line)
            train_prompts.append(item["prompt"])
            train_outputs.append(item["response"])
    dataset=SimpleSFTDataset(train_prompts,train_outputs)
    print(f"数据集读取完成！共加载{len(train_prompts)}条有效样本")
    dataloader=DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0, # todo
        collate_fn=lambda batch: sft_collate_fn(batch,tokenizer)
    )
    #初始化优化器和调度器
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4
    )
    total_training_steps = len(dataloader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(total_training_steps * 0.1),
        num_training_steps=total_training_steps
    )
    # 开始训练
    model.train()
    print(f"\n===== 开始SFT训练 =====")
    print(f"设备：{DEVICE} | 批次：{BATCH_SIZE} | 梯度累积：{GRAD_ACCUM_STEPS}")
    print(f"总步数：{total_training_steps} | 样本数：{len(train_prompts)} | 轮数：{EPOCHS}")

    # 新增：全局训练步数，匹配total_training_steps，监控真实训练进度
    global_step = 0

    for epoch in range(EPOCHS):
        epoch_total_loss = 0.0  # 初始化为浮点型，而非张量
        # 删去：epoch开头的model.zero_grad() → 梯度清空仅在优化器更新后执行
        print(f"\n--- 轮次 {epoch+1}/{EPOCHS} ---")
        
        for step, batch in enumerate(dataloader):
            global_step += 1
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            response_mask = batch["response_mask"].to(DEVICE)
            
            # 前向计算log_probs
            log_probs_dict=get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False  # 核心修正：无需熵计算，节省显存/计算
            )
            policy_log_probs=log_probs_dict["log_probs"]  # 形状(B,S)
            
            # 修正问题3 → 匹配模型bf16精度，避免混合精度数值问题
            policy_log_probs = policy_log_probs.to(torch.bfloat16)
            
            # 执行SFT微批次训练步骤
            microbatchloss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=GRAD_ACCUM_STEPS,
                normalize_constant=1.0
            )
            
            # 修正问题4 → 提取张量数值累加，避免GPU显存冗余
            epoch_total_loss += microbatchloss.item()
            
            # 梯度累积结束，执行优化器步骤（原有判断条件正确，保留）
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()  # 正确：优化器更新后清空梯度，为下一轮累积做准备
                
                # 优化：日志添加全局步数，匹配total_training_steps，直观监控进度
                print(f"全局步数 {global_step}/{total_training_steps} | 批次步数 {step + 1} | "
                    f"调整后损失: {metadata['adjusted_microbatch_loss']:.4f} | "
                    f"平均response对数概率: {metadata['avg_response_log_prob']:.4f}")
        
        # 轮次结束打印统计（原有逻辑正确，保留）
        avg_epoch_loss = epoch_total_loss / len(dataloader)
        print(f"===== E{epoch+1}/{EPOCHS} 训练完成 =====")
        print(f"轮次平均微批次损失：{avg_epoch_loss:.4f}\n")

    # ===== 8. 保存模型（无需修改） =====
    model.save_pretrained(SAVE_MODEL_PATH, safe_serialization=True)
    tokenizer.save_pretrained(SAVE_MODEL_PATH)
    print(f"微调完成！模型已保存至：{SAVE_MODEL_PATH}")

if __name__=="__main__":
    sft()