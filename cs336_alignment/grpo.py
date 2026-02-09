from transformers import AutoModelForCausalLM, AutoTokenizer,PreTrainedModel,get_linear_schedule_with_warmup
import torch
from typing import Dict, List, Tuple,Literal
import json
from torch.utils.data import DataLoader
from cs336_alignment.grpo_utils import compute_policy_gradient_loss,compute_group_normalized_rewards,masked_mean
import random
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import LLM,SamplingParams
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.sft_utils import get_response_log_probs
import gc
def release_gpu_memory():
    # 清除所有未使用的临时张量
    torch.cuda.empty_cache()
    # 强制垃圾回收，释放Python层面的无用对象
    gc.collect()
    # 清除模型的计算图（避免残留）
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.device.type == 'cuda':
                del obj
        except:
            pass
    # 再次清空缓存，确保碎片被回收
    torch.cuda.empty_cache()

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    """
    # 计算GRPO微批次的损失
    token_loss,stats=compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange
    )
    loss=masked_mean(token_loss,response_mask,dim=None)
    adjusted_loss=loss/gradient_accumulation_steps
    # 反向传播
    adjusted_loss.backward()
    # 封装元数据
    metadata = {
        "adjusted_microbatch_loss": adjusted_loss.item(),  # 梯度累积调整后的损失
        **stats,
    }
    return adjusted_loss,metadata

def grpo(data_path="../data/MATH/train.jsonl",model_name="Qwen/Qwen2.5-Math-1.5B",save_path="../output/model/grpo_qwen_math_1.5b"):
    reward_fn=r1_zero_reward_fn
    prompt_fpath="./prompts/r1_zero.prompt"
    device='cuda'
    n_grpo_steps: int = 3
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 64
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1 # On-policy
    train_batch_size: int = 32 # On-policy
    gradient_accumulation_steps: int = 16 # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.85
    cliprange=0.2
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
        ] = "reinforce_with_baseline"
    use_std_normalization: bool = True

    assert train_batch_size % gradient_accumulation_steps == 0, "train_batch_size must be divisible by gradient_accumulation_steps"
    train_microbatch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, "train_batch_size must be greater than or equal to group_size"
    on_policy = epochs_per_rollout_batch == 1 and train_batch_size == rollout_batch_size
    assert not (loss_type == "grpo_clip" and on_policy)



    seed=42


    #构建dataset
    dataset_train=[]
    cnt=0
    with open(data_path, "r") as file:
        for line in file:
            dataset_train.append(json.loads(line))
            cnt+=1
            if cnt>=100:   #for test
                break
    # 加载prompts模板
    with open(prompt_fpath, "r") as file:
        prompt_template = file.read()

    for step in range(n_grpo_steps):
        if step==0:
            model_name="Qwen/Qwen2.5-Math-1.5B"
        else:
            model_name="../output/model/grpo_qwen_math_1.5b"
        #处理rollout数据
        rollout_dataset = random.sample(dataset_train, n_prompts_per_rollout_batch)
        rollout_questions = [data["problem"] for data in rollout_dataset]
        rollout_prompts = [prompt_template.format(question=question) for question in rollout_questions]
        rollout_answers = [data["answer"] for data in rollout_dataset]
        #调用vllm进行推理
        #初始化vllm
        vllm_model=LLM(model=model_name,seed=seed)
        sampling_params = SamplingParams(
            temperature=sampling_temperature, 
            top_p=1.0, 
            min_tokens=sampling_min_tokens, 
            max_tokens=sampling_max_tokens, 
            stop=["</answer>"], 
            include_stop_str_in_output=True, 
            n=group_size, 
            seed=seed)
        outputs = vllm_model.generate(rollout_prompts, sampling_params)
        #处理模型输出
        repeated_answers = []
        generations = []
        prompts = []
        for output, answer in zip(outputs, rollout_answers):
            prompt = output.prompt
            for i in range(group_size):
                generated_text = output.outputs[i].text
                prompts.append(prompt)
                generations.append(generated_text)
                repeated_answers.append(answer)

        del vllm_model
        release_gpu_memory()

        #加载模型和分词器
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map='cuda'  # 强制模型在GPU上初始化，FA2组件同步构建在GPU
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        #构建优化器
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )
        tokenizations = tokenize_prompt_and_output(prompts, generations, tokenizer)
        input_ids, labels, response_mask = tokenizations["input_ids"], tokenizations["labels"], tokenizations["response_mask"]
        tokenizations_train = tokenizations #why?
        advantages_train, raw_rewards_train, metadata = compute_group_normalized_rewards(
                                                            reward_fn=reward_fn, 
                                                            rollout_responses=generations, 
                                                            repeated_ground_truths=repeated_answers, 
                                                            group_size=group_size, 
                                                            advantage_eps=advantage_eps, 
                                                            normalize_by_std=use_std_normalization,
                                                            device="cuda"
                                                            )
        # Print a few example rollouts
        print("\n=== Example Rollouts ===")
        for i in range(min(3, len(rollout_prompts))):
            print(f"Prompt: {rollout_prompts[i]!r}")
            print(f"Generated Output: {generations[i]!r}")
            print(f"Reward: {raw_rewards_train[i].item()}")
            print()
        print("=======================\n")

        #compute the old_log_probs over the entire dataset (split into microbatches), if off policy
        num_train_steps_per_epoch = rollout_batch_size // train_batch_size
        with torch.no_grad():
            old_log_probs_train=[]
            for train_step in range(num_train_steps_per_epoch):
                batch_idxs = train_step*train_batch_size, train_step*train_batch_size+train_batch_size
                for train_microstep in range(gradient_accumulation_steps):
                    microbatch_idxs = batch_idxs[0] + train_microstep*train_microbatch_size, batch_idxs[0] + train_microstep*train_microbatch_size+train_microbatch_size
                    # print(f"train_step = {train_step}, train_microstep = {train_microstep}: microbatch_idxs = {microbatch_idxs}")
                    # print(f"train_step = {train_step}, train_microstep = {train_microstep}: tokenizations.shape = {tokenizations}")
                    input_ids, labels, response_mask = tokenizations["input_ids"], tokenizations["labels"], tokenizations["response_mask"]
                    input_ids, labels, response_mask = input_ids[microbatch_idxs[0]:microbatch_idxs[1]], labels[microbatch_idxs[0]:microbatch_idxs[1]], response_mask[microbatch_idxs[0]:microbatch_idxs[1]]
                    log_probs_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
                    log_probs, token_entropy = log_probs_dict["log_probs"], log_probs_dict["token_entropy"]
                    old_log_probs_train.append(log_probs)
                    assert log_probs.shape[0] == microbatch_idxs[1]-microbatch_idxs[0]
            old_log_probs_train = torch.cat(old_log_probs_train)
        print(f"Grpo step {step}: Completed computing log probs on the old model, old_log_probs_train.shape = {old_log_probs_train.shape}")
        #train for epochs_per_rollout_batch on the same exact data
        for train_epoch in range(epochs_per_rollout_batch):
            for train_step in range(num_train_steps_per_epoch):
                batch_idxs = train_step*train_batch_size, train_step*train_batch_size+train_batch_size
                for train_microstep in range(gradient_accumulation_steps):
                    microbatch_idxs = batch_idxs[0] + train_microstep*train_microbatch_size, batch_idxs[0] + train_microstep*train_microbatch_size+train_microbatch_size
                    advantages = advantages_train[microbatch_idxs[0]:microbatch_idxs[1]].unsqueeze(-1)
                    raw_rewards = raw_rewards_train[microbatch_idxs[0]:microbatch_idxs[1]].unsqueeze(-1)
                    old_log_probs = old_log_probs_train[microbatch_idxs[0]:microbatch_idxs[1]]
                    input_ids, labels, response_mask = tokenizations_train["input_ids"], tokenizations_train["labels"], tokenizations_train["response_mask"]
                    input_ids, labels, response_mask = input_ids[microbatch_idxs[0]:microbatch_idxs[1]], labels[microbatch_idxs[0]:microbatch_idxs[1]], response_mask[microbatch_idxs[0]:microbatch_idxs[1]]

                    #compute new log_probs
                    log_probs_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
                    log_probs, token_entropy = log_probs_dict["log_probs"], log_probs_dict["token_entropy"]
                    policy_log_probs = log_probs

                    loss, metadata = grpo_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
                    print(f"TRAIN: Grpo step {step}, train epoch {train_epoch}, train_step {train_step}, microbatch_train_step: {train_microstep}: train loss = {loss}")
                    avg_token_entropy = masked_mean(token_entropy, response_mask, dim=None)
                    if loss_type == "grpo_clip":
                        clipped_fraction = masked_mean(metadata["clipped"], response_mask, dim=None)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()
                optimizer.zero_grad()
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

if __name__=="__main__":
    grpo()