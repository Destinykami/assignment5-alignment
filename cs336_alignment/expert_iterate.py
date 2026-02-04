from typing import Callable, List
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import json
from cs336_alignment.sft import sft
import torch
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


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    save_fpath: str,
    answers: List[str],
    ) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    results = vllm_model.generate(prompts, sampling_params=eval_sampling_params)
    valid_prompts = []
    valid_answers = []
    valid_generations = []
    valid_evaluations = []

    for output, prompt, answer in zip(results, prompts, answers):  # 新增：遍历prompt/answer
        generated_text = output.outputs[0].text
        rewards = reward_fn(generated_text, answer)
        if rewards['reward'] == 0.0:
            continue  # 过滤无效样本，不加入任何列表
        # 修正2：四个列表同步添加，保证长度完全一致，样本无错位
        valid_prompts.append(prompt)
        valid_answers.append(answer)
        valid_generations.append(generated_text)
        valid_evaluations.append(rewards)

    with open(save_fpath, 'w', encoding='utf-8') as f:
            # 遍历四条等长列表，为每条样本构建独立JSON对象（核心）
            for p, a, g, e in zip(valid_prompts, valid_answers, valid_generations, valid_evaluations):
                # 单样本JSON字典（字段与原数据对应，结构清晰）
                sample_dict = {
                    "prompt": p,
                    "response": g,
                    "ground_truth": a,
                }
                # 序列化+逐行写入，ensure_ascii=False保留中文，加\n保证行分隔
                f.write(json.dumps(sample_dict, ensure_ascii=False) + '\n')

if __name__=="__main__":
    reward_fn=r1_zero_reward_fn
    save_fpath="../ei_datasets/ei_sftdataset.jsonl"
    # 用train数据集生成sft数据集
    math_fpath="../data/MATH/train.jsonl"
    dataset=[]
    with open(math_fpath, 'r',encoding='utf-8') as f:
        cnt=0
        for line in f:
            dataset.append(json.loads(line))
            cnt+=1
            if cnt>=1000: #for test
                break
    questions=[item["problem"] for item in dataset]
    answers=[item["answer"] for item in dataset]
    #读取prompt模板
    prompts_fpath="./prompts/r1_zero.prompt"
    with open(prompts_fpath, 'r') as f:
        prompt_template=f.read()
    prompts=[prompt_template.format(question=q) for q in questions]
    # 设置评估参数
    samping_min_tokens=4
    samping_max_tokens=1024
    seed=42
    G=1  # 生成数量
    eval_sampling_params=SamplingParams(
        temperature=1.0,top_p=1.0, 
        max_tokens=samping_max_tokens,
        min_tokens=samping_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output = True,
        seed=seed,
        n=G
        )

    n_ei_steps=4 # 专家迭代次数
    for step in range(n_ei_steps):
        print(f"专家迭代第 {step} 步")
        if step==0:
            model="Qwen/Qwen2.5-Math-1.5B"
        else:
            model="../output/model/ei_sft_qwen_math_1.5b"
        # 加载当前模型
        vllm_model=LLM(model=model)
        # 用当前专家模型生成数据集
        evaluate_vllm(
            vllm_model=vllm_model,
            reward_fn=reward_fn,
            prompts=prompts,
            eval_sampling_params=eval_sampling_params,
            save_fpath=save_fpath,
            answers=answers,
        )
        del vllm_model
        release_gpu_memory()

        # SFT微调
        sft(data_path=save_fpath,
            model_name=model,
            save_path="../output/model/ei_sft_qwen_math_1.5b",
            )