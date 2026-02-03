from typing import Callable, List
from vllm import LLM, SamplingParams
import pickle
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import json
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
    generations = []
    evaluations = []
    for output, answer in zip(results, answers):
        generated_text = output.outputs[0].text
        rewards = reward_fn(generated_text, answer)
        generations.append(generated_text)
        evaluations.append(rewards)

    with open(save_fpath, 'wb') as file:
        pickle.dump((prompts, answers, generations, evaluations), file)

if __name__=="__main__":
    reward_fn=r1_zero_reward_fn
    save_fpath="../output/Qwen2.5_Math_1.5B_math_sft_results.pkl"
    sft_model_path = "/root/autodl-tmp/assignment5-alignment/output/model/sft_qwen_math_1.5b"  #微调后的模型
    # 读取数据集
    math_fpath="../data/MATH/validation.jsonl"
    dataset=[]
    with open(math_fpath, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    questions=[item["problem"] for item in dataset]
    answers=[item["answer"] for item in dataset]
    #读取prompt模板
    prompts_fpath="./prompts/r1_zero.prompt"
    with open(prompts_fpath, 'r') as f:
        prompt_template=f.read()
    prompts=[prompt_template.format(question=q) for q in questions]
    # 设置评估参数
    eval_sampling_params=SamplingParams(
        temperature=1.0,top_p=1.0, 
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output = True
        )
    # 初始化模型
    llm = LLM(
        model=sft_model_path,  # 加载本地SFT微调模型（核心修改）
        dtype="bfloat16",  # 匹配SFT训练的bf16精度，避免精度冲突
        device="cuda",  # 强制使用GPU推理，vllm默认GPU但显式指定更稳妥
        tensor_parallel_size=1  # 单卡推理，AutoDL单卡环境固定为1
    )
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_sampling_params=eval_sampling_params,
        save_fpath=save_fpath,
        answers=answers,
    )