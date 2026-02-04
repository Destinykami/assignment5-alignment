from typing import Callable, List
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import json
from cs336_alignment.sft import sft
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
        if rewards['reward']==0.0:
            continue  # 只保存reward不为0的结果
        generations.append(generated_text)
        evaluations.append(rewards)
    # 保存到jsonl文件
    with open(save_fpath, 'wb') as f:
        f.write(json.dumps({
            "prompts": prompts,
            "answers": answers,
            "generations": generations,
            "evaluations": evaluations
        }).encode('utf-8'))

if __name__=="__main__":
    reward_fn=r1_zero_reward_fn
    save_fpath="../ei_datasets/ei_sftdataset.jsonl"
    # 用train数据集生成sft数据集
    math_fpath="../data/MATH/train.jsonl"
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
    samping_min_tokens=4
    samping_max_tokens=1024
    seed=42
    G=4  # 生成数量
    eval_sampling_params=SamplingParams(
        temperature=1.0,top_p=1.0, 
        max_tokens=samping_max_tokens,
        min_tokens=samping_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output = True,
        seed=seed,
        n=G
        )

    n_ei_steps=1 # 专家迭代次数
    for step in range(n_ei_steps):
        print(f"专家迭代第 {step} 步")
        if step==0:
            model="Qwen/Qwen2.5-Math-1.5B"
        else:
            model="../output/model/ei_sft_qwen_math_1.5b"
        # 加载当前模型
        vllm_model=LLM(model=model)
        # 评估当前模型
        evaluate_vllm(
            vllm_model=vllm_model,
            reward_fn=reward_fn,
            prompts=prompts,
            eval_sampling_params=eval_sampling_params,
            save_fpath=save_fpath,
            answers=answers,
        )
        # SFT微调
        #sft()