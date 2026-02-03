import pickle
# 加载pkl文件，直接得到四组核心数据
prompts, answers, generations, evaluations = pickle.load(open("../output/Qwen2.5_Math_1.5B_math_sft_results.pkl", "rb"))
# 统计各种情况的分数
correct_count=0
for eval_dict in evaluations:
    if eval_dict['format_reward']==1.0 and eval_dict["answer_reward"]==1.0:
        correct_count+=1
print("format_reward=1 and answer_reward=1:",correct_count)

part_count=0
for eval_dict in evaluations:
    if eval_dict['format_reward']==1.0 and eval_dict["answer_reward"]==0.0:
        part_count+=1
print("format_reward=1 and answer_reward=0:",part_count)

invalid_count=0
for eval_dict in evaluations:
    if eval_dict['format_reward']==0.0 and eval_dict["answer_reward"]==0.0:
        invalid_count+=1
print("format_reward=0 and answer_reward=0:",invalid_count)
# 输出部分生成结果进行检查
# for i in range(5):
#     print(f"Prompt: {prompts[i]}")
#     print(f"Generated Answer: {generations[i]}")
#     print(f"Reference Answer: {answers[i]}")
#     print(f"Evaluation: {evaluations[i]}")
#     print("-" * 50)