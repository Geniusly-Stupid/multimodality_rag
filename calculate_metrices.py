import json

path = r"D:\Desktop\SI650\project\multimodality_rag\evaluation_results\frames_rag_results.jsonl"

num_total = 0
num_f1_not_one = 0

num_retrieval_hit = 0   # labels 中至少一个 1
top_k = 5               # 固定计算前 5 条
total_correct_retrievals = 0   # 所有问题中命中的总条数

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        num_total += 1

        # ---- 1. 统计 F1 != 0 ----
        f1 = record["metrics"]["f1"]
        if f1 != 0:
            num_f1_not_one += 1

        # ---- 2. 计算 top-5 hitrate ----
        labels = record.get("labels", [])
        labels_top5 = labels[:top_k]

        if any(label == 1 for label in labels_top5):
            num_retrieval_hit += 1

        # 统计这个问题 top-k 中有多少条命中
        correct_for_this_question = sum(1 for x in labels_top5 if x == 1)
        total_correct_retrievals += correct_for_this_question


# ===== 输出结果 =====
print(f"Total samples: {num_total}")
print(f"F1 != 0 count: {num_f1_not_one}")
print(f"F1 != 0 ratio: {num_f1_not_one / num_total:.4f}")

print(f"\nTop-{top_k} Retrieval Hit-Rate: {num_retrieval_hit / num_total:.4f}")
print(f"Hit count: {num_retrieval_hit}")


print(f"Average correct retrievals per question (top-{top_k}): "
      f"{total_correct_retrievals / num_total:.4f}")
print(f"Total correct retrievals: {total_correct_retrievals}")
