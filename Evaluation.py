import AIresults
import numpy as np
from scipy.optimize import linear_sum_assignment


#AESOP
import numpy as np
from scipy.optimize import linear_sum_assignment

def normalize_value(val):
    """简单归一化处理，比如去除前后空格"""
    if isinstance(val, str):
        return val.strip()
    return val

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets with normalization."""
    set1, set2 = set(map(normalize_value, set1)), set(map(normalize_value, set2))
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def match_start_time(pred_time, true_time):
    """Match start time, allowing ±1 year difference after normalization."""
    pred_time, true_time = normalize_value(pred_time), normalize_value(true_time)
    try:
        pred_year = int(''.join(filter(str.isdigit, pred_time)))  # Extract numeric part
        true_year = int(''.join(filter(str.isdigit, true_time)))
        return 1.0 if abs(pred_year - true_year) <= 1 else 0.0
    except ValueError:
        return 0.0

def calculate_aesop_optimized(predicted_data, true_data):
    """Compute AESOP score using maximum weight matching (Hungarian Algorithm) with normalized values."""
    object_weight = 7
    qualifier_weight = 3
    total_weight = object_weight + qualifier_weight

    # Step 1: Build the matching score matrix
    num_predict = len(predicted_data)
    num_true = len(true_data)
    cost_matrix = np.zeros((num_predict, num_true))

    for i, pred_entry in enumerate(predicted_data):
        pred_predicates = {k: v for k, v in pred_entry.items() if k != "name"}

        for j, true_entry in enumerate(true_data):
            if normalize_value(pred_entry["name"]) != normalize_value(true_entry["name"]):
                cost_matrix[i, j] = 0  # Set score to 0 for mismatched names
                continue

            true_predicates = {k: v for k, v in true_entry.items() if k != "name"}
            predicate_scores = []

            for predicate, pred_value in pred_predicates.items():
                true_value = true_predicates.get(predicate)

                if true_value is None:
                    continue  # Skip if predicate is missing in ground truth

                # Normalize object values
                pred_object = normalize_value(pred_value["value"])
                true_object = normalize_value(true_value["value"])

                # Step 1: Compute object value match score
                if pred_object == true_object:
                    object_score = 1.0
                else:
                    object_score = 0.0  # If object value is incorrect, overall score is 0

                # Step 2: Compute qualifiers similarity score
                if object_score > 0:  # Only compute qualifiers if object value matches
                    pred_qual = {k: normalize_value(v) for k, v in pred_value.get("qualifiers", {}).items()}
                    true_qual = {k: normalize_value(v) for k, v in true_value.get("qualifiers", {}).items()}

                    start_time_score = match_start_time(pred_qual.get("start time", ""), true_qual.get("start time", ""))
                    end_time_score = match_start_time(pred_qual.get("end time", ""), true_qual.get("end time", ""))
                    stated_in_score = jaccard_similarity(pred_qual.get("stated in", []), true_qual.get("stated in", []))
                    ref_url_score = 1.0 if pred_qual.get("reference URL") == true_qual.get("reference URL") else 0.0

                    qualifier_score = (start_time_score + end_time_score + stated_in_score + ref_url_score) / 4  # Average score
                else:
                    qualifier_score = 0.0

                # Compute AESOP score for this predicate (weighted sum)
                predicate_score = (object_weight * object_score + qualifier_weight * qualifier_score) / total_weight
                predicate_scores.append(predicate_score)

            if predicate_scores:
                cost_matrix[i, j] = sum(predicate_scores) / len(predicate_scores)  # Compute entity-wise average

    # Step 2: Perform maximum weight matching using Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # Maximize matching scores

    # Step 3: Compute final AESOP score
    matched_scores = [cost_matrix[i, j] for i, j in zip(row_ind, col_ind)]
    final_aesop_score = sum(matched_scores) / len(matched_scores) if matched_scores else 0.0

    return final_aesop_score

#Precision, recalls. R1
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_metrics(tp, fp, fn):
    """计算精确率、召回率和 F1 分数"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def normalize_value(val):
    """简单归一化处理，比如去除前后空格"""
    if isinstance(val, str):
        return val.strip()
    return val


def build_triple_dict_list(triples):
    """
    构造字典，key 为 (name, relation)，value 为该 key 下所有三元组组成的列表。
    同时统一将关系中的 "state in" 修改为 "stated in"。
    """
    triple_dict = {}
    for triple in triples:
        name = triple.get("name")
        for relation, content in triple.items():
            if relation == "name":
                continue
            normalized_relation = relation.replace("state in", "stated in")
            key = (name, normalized_relation)
            triple_dict.setdefault(key, []).append(content)
    return triple_dict


def compare_qualifiers(true_qual, pred_qual):
    """
    逐键比较 qualifiers 的键值对（部分匹配）。
    - 对于 "start time" 和 "end time"，允许数字误差 ±1；
    - 对于列表，转换为集合比较（忽略顺序）；
    - 其他情况直接归一化字符串比较。

    返回该 triple 中 qualifier 的 TP、FP、FN 数量。
    """
    tp = 0
    for key, true_val in true_qual.items():
        if key in pred_qual:
            pred_val = pred_qual[key]
            if key in ["start time", "end time"]:
                try:
                    true_int = int(''.join(filter(str.isdigit, str(true_val))))
                    pred_int = int(''.join(filter(str.isdigit, str(pred_val))))
                    if abs(true_int - pred_int) <= 1:
                        tp += 1
                except Exception:
                    if normalize_value(true_val) == normalize_value(pred_val):
                        tp += 1
            elif isinstance(true_val, list) and isinstance(pred_val, list):
                if set(str(x).strip() for x in true_val) == set(str(x).strip() for x in pred_val):
                    tp += 1
            else:
                if normalize_value(true_val) == normalize_value(pred_val):
                    tp += 1
    fp = len(pred_qual) - tp  # 多预测的 qualifier 键值对数
    fn = len(true_qual) - tp  # 漏掉的 qualifier 键值对数
    return tp, fp, fn


def evaluate_and_print_triples_hierarchical(true_data, pred_data, w_main=0.7, w_qual=0.3):
    """
    层化评估：只有当主值匹配正确时才对该三元组的 qualifiers 进行比较；
    分别计算主值部分和 qualifiers 部分的指标，并按权重合成综合指标。
    """
    # 将列表转换成字典，键为 (name, relation)，值为列表
    true_dict = build_triple_dict_list(true_data)
    pred_dict = build_triple_dict_list(pred_data)

    all_keys = set(true_dict.keys()).union(set(pred_dict.keys()))

    # 用于主值部分的统计（微平均）
    tp_main_total = 0
    fp_main_total = 0
    fn_main_total = 0
    # qualifiers 仅统计那些主值匹配成功的对
    tp_qual_total = 0
    fp_qual_total = 0
    fn_qual_total = 0

    for key in all_keys:
        gold_list = true_dict.get(key, [])
        pred_list = pred_dict.get(key, [])

        n_gold = len(gold_list)
        n_pred = len(pred_list)

        # 构造主值匹配的代价矩阵：若主值匹配（归一化后相等），代价为 0，否则为 1
        cost_matrix = np.ones((n_gold, n_pred))
        for i, gold in enumerate(gold_list):
            gold_value = normalize_value(gold.get("value"))
            for j, pred in enumerate(pred_list):
                pred_value = normalize_value(pred.get("value"))
                if gold_value == pred_value:
                    cost_matrix[i, j] = 0

        # 若双方均非空，采用匈牙利算法进行匹配
        if n_gold > 0 and n_pred > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = np.array([]), np.array([])

        # 统计主值匹配成功（cost==0）的数量及记录对应对
        matched_main = 0
        matched_pairs = []  # 记录主值匹配成功的 (gold, pred) 对
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] == 0:
                matched_main += 1
                matched_pairs.append((gold_list[r], pred_list[c]))

        tp_main_total += matched_main
        fp_main_total += n_pred - matched_main
        fn_main_total += n_gold - matched_main

        # 层化：仅对主值匹配成功的对计算 qualifiers 部分
        for gold, pred in matched_pairs:
            true_qual = gold.get("qualifiers", {})
            pred_qual = pred.get("qualifiers", {})
            # 统一处理 "state in" -> "stated in"
            true_qual_norm = {k.replace("state in", "stated in"): v for k, v in true_qual.items()}
            pred_qual_norm = {k.replace("state in", "stated in"): v for k, v in pred_qual.items()}
            tp_q, fp_q, fn_q = compare_qualifiers(true_qual_norm, pred_qual_norm)
            tp_qual_total += tp_q
            fp_qual_total += fp_q
            fn_qual_total += fn_q
        # 注意：对于主值不匹配的 gold 或 pred，不再单独计入 qualifiers错误，
        # 因为层化评估的思想是：只有主值正确才有进一步评估 qualifiers 的意义.

    main_precision, main_recall, main_f1 = compute_metrics(tp_main_total, fp_main_total, fn_main_total)
    # 如果没有任何主值匹配，则 qualifiers 部分无法评估（设为0）
    qual_precision, qual_recall, qual_f1 = compute_metrics(tp_qual_total, fp_qual_total, fn_qual_total)

    # 综合：只有主值正确的情况下，qualifiers 才有意义，
    # 所以可以采用加权平均方式来计算整体指标
    overall_precision = w_main * main_precision + w_qual * qual_precision
    overall_recall = w_main * main_recall + w_qual * qual_recall
    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)
                  if (overall_precision + overall_recall) > 0 else 0)

    # 打印各部分结果
    print("【主值部分】")
    print("TP_main =", tp_main_total, "FP_main =", fp_main_total, "FN_main =", fn_main_total)
    print("Precision (main) = {:.3f}".format(main_precision))
    print("Recall    (main) = {:.3f}".format(main_recall))
    print("F1        (main) = {:.3f}".format(main_f1))
    print()
    print("【Qualifiers 部分】 (仅统计主值匹配成功的对)")
    print("TP_qual =", tp_qual_total, "FP_qual =", fp_qual_total, "FN_qual =", fn_qual_total)
    print("Precision (qual) = {:.3f}".format(qual_precision))
    print("Recall    (qual) = {:.3f}".format(qual_recall))
    print("F1        (qual) = {:.3f}".format(qual_f1))
    print()
    print("【综合评估】 (主值70% + qualifiers30%)")
    print("Overall Precision = {:.3f}".format(overall_precision))
    print("Overall Recall    = {:.3f}".format(overall_recall))
    print("Overall F1        = {:.3f}".format(overall_f1))

    # 返回详细指标
    metrics = {
        "main": {
            "tp": tp_main_total, "fp": fp_main_total, "fn": fn_main_total,
            "precision": main_precision, "recall": main_recall, "f1": main_f1
        },
        "qualifiers": {
            "tp": tp_qual_total, "fp": fp_qual_total, "fn": fn_qual_total,
            "precision": qual_precision, "recall": qual_recall, "f1": qual_f1
        },
        "overall": {
            "precision": overall_precision, "recall": overall_recall, "f1": overall_f1,
            "w_main": w_main, "w_qual": w_qual
        }
    }
    return metrics
