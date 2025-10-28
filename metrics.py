import math

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set([actual[i]])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len([actual[user_id]]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set([actual[user_id]])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))

# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def mrr_at_k(actual, predicted, topk):
    """
    计算 Mean Reciprocal Rank (MRR) at k
    
    参数:
    - actual: list, 每个元素是用户的真实相关物品列表或单个物品ID
    - predicted: list of lists/arrays, 预测的物品排序列表
    - topk: int, 考虑的预测列表长度
    
    返回:
    - mrr: float
    """
    mrr_sum = 0.0
    
    for i in range(len(actual)):
        # 确保真实物品是列表类型
        if not isinstance(actual[i], (list, set)):
            act_items = [actual[i]]
        else:
            act_items = actual[i]
        
        # 获取预测的前topk个物品，并转换为列表
        pred_items = predicted[i][:topk].tolist()  # 将NumPy数组转换为列表
        
        # 查找第一个命中的位置
        rank = -1
        for j, item in enumerate(pred_items):
            if item in act_items:
                rank = j + 1  # 排名从1开始
                break
        
        # 计算 reciprocal rank
        if rank != -1:
            mrr_sum += 1.0 / rank
    
    # 避免除零错误
    if len(actual) == 0:
        return 0.0
        
    return mrr_sum / len(actual)