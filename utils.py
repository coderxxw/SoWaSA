import os
import random
import torch
import datetime
import argparse
import numpy as np
import logging

from math import sqrt
from tqdm import tqdm
import pickle

def set_logger(log_path, log_name='seqrec', mode='a'):
    """set up log file
    mode : 'a'/'w' mean append/overwrite,
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, mode=mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

def parse_args():

    parser = argparse.ArgumentParser()

    # basic args
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Beauty", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--load_model", default=None, type=str)
    parser.add_argument("--train_name", default=get_local_time(), type=str)
    parser.add_argument("--device", default="cuda", type=str)

    # train args
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate of adam")
    parser.add_argument("--batch_size", default=256, type=int, help="number of batch_size")
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", default=1, type=int, help="per epoch print res")
    parser.add_argument("--patience", default=10, type=int, help="how long to wait after last time validation loss improved")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--checkpoint_path", default="output/checkpoint.pth", type=str, help="加载模型的路径")

    parser.add_argument("--seed", default=5678, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")
    parser.add_argument("--variance", default=5, type=float)

    # model args
    parser.add_argument("--model_type", default='sowasa', type=str) #sowasa, bsarec, bert4rec
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--hidden_size", default=64, type=int, help="embedding dimension")
    parser.add_argument("--num_hidden_layers", default=2, type=int, help="number of blocks")
    parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--initializer_range", default=0.02, type=float)


    args, _ = parser.parse_known_args()

    if args.model_type.lower() == 'bsarec':
        parser.add_argument("--c", default=3, type=int)
        parser.add_argument("--alpha", default=0.9, type=float)

    elif args.model_type.lower() == 'bert4rec':
        parser.add_argument("--mask_ratio", default=0.2, type=float)

    elif args.model_type.lower() == 'caser':
        parser.add_argument("--nh", default=8, type=int)
        parser.add_argument("--nv", default=4, type=int)
        parser.add_argument("--reg_weight", default=1e-4, type=float)

    elif args.model_type.lower() == 'duorec':
        parser.add_argument("--tau", default=1.0, type=float)
        parser.add_argument("--lmd", default=0.1, type=float)
        parser.add_argument("--lmd_sem", default=0.1, type=float)
        parser.add_argument("--ssl", default='us_x', type=str)
        parser.add_argument("--sim", default='dot', type=str)

    elif args.model_type.lower() == 'fearec':
        parser.add_argument("--tau", default=1.0, type=float)
        parser.add_argument("--lmd", default=0.1, type=float)
        parser.add_argument("--lmd_sem", default=0.1, type=float)
        parser.add_argument("--ssl", default='us_x', type=str)
        parser.add_argument("--sim", default='dot', type=str)
        parser.add_argument("--spatial_ratio", default=0.1, type=float)
        parser.add_argument("--global_ratio", default=0.6, type=float)
        parser.add_argument("--fredom_type", default='us_x', type=str)
        parser.add_argument("--fredom", default='True', type=str) # use eval function to use as boolean

    elif args.model_type.lower() == 'gru4rec':
        parser.add_argument("--gru_hidden_size", default=64, type=int, help="hidden size of GRU")
    
    elif args.model_type.lower() == 'sowasa':
        parser.add_argument("--filter_type", default='sym2', type=str, help="小波滤波器类型 (haar, db2, sym2)")
        parser.add_argument("--dwt_levels", default=2, type=int, help="小波分解层数")
        parser.add_argument("--min_gain", default=0, type=float, help="最小增益")
        parser.add_argument("--max_gain", default=5.0, type=float, help="最大增益")


        parser.add_argument('--threshold', type=float, default=0.8, help='Threshold for trust calculation')
        parser.add_argument('--trust_topk', type=int, default=3, help='每个用户的 top-K 信任邻居数量')
        parser.add_argument("--min_common_items", type=int, default=5, help="最小交互次数")
        parser.add_argument("--social_layers", default=2, type=int, help="图卷积层数")



    return parser.parse_args()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, logger, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.logger = logger
        

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.logger.info(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score

def calculate_trust_matrix_with_topk(user_seq, args, logger=None):
    num_users = len(user_seq)
    topk = args.trust_topk
    threshold = args.threshold
    min_common = args.min_common_items

    def get_items_from_seq(seq):
        if seq is None:
            return []
        if isinstance(seq, dict):
            return seq.get('items') or seq.get('seq') or []
        return seq

    if logger is not None:
        logger.info(f"构建 item->users 倒排表 (用户数={num_users}) ...")
    item_to_users = {}
    for u, seq in tqdm(enumerate(user_seq), total=num_users, desc="构建倒排表"):
        items = get_items_from_seq(seq)
        for it in items:
            item_to_users.setdefault(it, []).append(u)

    user_deg = np.zeros(num_users, dtype=np.int32)
    for u, seq in enumerate(user_seq):
        items = get_items_from_seq(seq)
        if 0 <= u < num_users:
            user_deg[u] = len(items)
    user_deg_safe = user_deg.astype(np.float32)
    user_deg_safe[user_deg_safe == 0] = 1.0

    counts = np.zeros(num_users, dtype=np.int32)
    touched = []

    topk_trust_dict = {}

    for u in tqdm(range(num_users), desc="计算 top-K 信任邻居"):
        seq_u = user_seq[u] if u < len(user_seq) else []
        items_u = get_items_from_seq(seq_u)
        if not items_u or user_deg[u] == 0:
            topk_trust_dict[u] = []
            continue

        for it in items_u:
            for v in item_to_users.get(it, ()):
                if v == u:
                    continue
                if counts[v] == 0:
                    touched.append(v)
                counts[v] += 1

        if not touched:
            topk_trust_dict[u] = []
            continue

        touched_arr = np.array(touched, dtype=np.int32)
        common_counts = counts[touched_arr].astype(np.float32)

        valid_mask = common_counts >= float(min_common)
        if not np.any(valid_mask):
            topk_trust_dict[u] = []
            for idx in touched:
                counts[idx] = 0
            touched.clear()
            continue

        cand_idx = touched_arr[valid_mask]
        cand_counts = common_counts[valid_mask]

        deg_u = user_deg_safe[u]
        trust_scores = cand_counts / deg_u

        mask2 = trust_scores > float(threshold)
        if not np.any(mask2):
            topk_trust_dict[u] = []
            for idx in touched:
                counts[idx] = 0
            touched.clear()
            continue

        cand_idx = cand_idx[mask2]
        trust_scores = trust_scores[mask2]

        if cand_idx.size <= topk:
            order = np.argsort(-trust_scores)
            sel_idx = cand_idx[order]
            sel_scores = trust_scores[order]
        else:
            part = np.argpartition(-trust_scores, topk - 1)[:topk]
            top_part = part[np.argsort(-trust_scores[part])]
            sel_idx = cand_idx[top_part]
            sel_scores = trust_scores[top_part]

        neighbors = [{'user_id': int(int_v), 'trust': float(round(float(s), 4))} for int_v, s in zip(sel_idx, sel_scores)]
        topk_trust_dict[u] = neighbors

        for idx in touched:
            counts[idx] = 0
        touched.clear()

    return topk_trust_dict



def prepare_trust_matrix_and_topk_dict(seq_dic, args, logger):
    topk_trust_dict_path = os.path.join(args.output_dir, args.data_name + "_topk_trust_dict.pkl")
    if not os.path.exists(topk_trust_dict_path):
        logger.info("计算信任矩阵和 top-K 信任邻居...")
        topk_trust_dict = calculate_trust_matrix_with_topk(
            seq_dic['user_seq'],
            args
        )
        with open(topk_trust_dict_path, 'wb') as f:
            pickle.dump(topk_trust_dict, f)
        logger.info(f"top-K 信任邻居保存至 {topk_trust_dict_path}")
    else:
        with open(topk_trust_dict_path, 'rb') as f:
            topk_trust_dict = pickle.load(f)
        logger.info(f"从 {topk_trust_dict_path} top-K 信任邻居")
    return topk_trust_dict


def preprocess_trust_dict(trust_dict):
    edge_index = []
    edge_weight = []
    for user_id, neighbors in trust_dict.items():
        for neighbor in neighbors:
            neighbor_id = neighbor['user_id']
            trust = neighbor['trust']
            edge_index.append([user_id, neighbor_id])
            edge_weight.append(trust)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    
    return {'edge_index': edge_index, 'edge_weight': edge_weight}