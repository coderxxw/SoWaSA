import os
import torch
import numpy as np
import pickle

from model import MODEL_DICT
from trainers import Trainer
from utils import EarlyStopping, check_path, set_seed, parse_args, set_logger, calculate_trust_matrix_with_topk, prepare_trust_matrix_and_topk_dict, preprocess_trust_dict
from dataset import get_seq_dic, get_dataloder, get_rating_matrix


def main():
    args = parse_args()
    log_path = os.path.join(args.output_dir, args.train_name + '.log')
    logger = set_logger(log_path)

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')

    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1

    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(
        args, seq_dic, 
    )
    
    topk_trust_dict = None
    if args.model_type.lower() == "sowasa":
        topk_trust_dict = prepare_trust_matrix_and_topk_dict(seq_dic, args, logger)
        trust_graph_data = preprocess_trust_dict(topk_trust_dict)        
        model = MODEL_DICT[args.model_type.lower()](args=args, trust_graph_data=trust_graph_data)
    else:
        model = MODEL_DICT[args.model_type.lower()](args=args)
        
    logger.info(str(args))
    logger.info(model)

    trainer = Trainer(model, train_dataloader, eval_dataloader, test_dataloader, args, logger)
    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)

    if args.do_eval:
        if args.load_model is None:
            logger.info(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)
            logger.info(f"Load model from {args.checkpoint_path} for test!")

            scores, result_info = trainer.test(0)
            args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, logger=logger, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            logger.info(f"Epoch {epoch} finished")
            scores, _ = trainer.valid(epoch)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        logger.info("---------------Test Score---------------")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        _,_ = trainer.valid(0)
        
        scores, result_info = trainer.test(0)

    logger.info(args.train_name)


if __name__ == '__main__':
    main()