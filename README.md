# SoWaSA

# Requriements
- Install Python, Pytorch(>=2.6.0). We use Python 3.10, Pytorch 2.6.0.
- If you plan to use GPU computation, install CUDA.


# Quick Start

## Environment Setting
```
conda env create -f sowasa_env.yaml
conda activate sowasa
```

## How to train

```
python main.py  --data_name [DATASET] \
                --model_type [MODEL_TYPE] \
                --lr [LEARNING_RATE] \
                --train_name [LOG_NAME]\
                --filter_type [FILTER_TYPE]
                --dwt_levels [DWT_LEVELS]
                --hidden_dropout_prob [DROPOUT_PROB]
```

- Example for Beauty
```
python main.py  --data_name Beauty \
                --model_type sowasa \
                --lr 0.001 \
                --train_name Beauty_sowasa \
                --filter_type sym2 \
                --dwt_levels 2 \
                --hidden_dropout_prob 0.5
```

## How to test

```
python main.py  --data_name [DATASET] \
                --model_type [MODEL_TYPE] \
                --lr [LEARNING_RATE] \
                --train_name [LOG_NAME]\
                --filter_type [FILTER_TYPE]
                --dwt_levels [DWT_LEVELS]
                --hidden_dropout_prob [DROPOUT_PROB]
                --load_model [PRETRAINED_MODEL_NAME] \
                --do_eval
```

- Example for Beauty
```
python main.py  --data_name Beauty \
                --model_type sowasa \
                --lr 0.001 \
                --train_name Beauty_sowasa \
                --filter_type db2 \
                --dwt_levels 2 \
                --hidden_dropout_prob 0.5
                --load_model Beauty_sowasa \
                --do_eval
```



# Dataset
In our experiments, we utilize four datasets, all stored in the `data` folder. All data preprocessing methods are based on the methods described in the [BSARec](https://github.com/yehjin-shin/BSARec) paper.


# Acknowledgement
This repository is based on [BSARec](https://github.com/yehjin-shin/BSARec).