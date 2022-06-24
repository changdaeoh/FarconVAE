CUDA_VISIBLE_DEVICES=6 python main.py --scheduler=lr --data_name='german' --kernel=t --alpha=0.7  --beta=0.15 --gamma=1.0 --n_seed=10

CUDA_VISIBLE_DEVICES=6 python main.py --scheduler=lr --data_name='german' --kernel=g --alpha=0.7  --beta=0.2 --gamma=1.0 --n_seed=10