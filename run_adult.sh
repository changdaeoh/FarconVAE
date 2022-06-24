CUDA_VISIBLE_DEVICES=1 python main.py --scheduler=one --data_name='adult' --kernel=t --alpha=2.0 --beta=0.15 --gamma=0.75 --n_seed=10

CUDA_VISIBLE_DEVICES=1 python main.py --scheduler=one --data_name='adult' --kernel=g --alpha=1.0 --beta=0.05 --gamma=1.0 --n_seed=10