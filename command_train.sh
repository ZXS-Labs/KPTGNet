python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --max_restarts 0 --module train_dist \
--camera realsense --log_dir logs/log_rs_spotr/2023-11-07-17-30 --batch_size 2 \
--dataset_root data/Benchmark/graspnet --learning_rate 0.001

# CUDA_VISIBLE_DEVICES=0 python train.py --camera realsense --log_dir logs/log_rs/202309100944 --batch_size 4 --dataset_root data/Benchmark/graspnet
