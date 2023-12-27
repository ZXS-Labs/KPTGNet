# CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_rs/202309121651 --checkpoint_path logs/log_rs/202309100944/checkpoint.tar --camera realsense --dataset_root data/Benchmark/graspnet

# 202310062321_pts_30000
python test.py --dump_dir logs/dump_rs_spotr/2023-10-06-23-21 \
--checkpoint_path logs/log_rs_spotr/2023-10-06-23-21/checkpoint.tar \
--camera realsense --dataset_root data/Benchmark/graspnet

# CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_kn --checkpoint_path logs/log_kn/checkpoint.tar --camera kinect --dataset_root /data/Benchmark/graspnet
