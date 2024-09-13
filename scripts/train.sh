export OMP_NUM_THREADS=1
NUM_GPUS=2
CHECKPOINT_DIR=checkpoints/new && \
mkdir -p ${CHECKPOINT_DIR} && \
nohup python -W ignore::FutureWarning -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port=9989 main.py \
--launcher pytorch \
--checkpoint_dir ${CHECKPOINT_DIR} \
--stage speckle \
--batch_size 12 \
--val_dataset speckle \
--lr 2e-4 \
--num_transformer_layers 12 \
--upsample_factor 2 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--val_freq 5000 \
--save_ckpt_freq 5000 \
--num_steps 100000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log &
