CUDA_VISIBLE_DEVICES=0 python main.py \
--exp \
--resume checkpoints/step_080000.pth \
--exp_type rotation tension star5 mei realcrack \
--num_transformer_layers 12 \
--upsample_factor 2 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \

