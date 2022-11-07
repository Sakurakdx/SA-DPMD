

# meld baseline
nohup python driver/train.py --config_file molweni.cfg > log/base-meld.log 2>&1 & 

# meld baseline 编码器改为预训练的编码器
nohup python driver/train.py \
--config_file meld.cfg \
--save_dir save/meld-baseline-pretrain \
--bert_dir /data/kk/pretrained_model/ssp_model200000 \
> log/base-meld-pretrain.log 2>&1 & 
