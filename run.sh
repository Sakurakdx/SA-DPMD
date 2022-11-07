
# meld多模态cat实验
p=meld-cat && nohup python driver/train.py \
--config_file meld.cfg \
--save_dir save/$p \
> log/$p.log 2>&1 &


p=meld-cat && nohup python driver/train.py \
--config_file meld.cfg \
--save_dir save/$p \
--bert_dir /data/kk/pretrained_model/ssp_model200000 \
> log/$p.log 2>&1 &