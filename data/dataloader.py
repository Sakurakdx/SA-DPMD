import json
from typing import Counter
import torch
import copy
from data.dialog import *
import numpy as np
import re
import torchaudio as ta
import torch.nn.functional as F

def read_corpus(file, max_edu_num=10000, max_instance=10000000):
    audio_dir = "/".join(file.split("/")[:-1]) + "/audio/" + file.split("/")[-1].split(".")[0]
    with open(file, mode='r', encoding='UTF8') as infile:
        info = ""
        for line in infile.readlines():
            info += line
        data = json.loads(info)

        instances = []
        for dialog_info in data:
            instance = info2instance(dialog_info)
            instance.EDUs = instance.EDUs[:max_edu_num]
            instance.gold_arcs = instance.gold_arcs[:max_edu_num]
            instance.gold_rels = instance.gold_rels[:max_edu_num]
            sp_index(instance)

            # 构建audio文件的路径
            audio_list = [audio_dir + "/root.wav"]
            for id, edu in enumerate(instance.EDUs):
                if id == 0: continue
                audio_path = audio_dir + "/" + str(instance.id) + "_" + str(id) + ".wav"
                audio_list.append(audio_path)
            instance.audio_list = audio_list

            instances.append(instance)

            # filter
            if len(instances) > max_instance: break

        return instances

def sp_index(instance):
    speaker_counter = Counter()
    for edu in instance.EDUs:
        speaker = edu['speaker']
        speaker_counter[speaker] += 1
    id2sp = list()
    for sp in speaker_counter:
        id2sp.append(sp)
    reverse = lambda x: dict(zip(x, range(len(x))))
    sp2id = reverse(id2sp)

    instance.sp_index = list()
    for edu in instance.EDUs:
        speaker = edu['speaker']
        th_id = sp2id[speaker]
        instance.sp_index.append(th_id)

def info2instance(dialog_info):
    instance = Dialog()

    instance.id = dialog_info["id"]
    instance.original_EDUs = copy.deepcopy(dialog_info["edus"])

    root_edu = dict()
    root_edu['text'] = "<root>"
    root_edu['speaker'] = "<root>"
    root_edu['tokens'] = ["<root>"]

    instance.EDUs.append(root_edu)
    instance.EDUs += dialog_info["edus"]
    instance.id = dialog_info["id"]
    instance.relations = dialog_info["relations"]

    instance.real_relations = [[] for idx in range(len(instance.original_EDUs))]

    rel_matrix = np.zeros([len(instance.original_EDUs), len(instance.original_EDUs)]) ## arc flag
    for relation in instance.relations:
        index = relation['y']
        head = relation['x']
        if rel_matrix[index, head] >= 1: continue
        if head > index: continue
        if index >= len(instance.real_relations): continue
        if head >= len(instance.real_relations): continue

        rel_matrix[index, head] += 1
        instance.real_relations[index].append(relation)

    instance.sorted_real_relations = []
    for idx, rel_relation in enumerate(instance.real_relations):
        r = sorted(rel_relation,  key=lambda rel_relation:rel_relation['x'], reverse=False)
        instance.sorted_real_relations.append(r)

    instance.gold_arcs = [[] for idx in range(len(instance.EDUs))]
    instance.gold_rels = [[] for idx in range(len(instance.EDUs))]

    instance.gold_arcs[0].append(-1)
    instance.gold_rels[0].append('<root>')
    for idx, relation_list in enumerate(instance.sorted_real_relations):
        if len(relation_list) > 0:
            relation = relation_list[0]
            rel = relation['type']
            index = relation['y'] + 1
            head = relation['x'] + 1
            if head >= index:
                instance.gold_arcs[index].append(-1)
                instance.gold_rels[index].append('<root>')
            else:
                instance.gold_arcs[index].append(head)
                instance.gold_rels[index].append(rel)
    for idx, arc in enumerate(instance.gold_arcs):
        if len(arc) == 0:
            instance.gold_arcs[idx].append(0)
            instance.gold_rels[idx].append('<root>')

    for idx, arc in enumerate(instance.gold_arcs):
        assert len(arc) == 1
        assert arc[0] < idx
    for rel in instance.gold_rels:
        assert len(rel) == 1
    for idx, cur_EDU in enumerate(instance.EDUs):
        if idx == 0:
            turn = 0
        else:
            last_EDU = instance.EDUs[idx - 1]
            if last_EDU["speaker"] != cur_EDU["speaker"]:
                turn += 1
        cur_EDU["turn"] = turn

    return instance


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences


def data_iter(data, batch_size, shuffle=True):
    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

def batch_label_variable(onebatch, vocab):
    batch_gold_arcs = []
    batch_gold_rels = []
    for idx, instance in enumerate(onebatch):
        gold_arcs = np.zeros([len(instance.gold_arcs)])
        gold_rels = np.zeros([len(instance.gold_arcs)])
        for idy, gold_arc in enumerate(instance.gold_arcs):
            gold_arcs[idy] = instance.gold_arcs[idy][0]

        for idy, gold_rel in enumerate(instance.gold_rels):
            rel = instance.gold_rels[idy][0]
            if idy == 0:
                gold_rels[idy] = -1
            else:
                gold_rels[idy] = vocab.rel2id(rel)
        batch_gold_arcs.append(gold_arcs)
        batch_gold_rels.append(gold_rels)
    return batch_gold_arcs, batch_gold_rels


def batch_bert_variable(onebatch, config, tokenizer):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    for idx, instance in enumerate(onebatch):
        inst_texts = []
        for idy, EDU in enumerate(instance.EDUs):
            words = EDU['text'].split(" ")[:config.max_edu_len]
            text = " ".join(words)
            inst_texts.append(text)
        input_ids, token_type_ids, attention_mask = tokenizer.batch_bert_id(inst_texts)
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)
    batch_size = len(onebatch)
    max_edu_num = max([len(instance.EDUs) for instance in onebatch])
    max_tok_len = max([len(token_ids) for input_ids in input_ids_list for token_ids in input_ids])

    batch_input_ids = np.ones([batch_size, max_edu_num, max_tok_len], dtype=np.long) * tokenizer.pad_token_id()
    batch_token_type_ids = np.zeros([batch_size, max_edu_num, max_tok_len], dtype=np.long)
    batch_attention_mask = np.zeros([batch_size, max_edu_num, max_tok_len], dtype=np.long)
    token_lengths = np.ones([batch_size, max_edu_num])

    for idx in range(batch_size):
        edu_num = len(input_ids_list[idx])
        for idy in range(edu_num):
            tok_len = len(input_ids_list[idx][idy])
            token_lengths[idx][idy] = tok_len
            for idz in range(tok_len):
                batch_input_ids[idx, idy, idz] = input_ids_list[idx][idy][idz]
                batch_token_type_ids[idx, idy, idz] = token_type_ids_list[idx][idy][idz]
                batch_attention_mask[idx, idy, idz] = attention_mask_list[idx][idy][idz]

    batch_input_ids = torch.tensor(batch_input_ids)
    batch_token_type_ids = torch.tensor(batch_token_type_ids)
    batch_attention_mask = torch.tensor(batch_attention_mask)
    token_lengths = token_lengths.flatten()
    return batch_input_ids, batch_token_type_ids, batch_attention_mask, token_lengths

def batch_sp_variable(onebatch, vocab):
    batch_size = len(onebatch)
    edu_lengths = [len(instance.EDUs) for instance in onebatch]
    max_edu_len = max(edu_lengths)
    batch_sp = np.zeros([batch_size, max_edu_len], dtype=np.long)

    for idx, instance in enumerate(onebatch):
        for idy, u_id in enumerate(instance.sp_index):
            batch_sp[idx, idy] = u_id
    batch_sp = torch.tensor(batch_sp)
    return batch_sp

def batch_data_variable(onebatch, vocab):
    batch_size = len(onebatch)
    edu_lengths = [len(instance.EDUs) for instance in onebatch]
    max_edu_len = max(edu_lengths)
    arc_masks = np.zeros([batch_size, max_edu_len, max_edu_len])

    for idx, instance in enumerate(onebatch):
        edu_len = len(instance.EDUs)
        for idy in range(edu_len):
            for idz in range(idy):
                arc_masks[idx, idy, idz] = 1.

    arc_masks = torch.tensor(arc_masks)

    return edu_lengths, arc_masks

def batch_feat_variable(onebatch, vocab):
    batch_size = len(onebatch)
    edu_lengths = [len(instance.EDUs) for instance in onebatch]
    max_edu_len = max(edu_lengths)
    diaglog_feats = np.ones([batch_size, max_edu_len, max_edu_len, 3])

    for idx, instance in enumerate(onebatch):
        edu_len = len(instance.EDUs)
        for idy in range(edu_len):
            for idz in range(idy):
                diaglog_feats[idx, idy, idz, 0] = (idy - idz)
                diaglog_feats[idx, idy, idz, 1] = (instance.EDUs[idy]["speaker"] == instance.EDUs[idz]["speaker"])
                diaglog_feats[idx, idy, idz, 2] = (instance.EDUs[idy]["turn"] == instance.EDUs[idz]["turn"])

    diaglog_feats = torch.tensor(diaglog_feats).type(torch.FloatTensor)

    return diaglog_feats

def batch_audio_feature(onebatch, args):
    # convert to features 
    for inst in onebatch:
        audio_features = []
        for audio_path in inst.audio_list:
            wavform, sample_frequency = ta.load(audio_path) # [channel, time], 采样率
            # 获得Fbank特征
            feature = ta.compliance.kaldi.fbank(wavform, num_mel_bins=args.num_mel_bins, sample_frequency=sample_frequency, dither=0.0)
            # feature = ta.compliance.kaldi.mfcc(wavform, num_mel_bins=args.num_mel_bins, sample_frequency=sample_frequency, dither=0.0)
            if args.normalization:
                std, mean = torch.std_mean(feature)
                feature = (feature - mean) / std
            audio_features.append(feature)
        inst.audio_features = audio_features

    # convert batch features
    batch_size = len(onebatch)
    aduio_lengths = [len(instance.audio_list) for instance in onebatch]
    max_aduio_len = max(aduio_lengths)
    audio_feature_lengths = [len(audio_feat) for inst in onebatch for audio_feat in inst.audio_features]
    max_audio_feat_len = max(audio_feature_lengths)
    feature_length = onebatch[0].audio_features[0].shape[1]
    # batch_audio_feats = np.ones([batch_size * max_aduio_len, max_audio_feat_len, feature_length])
    # batch_audio_feature_masks = np.ones([batch_size * max_aduio_len, max_audio_feat_len, feature_length])
    
    padded_audio_features = []
    padded_audio_feature_masks = []
    for idx, inst in enumerate(onebatch):
        for idy, audio_feature in enumerate(inst.audio_features):
            audio_feature_length = len(audio_feature)
            padding_feature_len = max_audio_feat_len - audio_feature_length
            padded_audio_features.append(
                F.pad(audio_feature, pad=(0, 0, 0, padding_feature_len), value=0.0).unsqueeze(0))
            padded_audio_feature_masks.append([1] * audio_feature_length + [0] * padding_feature_len)

        while idy < max_aduio_len - 1:
            padded_audio_features.append(torch.zeros([1, max_audio_feat_len, feature_length]))
            padded_audio_feature_masks.append([0] * max_audio_feat_len)
            idy += 1

    batch_audio_feats = torch.cat(padded_audio_features, dim=0).view(batch_size , max_aduio_len, max_audio_feat_len, feature_length).float()
    batch_audio_feature_masks = (torch.IntTensor(padded_audio_feature_masks) > 0).view(batch_size , max_aduio_len, max_audio_feat_len)
    audio_feature_lengths = torch.IntTensor(audio_feature_lengths)

    return batch_audio_feats, batch_audio_feature_masks, audio_feature_lengths