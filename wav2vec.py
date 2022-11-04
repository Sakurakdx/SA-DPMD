"""测试一下语音预训练模型的使用"""

from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch
from datasets import load_dataset


dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = Wav2Vec2Processor.from_pretrained("/data/kk/pretrained_model/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("/data/kk/pretrained_model/wav2vec2-base-960h")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
pass