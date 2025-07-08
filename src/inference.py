import os
from dadagp import dadagp_conversion
#os.environ["CUDA_VISIBLE_DEVICES"]="0" #for GPU inference
from glob import glob
import pickle as pickle
from gen_utils import bass_trans_ev_model_tf, generate_bass_ev_trans_tf, create_onehot_enc
import numpy as np
import pathlib

thr_measures = 16
thr_max_tokens = 800
thr_min_tokens = 50
dec_seq_length = 773

'''load Encoders pickle for onehotencoders'''

#encoders pickle is created during pre-processing
encoders_trans = r'../ckpt/bass_encoders_cp.pickle'


with open(encoders_trans, 'rb') as handle:
    TransEncoders = pickle.load(handle)

'''Load Inference Transformer. You may download pre-trained model based 
on the paper. See instructions in ReadME.md'''
trans_bass_hb = bass_trans_ev_model_tf(TransEncoders, dec_seq_length)


'''Set Temperature'''
temperature = 0.9
# Remember to unzip data.zip before that
with open("../data/test_set_streams.pickle", 'rb') as f:
    test_set = pickle.load(f)

enc_in = test_set["Encoder_Input"]

print(f"Number of examples to generate: {len(enc_in)}.")

count = 0
for i, seq in enumerate(enc_in):
    if count == 200:
        trans_bass_hb = bass_trans_ev_model_tf(TransEncoders, dec_seq_length)
        count = 0
    count += 1

    generated_tokens = []
    save_path = f"gen_path/{i}_generated_bass.tokens.txt"
    gp_save_path = f"gen_path/{i}_generated_bass.gp5"
    if pathlib.Path(save_path).exists():
        print("already processed this file: ", save_path)
        continue

    if not (thr_min_tokens <= len(seq) <= thr_max_tokens):
        print(f"Token count {len(sequence_input)} not in range {thr_min_tokens} - {thr_max_tokens}")
        continue

    bass_HB = generate_bass_ev_trans_tf(trans_bass_hb, TransEncoders, temperature, seq, dec_seq_length=dec_seq_length)
    with open(save_path, 'w') as f:
        for token in bass_HB:
            f.write(f"{token}\n")

    print(f"Saved Generated sequence to {save_path}.")

    # Convert to .gp5
    dadagp_conversion('decode', save_path, gp_save_path, verbose=False)
    print(f"Saved corresponding gp5 file to {gp_save_path}.")


