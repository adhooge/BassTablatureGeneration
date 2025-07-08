import tensorflow as tf
import numpy as np
from tqdm import tqdm
from aux_files.aux_train_tf import HybridTransformer, create_masks


def create_onehot_enc(Encoder_RG, TransEncoders):
    vocab = TransEncoders[0].categories_[0]  # Get the vocabulary from the encoder
    
    # Check for out-of-vocabulary tokens
    oov_tokens = [token for token in Encoder_RG if token not in vocab]
    if oov_tokens:
        print("Warning: The following tokens are not in the vocabulary:", oov_tokens)

    # Transform sequence with 'sos' and 'eos'
    Enc_Input = TransEncoders[0].transform(np.array(['sos'] + Encoder_RG + ['eos']).reshape(-1, 1)).toarray()
    
    # Convert one-hot vectors to indices
    Enc_Input = [np.where(r == 1)[0][0] for r in Enc_Input]  # for embeddings
    
    # Shift by one to use 0 as padding
    Enc_Input = [x + 1 for x in Enc_Input]  
    
    return Enc_Input


def sample(preds, temperature=1.0):
    '''
    @param preds: a np.array with the probabilities to all categories
    @param temperature: the temperature. Below 1.0 the network makes more "safe"
                        predictions
    @return: the index after the sampling
    '''
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
   
    return np.argmax(probas)  



def generate_bass_ev_trans_tf(trans_bass, TransEncoders, temperature, Encoder_RG, dec_seq_length):
         
    #convert Encoder Inps to tensors
    Encoder_RG = tf.convert_to_tensor([Encoder_RG])
    
    #Prepare Decoder Inps 
    Bass_sos_idx = int(np.where(TransEncoders[1].categories_[0] == 'sos')[0][0]) + 1
    Bass_eos_idx = int(np.where(TransEncoders[1].categories_[0] == 'eos')[0][0]) + 1

    
    dec_bass_out = []
    dec_out_bass = [Bass_sos_idx]
    dec_out_bass = tf.convert_to_tensor(dec_out_bass)
    dec_out_bass = tf.expand_dims(dec_out_bass, 0)
    
    #start generating autoregressively WORD TRANSFORMER
    for _ in tqdm(range(dec_seq_length)):
       #masking
       
       _, combined_mask, dec_padding_mask = create_masks(Encoder_RG, dec_out_bass)
       
       preds_bass, _ = trans_bass(Encoder_RG, dec_out_bass, combined_mask, dec_padding_mask, training=False)
       
       #bass Out
       preds_bass = preds_bass[:, -1:, :].numpy() # (batch_size, 1, vocab_size) select the last word
       token_preds_bass = preds_bass[-1,:].reshape(preds_bass.shape[-1],)
       #apply diversity
       token_pred_dr = sample(token_preds_bass, temperature)
       dec_bass_out.append(token_pred_dr) #for numpy

       if token_pred_dr == Bass_eos_idx:
           print('Generated', len(dec_bass_out), 'tokens')
           break    #EOS
       
       dec_out_bass = tf.concat([dec_out_bass, tf.convert_to_tensor([[token_pred_dr]])], axis=1)

    #convert outs to event based rep
    dec_bass_out = [x-1 for x in dec_bass_out] #shift by -1 to get the original
    bassEV = []
    for i in range(0,len(dec_bass_out)-1): #exclude eos
        bassEV.append(str(TransEncoders[1].categories_[0][dec_bass_out[i]]))
    
    return bassEV


def bass_trans_ev_model_tf(TransEncoders, dec_seq_length):
    
    #get vocabs
    enc_vocab = len(TransEncoders[0].categories_[0])
    
    dec_vocab = len(TransEncoders[1].categories_[0])
    
    print(enc_vocab, dec_vocab)
    #create the architecture first  
    num_layers = 4  #4
    d_model_enc = 240 #Encoder Embedding
    
    d_model_dec = 192 #Decoder Embedding
    units = 1024 #for Dense
    num_heads = 8 #8
    dropout_rate = 0.3
    
    #for relative attention

    rel_dec_seq = dec_seq_length #int(dec_seq_length/2) 

    model = HybridTransformer(num_layers=num_layers, d_model_enc=d_model_enc,
                            d_model_dec=d_model_dec, num_heads=num_heads,
                            dff=units, input_vocab=enc_vocab+1, target_vocab=dec_vocab+1, 
                            pe_target=dec_seq_length, 
                            mode_choice='relative', #change to multihead for vanilla attention mechanism
                            max_rel_pos_tar=rel_dec_seq, rate=dropout_rate)
    
    checkpoint_path = './aux_files/checkpoints/'
    print('Loading Hybrid Music Transformer')

    #Set Optimizers and load checkpoints
    
    optimizer = tf.keras.optimizers.Adam(0.0005, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
        
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
    
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
      ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
      print('Latest checkpoint restored!')
    
    return model
