import numpy as np
import math
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Model
from tensorflow.keras.layers import *
from keras.callbacks import EarlyStopping


def super_attention(delta, pred_size, att_type):
    start_factor = 1 + delta
    end_factor = 1
    num_points = pred_size    
    
    if att_type == 'linear':
        factor = np.linspace(start_factor, end_factor, num_points)[np.newaxis,:,np.newaxis]

    elif att_type == 'exp': 
        factor = np.array([(start_factor)**((num_points-k-1)/(num_points-1)) 
                           for k in range(num_points)])[np.newaxis,:,np.newaxis]

    else:
        print("invalid delta")
        
    return factor

class RNN():
    def __init__(self,
                 history_arr, future_arr):
        self.history_arr = history_arr
        self.future_arr = future_arr
        
        self.history_train = []
        self.history_valid = []
        self.history_test  = []
        self.future_train  = []
        self.future_valid  = []
        self.future_test   = []
        
        self.history_train_sc = []
        self.history_valid_sc = []
        self.history_test_sc  = []
        self.future_train_sc  = []
        self.future_valid_sc  = []
        self.future_test_sc   = []
        
    def train_test(self, test_size, test_num):
        # test_size: 테스트 데이터의 비율
        # test_num: 테스트 시작 위치 지정
        num_sample = self.history_arr.shape[0]  # 전체 샘플 수
        num_test = int(num_sample*test_size)    # 테스트 샘플 수
        
        if test_num == -1:
            start_idx   = num_sample - num_test
            end_idx     = num_sample            
        else:
            start_idx   = num_test*test_num
            end_idx     = num_test*(test_num+1)
            
        assert end_idx <= num_sample 
        
        self.history_train  = np.concatenate([self.history_arr[range(0       ,start_idx),:,:],
                                                self.history_arr[range(end_idx ,num_sample),:,:]], axis=0)
        self.history_test   = self.history_arr[range(start_idx,end_idx),:,:]
        
        self.future_train   = np.concatenate([self.future_arr[range(0       ,start_idx),:,:],
                                                self.future_arr[range(end_idx ,num_sample),:,:]], axis=0)
        self.future_test    = self.future_arr[range(start_idx,end_idx),:,:]
        
    def train_valid(self, valid_size, random_state=42):
        self.history_train, self.history_valid = train_test_split(self.history_train, test_size=valid_size, random_state=random_state, shuffle=True)
        self.future_train, self.future_valid = train_test_split(self.future_train, test_size=valid_size, random_state=random_state, shuffle=True)        

    def scaling(self, scaler_type='standard'):
        if scaler_type == 'standard':
            self.history_mean = self.history_train[:,0,:].mean(axis=0)
            self.history_std  = self.history_train[:,0,:].std(axis=0)
            self.future_mean  = self.future_train[:,0,:].mean(axis=0)
            self.future_std   = self.future_train[:,0,:].std(axis=0)
            
            self.history_train_sc = (self.history_train-self.history_mean)/self.history_std
            self.history_test_sc  = (self.history_test-self.history_mean)/self.history_std
            self.future_train_sc  = (self.future_train-self.future_mean)/self.future_std
            self.future_test_sc   = (self.future_test-self.future_mean)/self.future_std
            
            if len(self.history_valid):
                self.history_valid_sc  = (self.history_valid-self.history_mean)/self.history_std                
                self.future_valid_sc   = (self.future_valid-self.future_mean)/self.future_std
                            
        elif scaler_type == 'minmax':
            self.history_min = self.history_train[:,0,:].min(axis=0)
            self.history_max = self.history_train[:,0,:].max(axis=0)
            self.future_min  = self.future_train[:,0,:].min(axis=0)
            self.future_max  = self.future_train[:,0,:].max(axis=0)
            
            self.history_train_sc = (self.history_train-self.history_min)/(self.history_max-self.history_min)
            self.history_test_sc  = (self.history_test-self.history_min)/(self.history_max-self.history_min)
            self.future_train_sc  = (self.future_train-self.future_min)/(self.future_max-self.future_min)
            self.future_test_sc   = (self.future_test-self.future_min)/(self.future_max-self.future_min)
            
            if len(self.history_valid):
                self.history_valid_sc  = (self.history_valid-self.history_min)/(self.history_max-self.history_min)          
                self.future_valid_sc   = (self.future_valid-self.future_min)/(self.future_max-self.future_min)          
                
    def unscaling(self, history_arr_sc, future_arr_sc, scaler_type='standard'):
        if scaler_type == 'standard':
            history_arr = history_arr_sc*self.history_std + self.history_mean
            future_arr  = future_arr_sc*self.future_std + self.future_mean
                            
        elif scaler_type == 'minmax':
            history_arr = history_arr_sc*(self.history_max-self.history_min) + self.history_min
            future_arr  = future_arr_sc*(self.future_max-self.future_min) + self.future_min 
                
        return history_arr, future_arr
           
    def build_model(self, num_layers, num_neurons, model_type, activation='relu'):
        # https://github.com/KerasKorea/KEKOxTutorial/blob/master/28_A_ten-minute_introduction_to_sequence-to-sequence_learning_in_Keras.md
        history_len = self.history_train.shape[1]
        history_num = self.history_train.shape[2]
        future_len  = self.future_train.shape[1]
        future_num  = self.future_train.shape[2]        
        
        if model_type == 'seq2seq_lstm':
            self.model = seq2seqLSTM(history_len, history_num, future_len, future_num, num_layers, num_neurons)
        
        
        
encoder_input = Input(shape=(history_len, history_num))


self.model = Model(encoder_input, decoder_output)

optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999)
self.model.compile(loss='mse',
                    optimizer = optimizer)
        

def seq2seqLSTM(history_len, history_num, future_len, future_num,
                num_layers=10, num_neurons=50): # sequence-to-sequence LSTM

    encoder_input = Input(shape=(history_len, history_num))
    
    # 인코더 층
    if num_layers == 1:
        encoder = LSTM(num_neurons, return_state=True, name='encoder')
        encoder_output, state_h, state_c = encoder(encoder_input)
        encoder_states = [state_h, state_c]
    else:
        for layer in range(num_layers):
            if not layer: # 첫번째 인코더 층
                encoder_output = LSTM(num_neurons, return_sequences=True, return_state=False, name="encoder_1")(encoder_input)
            elif layer == num_layers-1: # 마지막 인코더 층
                encoder_output, state_h, state_c  = LSTM(num_neurons, return_sequences=False, return_state=True, name=f"encoder_{layer+1}")(encoder_output)
                encoder_states = [state_h, state_c]
            else: # 중간 인코더 층
                encoder_output = LSTM(num_neurons, return_sequences=True, return_state=False, name=f"encoder_{layer+1}")(encoder_output)
                
    decoder_input = RepeatVector(future_len)(encoder_output)
    
    # 디코더 층
    if num_layers == 1:
        decoder = LSTM(num_neurons, return_sequences=True, name='decoder')
        decoder_output = decoder(decoder_input, initial_state=encoder_states)
    else:
        for layer in range(num_layers):
            if not layer: # 첫번째 디코더 층
                decoder_output = LSTM(num_neurons, return_sequences=True, return_state=False, name='decoder_1')(decoder_input, initial_state=encoder_states)
            else: # 디코더 층
                decoder_output  = LSTM(num_neurons, return_sequences=True, return_state=False, name=f"decoder_{layer+1}")(decoder_output)

    # 덴스 층                       
    decoder_dense = Dense(future_num)
    decoder_output = decoder_dense(decoder_output)
    
    # 모델 구성
    model = Model(encoder_input, decoder_output)
    optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mse', optimizer = optimizer)
    return model

def seq2seqGRU(history_len, history_num, future_len, future_num,
                num_layers=10, num_neurons=50): # sequence-to-sequence GRU
    
    encoder_input = Input(shape=(history_len, history_num))
    
    # 인코더 층
    if num_layers == 1:
        encoder = GRU(num_neurons, return_state=True, name='encoder')
        encoder_output, state_h = encoder(encoder_input)
    else:
        for layer in range(num_layers):
            if not layer: # 첫번째 인코더 층
                encoder_output = GRU(num_neurons,  return_sequences=True, return_state=False, name="encoder_1")(encoder_input)
            elif layer == num_layers-1: # 마지막 인코더 층
                encoder_output, state_h  = GRU(num_neurons, return_sequences=False, return_state=True, name=f"encoder_{layer+1}")(encoder_output)
            else: # 중간 인코더 층
                encoder_output = GRU(num_neurons, return_sequences=True, return_state=False, name=f"encoder_{layer+1}")(encoder_output)
    decoder_input = RepeatVector(future_len)(encoder_output)
    # 디코더 층
    if num_layers == 1:
        decoder = GRU(num_neurons, return_sequences=True, name='decoder')
        decoder_output = decoder(decoder_input, initial_state=state_h)
    else:
        for layer in range(num_layers):
            if not layer: # 첫번째 디코더 층
                decoder_output = GRU(num_neurons, return_sequences=True, return_state=False, name='decoder_1')(decoder_input, initial_state=state_h)
            else: # 디코더 층
                decoder_output  = GRU(num_neurons, return_sequences=True, return_state=False, name=f"decoder_{layer+1}")(decoder_output)
                
    # 덴스 층                    
    decoder_dense = Dense(future_num)
    decoder_output = decoder_dense(decoder_output)
    
    # 모델 구성
    model = Model(encoder_input, decoder_output)
    optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mse', optimizer = optimizer)
    return model

def ATTseq2seqLSTM(history_len, history_num, future_len, future_num,
                num_layers=10, num_neurons=50): # attention-based sequence-to-sequence LSTM
    
    encoder_input = Input(shape=(history_len, history_num))
    
    # 인코더 층
    if num_layers == 1:
        encoder = LSTM(num_neurons, return_sequences=True, return_state=True, name='encoder')
        encoder_output, state_h, state_c = encoder(encoder_input)
        encoder_states = [state_h, state_c]
    else:
        for layer in range(num_layers):
            if not layer: # 첫번째 인코더 층
                encoder_output = LSTM(num_neurons, return_sequences=True, return_state=False, name="encoder_1")(encoder_input)
            elif layer == num_layers-1: # 마지막 인코더 층
                encoder_output, state_h, state_c  = LSTM(num_neurons, return_sequences=True, return_state=True, name=f"encoder_{layer+1}")(encoder_output)
                encoder_states = [state_h, state_c]
            else: # 중간 인코더 층
                encoder_output = LSTM(num_neurons, return_sequences=True, return_state=False, name=f"encoder_{layer+1}")(encoder_output)
    decoder_input = RepeatVector(future_len)(state_h)
    # 디코더 층
    if num_layers == 1:
        decoder = LSTM(num_neurons, return_sequences=True, name='decoder')
        decoder_output = decoder(decoder_input, initial_state=encoder_states)
    else:
        for layer in range(num_layers):
            if not layer: # 첫번째 디코더 층
                decoder_output = LSTM(num_neurons, return_sequences=True, return_state=False, name='decoder_1')(decoder_input, initial_state=encoder_states)
            else: # 디코더 층
                decoder_output  = LSTM(num_neurons, return_sequences=True, return_state=False, name=f"decoder_{layer+1}")(decoder_output)
    
    # 어텐션 층 
    attention = dot([decoder_output, encoder_output], axes=[2,2])
    attention = Activation('softmax', name = 'alignment_score')(attention)
    context = dot([attention, encoder_output],axes=[2,1])
    decoder_conbined_context = concatenate([context, decoder_output])            
    
    # 덴스 층
    decoder_dense = Dense(future_num)
    decoder_output = decoder_dense(decoder_conbined_context)
        
    # 모델 구성
    model = Model(encoder_input, decoder_output)
    optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mse', optimizer = optimizer)
    return model

def ATTseq2seqGRU(history_len, history_num, future_len, future_num,
                num_layers=10, num_neurons=50): # attention-based sequence-to-sequence GRU
    
    encoder_input = Input(shape=(history_len, history_num))
    
    # 인코더 층
    if num_layers == 1:
        encoder = GRU(num_neurons,  return_sequences=True, return_state=True, name='encoder')
        encoder_output, state_h = encoder(encoder_input)
    else:
        for layer in range(num_layers):
            if not layer: # 첫번째 인코더 층
                encoder_output = GRU(num_neurons,  return_sequences=True, return_state=False, name="encoder_1")(encoder_input)
            elif layer == num_layers-1: # 마지막 인코더 층
                encoder_output, state_h  = GRU(num_neurons, return_sequences=True, return_state=True, name=f"encoder_{layer+1}")(encoder_output)
            else: # 중간 인코더 층
                encoder_output = GRU(num_neurons, return_sequences=True, return_state=False, name=f"encoder_{layer+1}")(encoder_output)
                
    decoder_input = RepeatVector(future_len)(state_h)
    
    # 디코더 층
    if num_layers == 1:
        decoder = GRU(num_neurons, return_sequences=True, name='decoder')
        decoder_output = decoder(decoder_input, initial_state=state_h)
    else:
        for layer in range(num_layers):
            if not layer: # 첫번째 디코더 층
                decoder_output = GRU(num_neurons, return_sequences=True, return_state=False, name='decoder_1')(decoder_input, initial_state=state_h)
            else: # 디코더 층
                decoder_output  = GRU(num_neurons, return_sequences=True, return_state=False, name=f"decoder_{layer+1}")(decoder_output)
    
    # 어텐션 층 
    attention = dot([decoder_output, encoder_output], axes=[2,2])
    attention = Activation('softmax', name = 'alignment_score')(attention)
    context = dot([attention, encoder_output],axes=[2,1])
    decoder_conbined_context = concatenate([context, decoder_output])            
    
    # 덴스 층
    decoder_dense = Dense(future_num)
    decoder_output = decoder_dense(decoder_conbined_context)
    
    # 모델 구성
    model = Model(encoder_input, decoder_output)
    optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mse', optimizer = optimizer)
    return model