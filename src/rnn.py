import numpy as np
import math
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
        
        
    def train_test_split(self, test_size, test_num):
        # test_size: 테스트 데이터의 비율
        # test_num: 테스트 시작 위치 지정
        
        num_sample = self.history_arr.shape[0]
        num_test = int(num_sample*test_size)
        
        start_idx   = num_test*test_num
        end_idx     = num_test*(test_num+1)
        
        assert end_idx < num_sample 
        
        self.history_train  = np.concatenate([self.history_arr[range(0       ,start_idx),:,:],
                                              self.history_arr[range(end_idx ,num_sample),:,:],
                                              axis=0]
        self.history_test   = self.history_arr[range(start_idx,end_idx),:,:]
        
        self.future_train   = np.concatenate([self.future_arr[range(0       ,start_idx),:,:],
                                              self.future_arr[range(end_idx ,num_sample),:,:],
                                              axis=0]
        self.future_test    = self.future_arr[range(start_idx,end_idx),:,:]


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
            
        elif scaler_type == 'minmax':
            self.history_min = self.history_train[:,0,:].min(axis=0)
            self.history_max = self.history_train[:,0,:].max(axis=0)
            self.future_min  = self.future_train[:,0,:].min(axis=0)
            self.future_max  = self.future_train[:,0,:].max(axis=0)
            
            self.history_train_sc = (self.history_train-self.history_min)/(self.history_max-self.history_min)
            self.history_test_sc  = (self.history_test-self.history_min)/(self.history_max-self.history_min)
            self.future_train_sc  = (self.future_train-self.future_min)/(self.future_max-self.future_min)
            self.future_test_sc   = (self.future_test-self.future_min)/(self.future_max-self.future_min)
            
            
    def model(self, num_layers, num_neurons, rnn_type, activation='relu'):
        history_len = self.history_train.shape[1]
        history_var = self.history_train.shape[2]
        future_len  = self.future_train.shape[1]
        future_var  = self.future_train.shape[2]        
        
        # https://github.com/KerasKorea/KEKOxTutorial/blob/master/28_A_ten-minute_introduction_to_sequence-to-sequence_learning_in_Keras.md
        
        """
        from keras.models import Model
        from keras.layers import Input, LSTM, Dense

        # 입력 시퀀스의 정의와 처리
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # `encoder_outputs`는 버리고 상태(`state_h, state_c`)는 유지
        encoder_states = [state_h, state_c]

        # `encoder_states`를 초기 상태로 사용해 decoder를 설정
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        # 전체 출력 시퀀스를 반환하고 내부 상태도 반환하도록 decoder를 설정. 
        # 학습 모델에서 상태를 반환하도록 하진 않지만, inference에서 사용할 예정.
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                            initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # `encoder_input_data`와 `decoder_input_data`를 `decoder_target_data`로 반환하도록 모델을 정의
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        """
        
        
        if rnn_type == 's2s_gru':
            encoder_input = Input(shape=(self.history_train.shape[1], self.history_train.shape[2]))
            
            if num_layers == 1:
                encoder = GRU(num_neurons, return_state=True, name='encoder')
                encoder_out, state_h = encoder(encoder_input)
                
            else:
                #인코더 은닉층
                hidden_layers = []
                for layer in range(num_layers):
                    if not layer:
                        encoder_output, encoder_h, encoder_c  = GRU(num_neurons, 
                                                                    return_sequences=True,
                                                                    return_state=False, name="encoder_last")(encoder_input)
                    
                    elif layer == num_layers-1:
                        encoder_output, encoder_h, encoder_c  = GRU(num_neurons, 
                                                                    return_sequences=False,
                                                                    return_state=True, name="encoder_last")(encoder_input)
                    else:
                        hidden_state_en = GRU(num_neurons, 
                                            return_sequences=True,
                                            return_state=False)(hidden_state_en)

            decoder_input = Input(shape=(None, self.history_train.shape[2]))
            RepeatVector(output.shape[1])(encoder_last_h1)

            if num_layers == 1:
                hidden_state_de = decoder_input

            else:
                #디코더 은닉층
                for layer in range(num_layers-1):
                    if layer == 0:
                        hidden_state_de = GRU(num_neurons, 
                                            return_sequences=True,
                                            return_state=False)(decoder_input)
                    else:
                        hidden_state_de = GRU(num_neurons, 
                                            return_sequences=True,
                                            return_state=False)(hidden_state_de)

            decoder = GRU(num_neurons,
                        return_sequences = True,
                        return_state=False)(hidden_state_de)  #, initial_state=[encoder_last_h1,encoder_last_c ])

            out = TimeDistributed(Dense(output.shape[2]))(decoder)

            if dnn_neurons:
                out = Dense(Len_Futr)(out)
                out = Dense(output.shape[2])(out)

            model = Model(inputs=input, outputs=out)

            optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999)
            model.compile(loss='mse',
                        optimizer = optimizer)

            return model