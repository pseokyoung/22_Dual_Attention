import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping

from src import models
from src import evaluation


def super_attention(delta, future_size, future_num, att_type):
    start_factor = 1 + delta
    end_factor = 1
    num_points = future_size    
    
    if att_type == 'linear':
        factor = np.linspace(start_factor, end_factor, num_points)
        factor = np.array([factor, factor, factor]).T
        factor = factor[np.newaxis,:,:]

    elif att_type == 'exp': 
        factor = np.array([(start_factor)**((num_points-k-1)/(num_points-1)) for k in range(num_points)])
        factor = np.array([factor, factor, factor]).T
        factor = factor[np.newaxis,:,:]

    else:
        print("invalid delta")
        
    return factor

class RNN():
    def __init__(self,
                 history_series, history_var, future_series, future_var):
        self.history_series = history_series
        self.history_var = history_var
        self.future_series = future_series
        self.future_var = future_var        
        
        self.history_train = []
        self.history_valid = []
        self.history_test  = []
        self.future_train  = []
        self.future_valid  = []
        self.future_test   = []
        
        self.scaler_type = ""
        self.history_train_sc = []
        self.history_valid_sc = []
        self.history_test_sc  = []
        self.future_train_sc  = []
        self.future_valid_sc  = []
        self.future_test_sc   = []
        
        self.model_type = ""
        self.factor = []
        
    def train_test(self, test_size, test_num):
        # test_size: 테스트 데이터의 비율
        # test_num: 테스트 시작 위치 지정
        num_sample = self.history_series.shape[0]  # 전체 샘플 수
        num_test = int(num_sample*test_size)    # 테스트 샘플 수
        
        if test_num == -1:
            start_idx   = num_sample - num_test
            end_idx     = num_sample            
        else:
            start_idx   = num_test*test_num
            end_idx     = num_test*(test_num+1)
            
        assert end_idx <= num_sample 
        
        self.history_train  = np.concatenate([self.history_series[range(0       ,start_idx),:,:],
                                                self.history_series[range(end_idx ,num_sample),:,:]], axis=0)
        self.history_test   = self.history_series[range(start_idx,end_idx),:,:]
        
        self.future_train   = np.concatenate([self.future_series[range(0       ,start_idx),:,:],
                                                self.future_series[range(end_idx ,num_sample),:,:]], axis=0)
        self.future_test    = self.future_series[range(start_idx,end_idx),:,:]
        
    def train_valid(self, valid_size, random_state=42):
        self.history_train, self.history_valid = train_test_split(self.history_train, test_size=valid_size, random_state=random_state, shuffle=True)
        self.future_train, self.future_valid = train_test_split(self.future_train, test_size=valid_size, random_state=random_state, shuffle=True)        

    def scaling(self, scaler_type='standard'):
        if scaler_type == 'standard':
            self.scaler_type = scaler_type
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
            self.scaler_type = scaler_type
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
                
    def unscaling(self, history_series_sc, future_series_sc):
        assert self.scaler_type != ""
        
        if self.scaler_type == 'standard':
            history_series = history_series_sc*self.history_std + self.history_mean
            future_series  = future_series_sc*self.future_std + self.future_mean
                            
        elif self.scaler_type == 'minmax':
            history_series = history_series_sc*(self.history_max-self.history_min) + self.history_min
            future_series  = future_series_sc*(self.future_max-self.future_min) + self.future_min 
                
        return history_series, future_series
           
    def build_model(self, num_layers, num_neurons, dense_layers, dense_neurons, model_type, factor=[], activation='relu'):
        # https://github.com/KerasKorea/KEKOxTutorial/blob/master/28_A_ten-minute_introduction_to_sequence-to-sequence_learning_in_Keras.md
        history_size = self.history_train.shape[1]
        history_num = self.history_train.shape[2]
        future_size  = self.future_train.shape[1]
        future_num  = self.future_train.shape[2]
        
        if model_type == 'seq2seq_lstm':
            self.model_type = model_type
            self.model = models.seq2seqLSTM(history_size, history_num, future_size, future_num, num_layers, num_neurons, dense_layers, dense_neurons)
            
        elif model_type == 'seq2seq_gru':
            self.model_type = model_type
            self.model = models.seq2seqGRU(history_size, history_num, future_size, future_num, num_layers, num_neurons, dense_layers, dense_neurons)
            
        elif model_type == 'att_seq2seq_lstm':
            self.model_type = model_type
            self.model = models.ATTseq2seqLSTM(history_size, history_num, future_size, future_num, num_layers, num_neurons, dense_layers, dense_neurons)
            
        elif model_type == 'att_seq2seq_gru':
            self.model_type = model_type
            self.model = models.ATTseq2seqGRU(history_size, history_num, future_size, future_num, num_layers, num_neurons, dense_layers, dense_neurons)  
            
        elif model_type == 'datt_seq2seq_lstm':
            self.model_type = model_type
            self.model = models.DATTseq2seqLSTM(history_size, history_num, future_size, future_num, factor, num_layers, num_neurons, dense_layers, dense_neurons)
            self.factor = factor
            self.future_train_wt = self.future_train_sc*factor
            self.future_test_wt = self.future_test_sc*factor
            if len(self.history_valid):      
                self.future_valid_wt   = self.future_valid_sc*factor
                
        elif model_type == 'datt_seq2seq_gru':
            self.model_type = model_type
            self.model = models.DATTseq2seqGRU(history_size, history_num, future_size, future_num, factor, num_layers, num_neurons, dense_layers, dense_neurons)  
            self.factor = factor
            self.future_train_wt = self.future_train_sc*factor
            self.future_test_wt = self.future_test_sc*factor
            if len(self.history_valid):      
                self.future_valid_wt   = self.future_valid_sc*factor
            
    def train(self, epochs=10000, verbose=2, batch_size=32, patience=10, monitor='val_loss'):
        early_stopping_cb = EarlyStopping(patience=patience, restore_best_weights= True, monitor=monitor)
        
        time_start = time.time()
        if self.model_type == 'datt_seq2seq_lstm' or self.model_type == 'datt_seq2seq_gru':
            self.history = self.model.fit(self.history_train_sc, self.future_train_wt, 
                                          epochs=epochs, callbacks=[early_stopping_cb], verbose=verbose, batch_size=batch_size, 
                                          validation_data=(self.history_valid_sc, self.future_valid_wt))
        else:
            self.history = self.model.fit(self.history_train_sc, self.future_train_sc, 
                                          epochs=epochs, callbacks=[early_stopping_cb], verbose=verbose, batch_size=batch_size, 
                                          validation_data=(self.history_valid_sc, self.future_valid_sc))
        time_end = time.time()
        self.train_time = time_start - time_end
        
    def save_model(self, save_path):
        self.model.save(save_path)
        
    def test(self):        
        if self.model_type == 'datt_seq2seq_lstm' or self.model_type == 'datt_seq2seq_gru':      
            prediction_wt = self.model.predict(self.history_test_sc)            # weighted prediction
            prediction_sc = prediction_wt/self.factor                           # scaled prediction
            _, prediction = self.unscaling(self.history_test_sc, prediction_sc) # original prediction
            self.r2 = evaluation.r_squared(prediction, self.future_test)
            self.nrmse = evaluation.nrmse(prediction, self.future_test)
            
        else:
            prediction_sc = self.model.predict(self.history_test_sc)            # scaled prediction
            _, prediction = self.unscaling(self.history_test_sc, prediction_sc) # original prediction
            self.r2 = evaluation.r_squared(prediction, self.future_test)
            self.nrmse = evaluation.nrmse(prediction, self.future_test)
            
        r2_index = pd.DataFrame(['R2']*(len(self.future_var)+1), index=self.future_var+['mean'], columns=['index']).T 
        df_r2 = pd.DataFrame(self.r2, columns=self.future_var)
        mean = pd.DataFrame([df_r2.mean(axis=0)], index=['mean'])
        df_r2 = pd.concat([df_r2, mean], axis=0)
        mean = pd.DataFrame(df_r2.mean(axis=1), columns=['mean'])
        df_r2 = pd.concat([df_r2, mean], axis=1)
        df_r2 = pd.concat([r2_index, df_r2], axis=0)
        
        nrmse_index = pd.DataFrame(['nRMSE']*(len(self.future_var)+1), index=self.future_var+['mean'], columns=['index']).T        
        df_nrmse = pd.DataFrame(self.nrmse, columns=self.future_var)
        mean = pd.DataFrame([df_nrmse.mean(axis=0)], index=['mean'])
        df_nrmse = pd.concat([df_nrmse, mean], axis=0)
        mean = pd.DataFrame(df_nrmse.mean(axis=1), columns=['mean'])
        df_nrmse = pd.concat([df_nrmse, mean], axis=1)
        df_nrmse = pd.concat([nrmse_index, df_nrmse], axis=0)
        
        df_evaluation = pd.concat([df_r2, df_nrmse], axis=1)
        display(df_evaluation)
        
    def get_attention(self, history_series_sc):
        attention_model = Model(inputs=self.model.input,
                                outputs=self.model.get_layer('alignment_score').output)
        return attention_model.predict(history_series_sc)