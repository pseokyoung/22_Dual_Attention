from pickle import TRUE
from tensorflow import keras
from tensorflow.keras.layers import *
from keras.models import Model

def seq2seqLSTM(history_size, history_num, future_size, future_num,
                num_layers=10, num_neurons=50, dense_layers=1, dense_neurons=50): # sequence-to-sequence LSTM

    encoder_input = Input(shape=(history_size, history_num))
    
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
                
    decoder_input = RepeatVector(future_size)(encoder_output)
    
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
    if dense_layers == 1:
        decoder_output = Dense(future_num, name='dense')(decoder_output)
    else:
        for num in range(dense_layers):
            if not num:
                decoder_output = Dense(dense_neurons, name='dense_1')(decoder_output)
            elif num == dense_layers-1:
                decoder_output = Dense(future_num, name=f'dense_{num+1}')(decoder_output)
            else:
                decoder_output = Dense(dense_neurons, name=f'dense_{num+1}')(decoder_output)      
    
    # 모델 구성
    model = Model(encoder_input, decoder_output)
    optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mse', optimizer = optimizer)
    return model

def seq2seqGRU(history_size, history_num, future_size, future_num,
               num_layers=10, num_neurons=50, dense_layers=1, dense_neurons=50): # sequence-to-sequence GRU
    
    encoder_input = Input(shape=(history_size, history_num))
    
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
    decoder_input = RepeatVector(future_size)(encoder_output)
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
    if dense_layers == 1:
        decoder_output = Dense(future_num, name='dense')(decoder_output)
    else:
        for num in range(dense_layers):
            if not num:
                decoder_output = Dense(dense_neurons, name='dense_1')(decoder_output)
            elif num == dense_layers-1:
                decoder_output = Dense(future_num, name=f'dense_{num+1}')(decoder_output)
            else:
                decoder_output = Dense(dense_neurons, name=f'dense_{num+1}')(decoder_output)     
    
    # 모델 구성
    model = Model(encoder_input, decoder_output)
    optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mse', optimizer = optimizer)
    return model

def ATTseq2seqLSTM(history_size, history_num, future_size, future_num,
                   num_layers=10, num_neurons=50, dense_layers=1, dense_neurons=50): # attention-based sequence-to-sequence LSTM
    
    encoder_input = Input(shape=(history_size, history_num))
    
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
    decoder_input = RepeatVector(future_size)(state_h)
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
    if dense_layers == 1:
        decoder_output = Dense(future_num, name='dense')(decoder_conbined_context)
    else:
        for num in range(dense_layers):
            if not num:
                decoder_output = Dense(dense_neurons, name='dense_1')(decoder_conbined_context)
            elif num == dense_layers-1:
                decoder_output = Dense(future_num, name=f'dense_{num+1}')(decoder_output)
            else:
                decoder_output = Dense(dense_neurons, name=f'dense_{num+1}')(decoder_output)      
        
    # 모델 구성
    model = Model(encoder_input, decoder_output)
    optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mse', optimizer = optimizer)
    
    return model

def ATTseq2seqGRU(history_size, history_num, future_size, future_num,
                  num_layers=10, num_neurons=50, dense_layers=1, dense_neurons=50): # attention-based sequence-to-sequence GRU
    
    encoder_input = Input(shape=(history_size, history_num))
    
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
                
    decoder_input = RepeatVector(future_size)(state_h)
    
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
    if dense_layers == 1:
        decoder_output = Dense(future_num, name='dense')(decoder_conbined_context)
    else:
        for num in range(dense_layers):
            if not num:
                decoder_output = Dense(dense_neurons, name='dense_1')(decoder_conbined_context)
            elif num == dense_layers-1:
                decoder_output = Dense(future_num, name=f'dense_{num+1}')(decoder_output)
            else:
                decoder_output = Dense(dense_neurons, name=f'dense_{num+1}')(decoder_output)        
    
    # 모델 구성
    model = Model(encoder_input, decoder_output)
    optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mse', optimizer = optimizer)
    
    return model

def DATTseq2seqLSTM(history_size, history_num, future_size, future_num,
                    factor, num_layers=10, num_neurons=50, dense_layers=1, dense_neurons=50): # dual attention-based sequence-to-sequence LSTM
    
    encoder_input = Input(shape=(history_size, history_num))
    
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
    decoder_input = RepeatVector(future_size)(state_h)
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
    if dense_layers == 1:
        decoder_output = Dense(future_num, name='dense')(decoder_conbined_context)
    else:
        for num in range(dense_layers):
            if not num:
                decoder_output = Dense(dense_neurons, name='dense_1')(decoder_conbined_context)
            elif num == dense_layers-1:
                decoder_output = Dense(future_num, name=f'dense_{num+1}')(decoder_output)
            else:
                decoder_output = Dense(dense_neurons, name=f'dense_{num+1}')(decoder_output)                    
    
    # 가중 어텐션 층
    decoder_output = Multiply(name='multiply')([decoder_output, factor])
    
    # 모델 구성
    model = Model(encoder_input, decoder_output)
    optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mse', optimizer = optimizer)
    
    return model

def DATTseq2seqGRU(history_size, history_num, future_size, future_num,
                   factor, num_layers=10, num_neurons=50, dense_layers=1, dense_neurons=50): # dual attention-based sequence-to-sequence GRU
    
    encoder_input = Input(shape=(history_size, history_num))
    
    # 인코더 층
    if num_layers == 1:
        encoder = GRU(num_neurons,  return_sequences=True, return_state=True, name='encoder')
        encoder_output, state_h = encoder(encoder_input)
    else:
        for layer in range(num_layers):
            if not layer: # 첫번째 인코더 층
                encoder_output = GRU(num_neurons,  return_sequences=True, return_state=False, name='encoder_1')(encoder_input)
            elif layer == num_layers-1: # 마지막 인코더 층
                encoder_output, state_h  = GRU(num_neurons, return_sequences=True, return_state=True, name=f'encoder_{layer+1}')(encoder_output)
            else: # 중간 인코더 층
                encoder_output = GRU(num_neurons, return_sequences=True, return_state=False, name=f'encoder_{layer+1}')(encoder_output)
                
    decoder_input = RepeatVector(future_size)(state_h)
    
    # 디코더 층
    if num_layers == 1:
        decoder = GRU(num_neurons, return_sequences=True, name='decoder')
        decoder_output = decoder(decoder_input, initial_state=state_h)
    else:
        for layer in range(num_layers):
            if not layer: # 첫번째 디코더 층
                decoder_output = GRU(num_neurons, return_sequences=True, return_state=False, name='decoder_1')(decoder_input, initial_state=state_h)
            else: # 디코더 층
                decoder_output  = GRU(num_neurons, return_sequences=True, return_state=False, name=f'decoder_{layer+1}')(decoder_output)
    
    # 어텐션 층 
    attention = dot([decoder_output, encoder_output], axes=[2,2])
    attention = Activation('softmax', name='alignment_score')(attention)
    context = dot([attention, encoder_output],axes=[2,1])
    decoder_conbined_context = concatenate([context, decoder_output])            
    
    # 덴스 층
    if dense_layers == 1:
        decoder_output = Dense(future_num, name='dense')(decoder_conbined_context)
    else:
        for num in range(dense_layers):
            if not num:
                decoder_output = Dense(dense_neurons, name='dense_1')(decoder_conbined_context)
            elif num == dense_layers-1:
                decoder_output = Dense(future_num, name=f'dense_{num+1}')(decoder_output)
            else:
                decoder_output = Dense(dense_neurons, name=f'dense_{num+1}')(decoder_output)                    
    
    # 가중 어텐션 층
    decoder_output = Multiply(name='multiply')([decoder_output, factor])
    
    # 모델 구성
    model = Model(encoder_input, decoder_output)
    optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mse', optimizer = optimizer)
    
    return model