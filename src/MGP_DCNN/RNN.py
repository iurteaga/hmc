from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Input, Dropout, LSTM, TimeDistributed, Reshape, UpSampling1D
import numpy

def RNN_time_prediction(x_length, y_length , nb_sensors, dropout_ratio = 0):

    input = Input(shape=(x_length, nb_sensors))
    ''' Here we apply Dropout even if we are in the TEST phase, because training is always set to TRUE'''
    x = Dropout(dropout_ratio)(input,training=True)
    x = LSTM(10, return_sequences = True)(x)
    x = TimeDistributed(Dense(nb_sensors))(x)
    x = Reshape((y_length,nb_sensors))(x)
    model = Model(input, x, name='time_rnn')
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model
    
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
