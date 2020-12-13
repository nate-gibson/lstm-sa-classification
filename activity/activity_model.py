# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:14:47 2020

@author: Nate

Data from: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
Adapted from: https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
Layers.py from: https://github.com/ajbrookhouse/SelfAttention by Aaron Brookhouse
"""

from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from Layers import SelfAttention
from Layers import AddSinusoidalPositionalEncodings
from Layers import MultiHeadAttention
from timeit import default_timer as timer   

#%%
# All Code in this section is unchanged from machinelearningmastery article:

# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'UCI HAR Dataset/')
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'UCI HAR Dataset/')
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

#%%
# Code in this section is modified from article or original work:

# fit and evaluate a model using an LSTM layer
def evaluate_lstm(trainX, trainy, testX, testy):
	#stuff to tune:
	epochs = 15
	
	verbose, batch_size = 1, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps,n_features))) # LSTM layer
	
	# Same layers on both models:
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	# fit and evaluate model
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# fit and evaluate a model using an SPE and Self-Attention layer
def evaluate_attn(trainX, trainY, testX, testY):
	#stuff to tune:
	kqlen = 100 # key/query length
	vlen = 100 # value length
	epochs = 4
	
	verbose, batch_size = 1, 32
	n_outputs = trainY.shape[1]
	
	model = Sequential()
	# Add Sinusoidal Positional Encoding Layer:
	model.add(AddSinusoidalPositionalEncodings())
	# Add Self-Attention Layer, (can use sa or mh-sa):
	model.add(SelfAttention(kqlen,vlen,return_sequence=False, dropout=.5))
	
	# Same layers on both models:
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	# fit and evaluate model
	model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
	_, accuracy = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
	return accuracy

# run an experiment for a certain number of trials
def run_experiment(trials, method, trainX, trainY, testX, testY):
	scores = list()
	times = list()
	for t in range(trials):
		start = timer()
		if method == 'lstm':
			print(f"Evaluating LSTM Model ({t+1}/{trials})")
			score = evaluate_lstm(trainX, trainY, testX, testY)
		elif method == 'sa':
			print(f"Evaluating Self-Attention Model ({t+1}/{trials})")
			score = evaluate_attn(trainX, trainY, testX, testY)
		time = timer() - start
		times.append(time)
		score = score * 100.0
		print('>#%d: %.3f' % (t+1, score))
		print('Time: %.3fs' % (time))
		scores.append(score)
	# return results
	print("\n")
	return scores, times

# run lstm experiments and sa experiments
def run_experiments(num_lstm, num_sa):
	# load training and testing data
	trainX, trainY, testX, testY = load_dataset()
	print("Data shape:", trainX.shape, trainY.shape, testX.shape, testY.shape, "\n\n")
	
	# run trials
	if num_lstm > 0:
		print(f"RUNNING {num_lstm} LSTM TRIAL(S):")
		lstm_scores, lstm_times = run_experiment(num_lstm, 'lstm', trainX, trainY, testX, testY)
	if num_sa > 0:
		print(f"RUNNING {num_sa} SELF-ATTENTION TRIAL(S)")
		sa_scores, sa_times = run_experiment(num_sa, 'sa', trainX, trainY, testX, testY)
	
	# print results
	if num_lstm > 0:
		print('LSTM Results:')
		summarize_results(lstm_scores, lstm_times)
		print('Trials:', num_lstm, "\n")
	if num_sa > 0:
		print('Self-Attention Results:')
		summarize_results(sa_scores, sa_times)
		print('Trials:', num_sa, "\n")

# summarize scores and times
def summarize_results(scores, times):
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
	m, s = mean(times), std(times)
	print('Avg Time: %.3fs (+/-%.3f)' % (m, s))


# experiments:
num_lstm = 10 # number of lstm trials
num_sa = 10 # number of sa trials
run_experiments(num_lstm, num_sa)
