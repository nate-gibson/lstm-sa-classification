# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:33:27 2020

@author: Nate

Models adapted from: https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
Layers.py from: https://github.com/ajbrookhouse/SelfAttention by Aaron Brookhouse
"""

# Some code is copied from the machinelearningmastery.com article
# (particularly the models) but the vast majority is original work:

import math
from numpy import mean
from numpy import std
from numpy import dstack
from numpy import transpose
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from Layers import SelfAttention
from Layers import AddSinusoidalPositionalEncodings
from Layers import MultiHeadAttention # use for multi-head attention
from timeit import default_timer as timer

# split x into win_size windows
def build_x_windows(x, win_size):
	data_size = x.shape[0]
	num_windows = math.ceil(data_size / win_size)
	windows = list()
	# Loop over each window
	for i in range(0, num_windows):
		start_index = i * win_size
		end_index = min(start_index + win_size, data_size)
		win = x[start_index:end_index]
		# Drop window if it is shorter than win_size
		# (this will happen for the last window if
		# data_size is not divisible by win_size)
		if (win.shape[0] == win_size):
			windows.append(win)
	windows = dstack(windows)
	windows = transpose(windows, (2, 0, 1))
	return windows

# split y into winsize windows,
# classifies each window based on the majority class of all timeslices
def build_y_windows(y, win_size):
	data_size = y.shape[0]
	num_windows = math.ceil(data_size / win_size)
	windows = list()
	# Loop for each window
	for i in range(0, num_windows):
		start_index = i * win_size
		end_index = min(start_index + win_size, data_size)
		win = y[start_index:end_index]
		# Drop last window if too short (same as build_x_windows)
		if (win.shape[0] == win_size):
			num_occupied = 0
			# Loop over y values in window to vote on class
			for j in win:
				if j == 1:
					num_occupied += 1
			if num_occupied >= (win_size / 2):
				windows.append(1)
			else:
				windows.append(0)
	return windows

# get the x and y data from file
def get_data(filename):
	filepath = filename
	df = pd.read_csv(filepath)
	
	lastCol = df.shape[1] - 1
	x = df.iloc[:, 1:lastCol].values
	y = df.iloc[:, lastCol].values
	
	win_size = 25 # the number of datapoints per window (odd # prefered for voting)
	x = build_x_windows(x, win_size)
	y = build_y_windows(y, win_size)
	y = to_categorical(y)
	
	return x, y

# fit and evaluate a model using an LSTM layer
def evaluate_lstm(trainX, trainy, testX, testy):
	#stuff to tune:
	epochs = 5
	
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
	kqlen = 8 # key/query length
	vlen = 8 # value length
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
	trainX, trainY = get_data('datatraining.txt')
	testX, testY = get_data('datatest.txt') # (2 test files to choose from)
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
num_lstm = 15 # number of lstm trials
num_sa = 15 # number of sa trials
run_experiments(num_lstm, num_sa)



