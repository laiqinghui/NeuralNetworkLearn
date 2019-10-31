import numpy as np
import pandas
import tensorflow as tf
import csv
import time
import multiprocessing as mp

import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20
BATCH_SIZE = 128
no_epochs = 100 
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)




def char_model(x, model, num_layers, gradient_clip):
	input_layer = tf.one_hot(x, 256)
	char_list = tf.unstack(input_layer, axis=1)

	if num_layers > 1:
		cells = []
		for layer in range(num_layers):
			cells.append(tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE))
		cell = tf.nn.rnn_cell.MultiRNNCell(cells)
		outputs, states = tf.nn.static_rnn(cell, char_list, dtype=tf.float32)
		logits = tf.layers.dense(outputs[-1], MAX_LABEL, activation=None)
	else:
		if model == 'GRU':
			cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
			outputs, states = tf.nn.static_rnn(cell, char_list, dtype=tf.float32)
			logits = tf.layers.dense(outputs[-1], MAX_LABEL, activation=None)
		elif model == 'Vanilla':
			cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
			_, encoding = tf.nn.static_rnn(cell, char_list, dtype=tf.float32)
			logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
		elif model == 'LSTM':
			cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE, use_peepholes=True, forget_bias=1)
			outputs, states = tf.nn.static_rnn(cell, char_list, dtype=tf.float32)
			logits = tf.layers.dense(outputs[-1], MAX_LABEL, activation=None)
	
	return logits, input_layer



def word_model(x, model, num_layers, gradient_clip):
	word_vectors = tf.contrib.layers.embed_sequence(
			x, vocab_size=no_words, embed_dim=EMBEDDING_SIZE)

	word_list = tf.unstack(word_vectors, axis=1)

	if num_layers > 1:
		cells = []
		for layer in range(num_layers):
			cells.append(tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE))
		cell = tf.nn.rnn_cell.MultiRNNCell(cells)
		outputs, states = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)
		logits = tf.layers.dense(outputs[-1], MAX_LABEL, activation=None)

	else:
		if model == 'GRU':
			cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
			_, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)
			logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
		elif model == 'Vanilla':
			cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
			_, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)
			logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
		elif model == 'LSTM':
			cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE, use_peepholes=True, forget_bias=1)
			outputs, states = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)
			logits = tf.layers.dense(outputs[-1], MAX_LABEL, activation=None)
			
	return logits, word_list

def data_read_words():
	
	x_train, y_train, x_test, y_test = [], [], [], []
	
	with open('train_medium.csv', encoding='utf-8') as filex:
		reader = csv.reader(filex)
		for row in reader:
			x_train.append(row[2])
			y_train.append(int(row[0]))

	with open("test_medium.csv", encoding='utf-8') as filex:
		reader = csv.reader(filex)
		for row in reader:
			x_test.append(row[2])
			y_test.append(int(row[0]))
	
	x_train = pandas.Series(x_train)
	y_train = pandas.Series(y_train)
	x_test = pandas.Series(x_test)
	y_test = pandas.Series(y_test)
	y_train = y_train.values
	y_test = y_test.values

	vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
			MAX_DOCUMENT_LENGTH)

	x_transform_train = vocab_processor.fit_transform(x_train)
	x_transform_test = vocab_processor.transform(x_test)


	x_train = np.array(list(x_transform_train))
	x_test = np.array(list(x_transform_test))


	no_words = len(vocab_processor.vocabulary_)

	return x_train, y_train, x_test, y_test, no_words

def data_read_chars():
	
	x_train, y_train, x_test, y_test = [], [], [], []
	
	with open('train_medium.csv', encoding='utf-8') as filex:
		reader = csv.reader(filex)
		for row in reader:
			x_train.append(row[1])
			y_train.append(int(row[0]))

	with open("test_medium.csv", encoding='utf-8') as filex:
		reader = csv.reader(filex)
		for row in reader:
			x_test.append(row[1])
			y_test.append(int(row[0]))
	
	x_train = pandas.Series(x_train)
	y_train = pandas.Series(y_train)
	x_test = pandas.Series(x_test)
	y_test = pandas.Series(y_test)
	y_train = y_train.values
	y_test = y_test.values

	char_processor = tf.contrib.learn.preprocessing.ByteProcessor(
			MAX_DOCUMENT_LENGTH)

	x_transform_train = char_processor.fit_transform(x_train)
	x_transform_test = char_processor.transform(x_test)


	x_train = np.array(list(x_transform_train))
	x_test = np.array(list(x_transform_test))


	return x_train, y_train, x_test, y_test

def main():
	global no_words
	#### Define models ####
	models = {
		# 1 Layer Character RNN Models
		"Character-RNN GRU": ('Char', 'GRU', 1, False),
		"Character-RNN Vanilla RNN": ('Char', 'Vanilla', 1, False),
		"Character-RNN LSTM": ('Char', 'LSTM', 1, False),

		# 1 Layer Word RNN Models
		"Word-RNN GRU": ('Word', 'GRU', 1, False),
		"Word-RNN Vanilla RNN": ('Word', 'Vanilla', 1, False),
		"Word-RNN LSTM": ('Word', 'LSTM', 1, False),

		# 2 Layer Character and Word RNN Models
		"2 Layer GRU Character-RNN": ('Char', 'GRU', 2, False),
		"2 Layer GRU Word-RNN": ('Word', 'GRU', 2, False),

		# # Gradient clipping on the RNN Models
		"Word-RNN GRU Gradient Clip with Threshold 2": ('Word', 'GRU', 1, True),
		# "Character-RNN GRU Gradient Clip with Threshold 2": ('Char', 'GRU', 1, True)

	}

	for _, model in models.items():
		tf.reset_default_graph()
		# Create the model
		x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
		y_ = tf.placeholder(tf.int64)

		# One hot the Y labels
		y_one_hot = tf.one_hot(y_, MAX_LABEL)
		dataType, RNN_Model, num_layers, grad_clip = model
		print("Evaluating current Model - Type: {}, RNN_Model: {}, Layers: {}, GradientClip: {}".format(
			dataType, RNN_Model, num_layers, grad_clip))

		if dataType == 'Char':
			x_train, y_train, x_test, y_test = data_read_chars()
			logits, char_list = char_model(x, RNN_Model, num_layers, grad_clip)
		elif dataType == 'Word':
			x_train, y_train, x_test, y_test, no_words = data_read_words()
			logits, word_list = word_model(x, RNN_Model, num_layers, grad_clip)


		entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_one_hot, logits=logits))


		if (grad_clip == False): # Grad_Clipping is disabled
			train_op = tf.train.AdamOptimizer(lr).minimize(entropy)
		else:
			minimizer = tf.train.AdamOptimizer(lr)
			grads_vars = minimizer.compute_gradients(entropy)

			grad_clipping = tf.constant(2.0, name="grad_clipping")
			grad_clip_vars = []
			for grad, var in grads_vars:
				if (grad is not None):
					clip_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
					grad_clip_vars.append((clip_grad, var))

			train_op = minimizer.apply_gradients(grad_clip_vars)


		# argmax returns the index of the largest value across axes of a tensor
		classification_error = tf.reduce_sum(tf.cast(tf.not_equal(tf.argmax(logits, axis=1), tf.argmax(y_one_hot, axis=1)), tf.int32))
		correct_prediction = tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(y_one_hot, axis=1)), tf.float32) # Cast to float
		accuracy = tf.reduce_mean(correct_prediction)

		N = len(x_train)
		idx = np.arange(N)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			time_to_run = 0
			# training
			train_loss = []
			test_loss = []
			train_acc = []
			test_acc = []
			train_classi_error = []
			test_classi_error = []

			for epoch in range(no_epochs):
				np.random.shuffle(idx)
				x_train, y_train = x_train[idx], y_train[idx]
				t = time.time()
				for start, end in zip(range(0, N, BATCH_SIZE), range(BATCH_SIZE, N, BATCH_SIZE)):
					train_op.run(feed_dict={x: x_train[start:end], y_: y_train[start:end]})
				time_to_run += time.time() - t

				# Entropy Cost on Training and Test Data
				train_loss.append(entropy.eval(feed_dict={x: x_train, y_: y_train})) 
				test_loss.append(entropy.eval(feed_dict={x: x_test, y_: y_test}))

				# Accuracy on Training and Test Data
				train_acc.append(accuracy.eval(feed_dict={x: x_train, y_:y_train})) # Accuracy on Training Data
				test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test})) # Accuracy on Testing Data

				# Classification Errors on Training and Test Data
				train_classi_error.append(classification_error.eval(feed_dict={x: x_train, y_: y_train}))
				test_classi_error.append(classification_error.eval(feed_dict={x: x_test, y_: y_test}))


				if epoch%1 == 0:
					print('Epochs: {}, Training Loss: {}'.format(epoch, train_loss[epoch]))

			print("Total Time to Train the Network is: {}".format(time_to_run)) 
			####### Comparison across the different RNN Architecture #######
			if dataType == "Char" and num_layers == 1 and grad_clip == False:
				plt.figure(1)
				plt.plot(range(no_epochs), train_loss, label="Training Loss Character RNN - " + RNN_Model)

				plt.figure(2)
				plt.plot(range(no_epochs), test_loss, label="Testing Loss Character RNN - " + RNN_Model)

				plt.figure(3)
				plt.plot(range(no_epochs), train_acc, label="Training Accuracy Character RNN - " + RNN_Model)

				plt.figure(4)
				plt.plot(range(no_epochs), test_acc, label="Testing Accuracy Character RNN - " + RNN_Model)

				plt.figure(5)
				plt.plot(range(no_epochs), train_classi_error, label="Training Errors Character RNN - " + RNN_Model)

				plt.figure(6)
				plt.plot(range(no_epochs), test_classi_error, label="Testing Errors Character RNN - " + RNN_Model)


			elif dataType == "Word" and num_layers == 1 and grad_clip == False:
				plt.figure(7)
				plt.plot(range(no_epochs), train_loss, label="Training Loss Word RNN - " + RNN_Model)

				plt.figure(8)
				plt.plot(range(no_epochs), test_loss, label="Testing Loss Word RNN - " + RNN_Model)

				plt.figure(9)
				plt.plot(range(no_epochs), train_acc, label="Training Accuracy Word RNN - " + RNN_Model)

				plt.figure(10)
				plt.plot(range(no_epochs), test_acc, label="Testing Accuracy Word RNN - " + RNN_Model)

				plt.figure(11)
				plt.plot(range(no_epochs), train_classi_error, label="Training Errors Word RNN - " + RNN_Model)

				plt.figure(12)
				plt.plot(range(no_epochs), test_classi_error, label="Testing Errors Word RNN - " + RNN_Model)

			##### Number of Layers == 2 #########
			elif dataType == "Char" and num_layers == 2 and grad_clip == False:
				plt.figure(13)
				plt.plot(range(no_epochs), train_loss)
				plt.xlabel('Epochs')
				plt.ylabel('Training Loss')
				plt.title('2-Layer Char RNN Training Cross Entropy Loss vs Epochs')
				plt.savefig('./TrainingLossAcross2LayerCharRNNModels')

				plt.figure(14)
				plt.plot(range(no_epochs), test_loss)
				plt.xlabel('Epochs')
				plt.ylabel('Testing Loss')
				plt.title('2-Layer Char RNN Testing Cross Entropy Loss vs Epochs')
				plt.savefig('./TestingLossAcross2LayerCharRNNModels')

				plt.figure(15)
				plt.plot(range(no_epochs), train_acc)
				plt.xlabel('Epochs')
				plt.ylabel('Training Accuracy')
				plt.title('2-Layer Char RNN Training Accuracy vs Epochs')
				plt.savefig('./TrainingAccuracyAcross2LayerCharRNNModels')

				plt.figure(16)
				plt.plot(range(no_epochs), test_acc)
				plt.xlabel('Epochs')
				plt.ylabel('Testing Accuracy')
				plt.title('2-Layer Char RNN Testing Accuracy vs Epochs')
				plt.savefig('./TestingAccuracyAcross2LayerCharRNNModels')

				plt.figure(17)
				plt.plot(range(no_epochs), train_classi_error)
				plt.xlabel('Epochs')
				plt.ylabel('Training Errors')
				plt.title('2-Layer Char RNN Training Errors vs Epochs')
				plt.savefig('./TrainingErrorsAcross2LayerCharRNNModels')

				plt.figure(18)
				plt.plot(range(no_epochs), test_classi_error)
				plt.xlabel('Epochs')
				plt.ylabel('Testing Errors')
				plt.title('2-Layer Char RNN Testing Errors vs Epochs')
				plt.savefig('./TestingErrorsAcross2LayerCharRNNModels')

			elif dataType == "Word" and num_layers == 2 and grad_clip == False:
				plt.figure(19)
				plt.plot(range(no_epochs), train_loss)
				plt.xlabel('Epochs')
				plt.ylabel('Training Loss')
				plt.title('2-Layer Word RNN Training Cross Entropy Loss vs Epochs')
				plt.savefig('./TrainingLossAcross2LayerWordRNNModels')

				plt.figure(20)
				plt.plot(range(no_epochs), test_loss)
				plt.xlabel('Epochs')
				plt.ylabel('Testing Loss')
				plt.title('2-Layer Word RNN Testing Cross Entropy Loss vs Epochs')
				plt.savefig('./TestingLossAcross2LayerWordRNNModels')

				plt.figure(21)
				plt.plot(range(no_epochs), train_acc)
				plt.xlabel('Epochs')
				plt.ylabel('Training Accuracy')
				plt.title('2-Layer Word RNN Training Accuracy vs Epochs')
				plt.savefig('./TrainingAccuracyAcross2LayerWordRNNModels')

				plt.figure(22)
				plt.plot(range(no_epochs), test_acc)
				plt.xlabel('Epochs')
				plt.ylabel('Testing Accuracy')
				plt.title('2-Layer Word RNN Testing Accuracy vs Epochs')
				plt.savefig('./TestingAccuracyAcross2LayerWordRNNModels')

				plt.figure(23)
				plt.plot(range(no_epochs), train_classi_error)
				plt.xlabel('Epochs')
				plt.ylabel('Training Errors')
				plt.title('2-Layer Word RNN Training Errors vs Epochs')
				plt.savefig('./TrainingErrorsAcross2LayerWordRNNModels')

				plt.figure(24)
				plt.plot(range(no_epochs), test_classi_error)
				plt.xlabel('Epochs')
				plt.ylabel('Testing Errors')
				plt.title('2-Layer Word RNN Testing Errors vs Epochs')
				plt.savefig('./TestingErrorsAcross2LayerWordRNNModels')


			# Gradient Clipping occurs here!
			elif dataType == "Char" and num_layers == 1 and grad_clip == True:
				plt.figure(25)
				plt.plot(range(no_epochs), train_loss)
				plt.xlabel('Epochs')
				plt.ylabel('Training Loss')
				plt.title('Gradient-Clipping Char RNN Training Cross Entropy Loss vs Epochs')
				plt.savefig('./TrainingLossGradientClipCharRNNModels')

				plt.figure(26)
				plt.plot(range(no_epochs), test_loss)
				plt.xlabel('Epochs')
				plt.ylabel('Testing Loss')
				plt.title('Gradient-Clipping Char RNN Testing Cross Entropy Loss vs Epochs')
				plt.savefig('./TestingLossGradientClipCharRNNModels')

				plt.figure(27)
				plt.plot(range(no_epochs), train_acc)
				plt.xlabel('Epochs')
				plt.ylabel('Training Accuracy')
				plt.title('Gradient-Clipping Char RNN Training Accuracy vs Epochs')
				plt.savefig('./TrainingAccuracyGradientClipCharRNNModels')

				plt.figure(28)
				plt.plot(range(no_epochs), test_acc)
				plt.xlabel('Epochs')
				plt.ylabel('Testing Accuracy')
				plt.title('Gradient-Clipping Char RNN Testing Accuracy vs Epochs')
				plt.savefig('./TestingAccuracyGradientClipCharRNNModels')

				plt.figure(29)
				plt.plot(range(no_epochs), train_classi_error)
				plt.xlabel('Epochs')
				plt.ylabel('Training Errors')
				plt.title('Gradient-Clipping Char RNN Training Errors vs Epochs')
				plt.savefig('./TrainingErrorsGradientClipCharRNNModels')

				plt.figure(30)
				plt.plot(range(no_epochs), test_classi_error)
				plt.xlabel('Epochs')
				plt.ylabel('Testing Errors')
				plt.title('Gradient-Clipping Char RNN Testing Errors vs Epochs')
				plt.savefig('./TestingErrorsGradientClipCharRNNModels')


			elif dataType == "Word" and num_layers == 1 and grad_clip == True:
				plt.figure(25)
				plt.plot(range(no_epochs), train_loss)
				plt.xlabel('Epochs')
				plt.ylabel('Training Loss')
				plt.title('Gradient-Clipping Word RNN Training Cross Entropy Loss vs Epochs')
				plt.savefig('./TrainingLossGradientClipWordRNNModels')

				plt.figure(26)
				plt.plot(range(no_epochs), test_loss)
				plt.xlabel('Epochs')
				plt.ylabel('Testing Loss')
				plt.title('Gradient-Clipping Word RNN Testing Cross Entropy Loss vs Epochs')
				plt.savefig('./TestingLossGradientClipWordRNNModels')

				plt.figure(27)
				plt.plot(range(no_epochs), train_acc)
				plt.xlabel('Epochs')
				plt.ylabel('Training Accuracy')
				plt.title('Gradient-Clipping Word RNN Training Accuracy vs Epochs')
				plt.savefig('./TrainingAccuracyGradientClipWordRNNModels')

				plt.figure(28)
				plt.plot(range(no_epochs), test_acc)
				plt.xlabel('Epochs')
				plt.ylabel('Testing Accuracy')
				plt.title('Gradient-Clipping Word RNN Testing Accuracy vs Epochs')
				plt.savefig('./TestingAccuracyGradientClipWordRNNModels')

				plt.figure(29)
				plt.plot(range(no_epochs), train_classi_error)
				plt.xlabel('Epochs')
				plt.ylabel('Training Errors')
				plt.title('Gradient-Clipping Word RNN Training Errors vs Epochs')
				plt.savefig('./TrainingErrorsGradientClipWordRNNModels')

				plt.figure(30)
				plt.plot(range(no_epochs), test_classi_error)
				plt.xlabel('Epochs')
				plt.ylabel('Testing Errors')
				plt.title('Gradient-Clipping Word RNN Testing Errors vs Epochs')
				plt.savefig('./TestingErrorsGradientClipWordRNNModels')



	plt.figure(1)
	plt.xlabel('Epochs')
	plt.ylabel('Training Loss')
	plt.legend(title='Layers')
	plt.title('Training Cross Entropy Loss vs Epochs')
	plt.savefig('./TrainingLossAcrossCharRNNModels')

	plt.figure(2)
	plt.xlabel('Epochs')
	plt.ylabel('Testing Loss')
	plt.legend(title='Layers')
	plt.title('Testing Cross Entropy Loss vs Epochs')
	plt.savefig('./TestingLossAcrossCharRNNModels')

	plt.figure(3)
	plt.xlabel('Epochs')
	plt.ylabel('Training Accuracy')
	plt.legend(title='Layers')
	plt.title('Training Accuracy vs Epochs')
	plt.savefig('./TrainingAccuracyAcrossCharRNNModels')

	plt.figure(4)
	plt.xlabel('Epochs')
	plt.ylabel('Testing Accuracy')
	plt.legend(title='Layers')
	plt.title('Testing Accuracy vs Epochs')
	plt.savefig('./TestingAccuracyAcrossCharRNNModels')

	plt.figure(5)
	plt.xlabel('Epochs')
	plt.ylabel('Training Errors')
	plt.legend(title='Layers')
	plt.title('Training Errors vs Epochs')
	plt.savefig('./TrainingErrorsAcrossCharRNNModels')

	plt.figure(6)
	plt.xlabel('Epochs')
	plt.ylabel('Testing Errors')
	plt.legend(title='Layers')
	plt.title('Testing Errors vs Epochs')
	plt.savefig('./TestingErrorsAcrossWordRNNModels')


	plt.figure(7)
	plt.xlabel('Epochs')
	plt.ylabel('Training Loss')
	plt.legend(title='Layers')
	plt.title('Training Cross Entropy Loss vs Epochs')
	plt.savefig('./TrainingLossAcrossWordRNNModels')

	plt.figure(8)
	plt.xlabel('Epochs')
	plt.ylabel('Testing Loss')
	plt.legend(title='Layers')
	plt.title('Testing Cross Entropy Loss vs Epochs')
	plt.savefig('./TestingLossAcrossWordRNNModels')

	plt.figure(9)
	plt.xlabel('Epochs')
	plt.ylabel('Training Accuracy')
	plt.legend(title='Layers')
	plt.title('Training Accuracy vs Epochs')
	plt.savefig('./TrainingAccuracyAcrossWordRNNModels')

	plt.figure(10)
	plt.xlabel('Epochs')
	plt.ylabel('Testing Accuracy')
	plt.legend(title='Layers')
	plt.title('Testing Accuracy vs Epochs')
	plt.savefig('./TestingAccuracyAcrossWordRNNModels')

	plt.figure(11)
	plt.xlabel('Epochs')
	plt.ylabel('Training Errors')
	plt.legend(title='Layers')
	plt.title('Training Errors vs Epochs')
	plt.savefig('./TrainingErrorsAcrossWordRNNModels')

	plt.figure(12)
	plt.xlabel('Epochs')
	plt.ylabel('Testing Errors')
	plt.legend(title='Layers')
	plt.title('Testing Errors vs Epochs')
	plt.savefig('./TestingErrorsAcrossWordRNNModels')




if __name__ == '__main__':
	main()