import tensorflow as tf
import os
from sklearn.model_selection import train_test_split # por algun motivo me toma una version vieja de sklearn
import numpy as np
import csv

#parametros usados para entrenar la red
learning_rate = 0.2 # tasa de aprendizaje
num_steps = 1000 # cantidad de pasos de entrenamiento
batch_size = 256 # cantidad de ejemplos por paso
display_step = 100 # cada cuanto imprime algo por pantalla
n_hidden_1 = 128 # numero de neuronas en la capa oculta 1
n_hidden_2 = 128 # numero de neuronas en la capa oculta 2
n_hidden_3 = 128 # numero de neuronas en la capa oculta 3
n_hidden_4 = 128 # numero de neuronas en la capa oculta 3
num_input = 7
num_classes = 3

# Definimos la red neuronal
def neural_net (x_dict):
	x = x_dict['players'] 
	layer_1 = tf.layers.dense(x, n_hidden_1)
	layer_2 = tf.layers.dense(layer_1, n_hidden_2)
	layer_3 = tf.layers.dense(layer_2, n_hidden_3)
	layer_4 = tf.layers.dense(layer_3, n_hidden_4)
	out_layer = tf.layers.dense(layer_4, num_classes)
	return out_layer

def model_fn (features, labels, mode):
	logits = neural_net(features)
	# Predicciones
	pred_classes = tf.argmax(logits, axis=1)
	pred_probas = tf.nn.softmax(logits)

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

	# Definimos nuestro error
	loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
	logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
	
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op,
	global_step=tf.train.get_global_step())
	
	acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
	
	estim_specs = tf.estimator.EstimatorSpec(
	mode=mode,
	predictions=pred_classes,
	loss=loss_op,
	train_op=train_op,
	eval_metric_ops={'accuracy': acc_op})
	return estim_specs

def processCsv(input_file, train = True):
	dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'
	filename = dir_path + input_file
	# Old values for sporQ csv
	# valid_columns = ['DP Normalizado', 'Tm # Normalizado', 'Pos # Normalizado', 'Age Normalizado', 'AvgAV Categorizado', 'Conference # Norm', 'NFL.com Grade', 'SPORQ Normalizado']
	valid_columns = ['DP Normalizado', 'Tm # Normalizado', 'Pos # Normalizado', 'Age Normalizado', 'AltAAV Cat', 'C# Norm', 'Grade', 'SPORQ Normalizado']
	input_matrix = []
	if(train):
		input_prediction = []
	with open(filename) as inf:
		records = csv.DictReader(inf)
		for row in records:
			line_values = []
			for column in valid_columns:
				# Old value for sporQ csv
				# if column == 'AvgAV Categorizado':
				if column == 'AltAAV Cat':
					avg = int(row[column])
				else:
					line_values.append(float(row[column]))
			input_matrix.append(np.array(line_values))
			if(train):
				input_prediction.append(avg)
	if(train):
		return input_matrix, input_prediction
	return input_matrix

train_1, train_2 = processCsv('sin_2018_clean.csv', True)
trainX, testX, trainY, testY = train_test_split(train_1, train_2, test_size=0.33, random_state=42)

model = tf.estimator.Estimator(model_fn)

x_dict = {'players': np.array(trainX)}
y_value = np.array(trainY)
input_fn = tf.estimator.inputs.numpy_input_fn(
x=x_dict, y=y_value,
batch_size=batch_size, num_epochs=None, shuffle=True)
#Entrenamos el modelo
model.train(input_fn, steps=num_steps)

# Definimos la entrada para evaluar
x_dict = {'players': np.array(testX)}
y_value = np.array(testY)
input_fn = tf.estimator.inputs.numpy_input_fn(
x=x_dict, y=y_value,
batch_size=batch_size, shuffle=False)
e = model.evaluate(input_fn)

print("Precision en el conjunto de prueba:", e['accuracy'])

testX = processCsv('con_2018_clean.csv', False)

x_dict = {'players': np.array(testX)}
input_fn = tf.estimator.inputs.numpy_input_fn(
x=x_dict, num_epochs=1, shuffle=False)

initialP = 0
predictions = model.predict(input_fn)
for prediction in predictions:
	print(str(initialP) + ': {}'.format(prediction))
	initialP += 1