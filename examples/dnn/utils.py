
import os

import dill
import numpy as np


def accuracy_fn(y_pred:np.ndarray, y_true:np.ndarray):
	return np.mean(np.argmax(y_pred, axis=-1, keepdims=True) == np.argmax(y_true, axis=-1, keepdims=True))


def save_model(save_dir, model):

	# saves the model and dataset
	abs_path = os.path.abspath(save_dir)
	if not os.path.exists(abs_path):
		os.makedirs(abs_path)
	
	# save model
	with open(f'{abs_path}/model.pkl', 'wb') as f:
		dill.dump(model, f)
