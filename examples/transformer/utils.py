
import gc
import os
from typing import Union

import dill
import numpy as np

from pygrad.tensor import Tensor


class PrepareDataset:
	def __init__(self):
		self.enc_dataset_vocab:dict 		= {"<EOS>": 0, "<START>": 1}
		self.dec_dataset_vocab:dict 		= {"<EOS>": 0, "<START>": 1}
		self.dataset:Union[None,np.ndarray]	= None
		self.trainX:Union[None,np.ndarray] 	= None
		self.trainY:Union[None,np.ndarray] 	= None
		self.train:Union[None,np.ndarray] 	= None
		self.test:Union[None,np.ndarray] 	= None

	def load_dataset(self, filename, max_len=10000, store:bool=False):
		dataset 		= dill.load(open(filename, 'rb'))[:max_len, :]
		dataset 		= self.add_start_eos(dataset)
		if store: self.dataset 	= dataset
		enc_vocab 		= self._update_enc_vocab(dataset[:, 0])
		dec_vocab 		= self._update_dec_vocab(dataset[:, 1])
		return dataset, enc_vocab, dec_vocab

	def train_test_split(self, dataset, split:float=0.8, shuffle:bool=True, store:bool=False):
		if shuffle: np.random.shuffle(dataset)
		cutoff= int(len(dataset) * split)
		train = dataset[:cutoff]
		test  = dataset[cutoff:]
		if store:
			self.train = train
			self.test  = test
		return train, test

	def create_enc_dec_inputs(self, dataset, store:bool=False):
		trainX   = self.words_to_tokens(dataset[:,0], self.enc_dataset_vocab, pad=True)
		trainY   = self.words_to_tokens(dataset[:,1], self.dec_dataset_vocab, pad=True)
		if store:
			self.trainX = trainX
			self.trainY = trainY
		return trainX, trainY

	def _update_enc_vocab(self, dataset:list):
		counter = max(self.enc_dataset_vocab.values())+1
		for sentence in dataset:
			words = sentence.split()
			for word in words:
				if word not in self.enc_dataset_vocab:
					self.enc_dataset_vocab[word] = counter
					counter += 1
		return self.enc_dataset_vocab

	def _update_dec_vocab(self, dataset:list):
		counter = max(self.dec_dataset_vocab.values())+1
		for sentence in dataset:
			words = sentence.split()
			for word in words:
				if word not in self.dec_dataset_vocab:
					self.dec_dataset_vocab[word] = counter
					counter += 1
		return self.dec_dataset_vocab

	def enc_vocab_size(self):
		return len(self.enc_dataset_vocab)+1
	
	def dec_vocab_size(self):
		return len(self.dec_dataset_vocab)+1

	def find_seq_length(self, dataset):
		return max(len(seq.split()) for seq in dataset)

	def add_start_eos(self, dataset):
		for i in range(dataset[:, 0].size):
			dataset[i, 0] = "<START> " + dataset[i, 0] + " <EOS>"
			dataset[i, 1] = "<START> " + dataset[i, 1] + " <EOS>"
		return dataset

	def words_to_tokens(self, dataset, vocab:dict, pad:bool=True):
		pad_length = self.find_seq_length(dataset)
		tokens = []
		for sentence in dataset:
			tok_sentence = []
			words = sentence.split()
			if pad:
				words = words + ["<EOS>"]*(pad_length-len(words))
			for word in words:
				tok_sentence.append(vocab[word])
			tokens.append(tok_sentence)
		return np.asarray(tokens, dtype=int)

	def tokens_to_words(self, predictions, vocab:dict):
		inverted_vocab = {int(v): k for k, v in vocab.items()}
		all_words = []
		for pred in predictions:
			sentence_preds  = ''
			for tok in pred:
				sentence_preds += inverted_vocab[int(tok)] + ' '
			all_words.append(sentence_preds)
		return all_words

def accuracy_fn(y_pred:np.ndarray, y_true:np.ndarray):
	mask            = (y_true[:,:,0] != 1)[...,None]
	unmasked_acc    = np.argmax(y_pred, axis=-1, keepdims=True) == np.argmax(y_true, axis=-1, keepdims=True)
	return np.sum(unmasked_acc * mask)/np.sum(mask)

def infer_one(model, prepare_dataset):
	model.model_reset()
	encoder_input   = Tensor(np.array(prepare_dataset.trainX[0:0+1,1:]), leaf=True)
	decoder_input   = Tensor(np.array(prepare_dataset.trainY[0:0+1,:-1]), leaf=True)
	decoder_output  = Tensor(np.array(prepare_dataset.trainY[0:0+1,1:]), leaf=True)
	y_pred          = model(enc_inp=encoder_input, dec_inp=decoder_input, training=True)            # calls either new or same model copy

	inp_txt         = prepare_dataset.tokens_to_words(encoder_input.value, prepare_dataset.enc_dataset_vocab)[0]
	inp_txt_d       = prepare_dataset.tokens_to_words(decoder_input.value, prepare_dataset.dec_dataset_vocab)[0]
	pred_txt        = prepare_dataset.tokens_to_words(np.argmax(y_pred.softmax().value, axis=-1), prepare_dataset.dec_dataset_vocab)[0]
	act_txt         = prepare_dataset.tokens_to_words(decoder_output.value, prepare_dataset.dec_dataset_vocab)[0]
	print(f"ine_txt: {inp_txt.split('<EOS>', 1)[0]}<EOS>")
	print(f"ind_txt: {inp_txt_d.split('<EOS>', 1)[0]}")
	print(f"prd_txt: {pred_txt.split('<EOS>', 1)[0]}<EOS>")
	print(f"act_txt: {act_txt.split('<EOS>', 1)[0]}<EOS>")
	model.model_reset()

def save_model(save_dir, model, prepare_dataset):

	# saves the model and dataset
	abs_path = os.path.abspath(save_dir)
	if not os.path.exists(abs_path):
		os.makedirs(abs_path)
	
	# save model
	with open(f'{abs_path}/model.pkl', 'wb') as f:
		model.model_copy = None
		gc.collect()
		dill.dump(model, f)
	
	# save dataset
	if prepare_dataset.dataset is None:
		print("Warning, the PrepareDataset saved with the model has no dataset.")
	with open(f'{abs_path}/dataset.pkl', 'wb') as f:
		dill.dump(prepare_dataset, f)
