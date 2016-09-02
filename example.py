import numpy as np
import convnet, recurrentnet
import audio_processor as ap
import pdb

def sort_result(tags, preds):
	result = zip(tags, preds)
	sorted_result = sorted(result, key=lambda x:x[1], reverse=True)
	return [(name, '%5.3f'%score) for name, score in sorted_result]

def main(net):
	print('Running main() with network: %s' % net)
	# setting
	audio_paths = ['data/bensound-cute.mp3', 'data/bensound-actionable.mp3', 'data/bensound-dubstep.mp3'
	, 'data/bensound-thejazzpiano.mp3']
	tags = ['rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists', 
			'dance', '00s', 'alternative rock', 'jazz', 'beautiful', 'metal', 
			'chillout', 'male vocalists', 'classic rock', 'soul', 'indie rock',
			'Mellow', 'electronica', '80s', 'folk', '90s', 'chill', 'instrumental',
			'punk', 'oldies', 'blues', 'hard rock', 'ambient', 'acoustic', 'experimental',
			'female vocalist', 'guitar', 'Hip-Hop', '70s', 'party', 'country', 'easy listening',
			'sexy', 'catchy', 'funk', 'electro' ,'heavy metal', 'Progressive rock',
			'60s', 'rnb', 'indie pop', 'sad', 'House', 'happy']

	# prepare data like this
	melgrams = np.zeros((0, 1, 96, 1366))
	for audio_path in audio_paths:
		melgram = ap.compute_melgram(audio_path)
		melgrams = np.concatenate((melgrams, melgram), axis=0)

	# load model like this
	if net == 'cnn':
		model = convnet.build_convnet_model()
	elif net == 'rnn':
		model = recurrentnet.build_recurrentnet_model()

	model.summary()
	print('Loading weights of %s...' % net)
	model.load_weights('data/%s_weights_best.hdf5' % net)
	pdb.set_trace()
	# predict the tags like this
	print('Predicting...')
	pred_tags = model.predict(melgrams)
	# print like this...
	print('Printing top-15 tags for each track...')
	for song_idx, audio_path in enumerate(audio_paths):
		sorted_result = sort_result(tags, pred_tags[song_idx,:].tolist())
		print(audio_path)
		print(sorted_result[:5])
		print(sorted_result[5:10])
		print(sorted_result[10:15])
		print(' ')
	return

if __name__=='__main__':

	networks = ['rnn', 'cnn']
	for net in networks:
		main(net)



