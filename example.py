import numpy as np
import convnet
import audio_processor as ap

def sort_result(tags, preds):
	result = zip(tags, preds)
	return sorted(result, key=lambda x:x[1], reverse=True)

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
	np.save(audio_path.replace('.mp3', '.npy'), melgram)
	melgrams = np.concatenate((melgrams, melgram), axis=0)

# load model like this
model = convnet.build_convnet_model()
model.load_weights('data/weights_best.hdf5')
# predict the tags like this
pred_tags = model.predict(melgrams)
# print like this...
print('Printing top-10 tags for each track...')
for song_idx, audio_path in enumerate(audio_paths):
	sorted_result = sort_result(tags, pred_tags[song_idx,:].tolist())
	print(audio_path)
	print(sorted_result[:10])
	print(' ')


