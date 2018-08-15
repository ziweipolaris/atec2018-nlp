import gensim
import os
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim

# Set file names for train and test data
data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
lee_train_file = data_dir + 'lee_background.cor'
lee_data = LineSentence(lee_train_file)

model_gensim = FT_gensim(size=300)

# build the vocabulary
model_gensim.build_vocab(lee_data)

# train the model
model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.iter)

print(model_gensim)

model_gensim.save('pai_model/word2vec_fastcbow300')
