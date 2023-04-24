# The next word predictor
Inspired by the iPhone word predictor I am trying to recreate such a language model.
<img src="images/iphone-texting-support.jpg" alt=inspiration width="50%" height="50%">

## Dataset
Both the Word2vec and the feed forward neural network are trained on the famous [dataset](http://ai.stanford.edu/~amaas/data/sentiment/) of movie reviews.

## Architecture
<img src="images/pipeline.png" alt=pipeline width="100%" height="100%">

### Word embedding
The Word2vec embedding was trained on the whole corpus, then applied when needed on the words. The embedding was evaluated by checking the most similar words and it was giving satisfying results, so I stuck with this vectorizer. However, using a different approach may improve performance. 
The biggest drawback is the fact that if a word didn't previously occur in the vocabulary the model cannot provide a prediction. It would make sense to apply a contextual embedding such as [BERT](https://arxiv.org/pdf/1810.04805.pdf).

### Neural network
As the first approach is with a dense neural network (more accurate would be RNN, I will add LSTM in the future). As the input it takes a hardcoded number (`PREVIOUS_WORDS_CONSIDERED` in [config.py](https://github.com/janfiszer/next-word-predictor/blob/main/config.py)) of vectorized words and as the output the is a layer with one neuron for each word in the vocabulary. Then after applying the softmax activation function we get a probability distribution for each word. 

## Results
Already some satisfying results are at the end of [model-evaluation.ipynb](https://github.com/janfiszer/next-word-predictor/blob/main/model-evaluation.ipynb).

Example:
```
actually more important...
PREDICTED:
                      films: 44%
                        but: 51%
                         to: 53%
                        and: 75%
                       than: 90%
``` 

### TODO:
- [ ] RNN instead of FNN
- [ ] Different word embedding (BERT) 
- [ ] More detailed architecture instead of pipeline    
- [ ] DataGenerator improvement allowing to load documents to memory by batches


More coming soon...