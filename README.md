# Multilingual Named Entity Recognition
##### Tony DiPadova and Steven Jiang

## Goals
Mapping companies, such as TomTom, OpenStreetMap, and MapBox, face the difficult issue of how to automatically extract pertinent information from news and social media to update their maps. Tweets about new restaurants or Facebook postings about road construction complaints introduce novel information relevant to map updates. Named Entity Recognition (NER) is a key part of processing this information. For example, when processing a tweet like “Must try the new Tony’s Pizzeria on Main Street!”, it is important to recognize the name of the organization (Tony’s Pizzeria) and the location (Main Street). Furthermore, companies such as Google, Apple, and TomTom, which operate on a global scale, need to be able to process social media data in any language. We propose a Multilingual Named Entity Recognition (MNER) using Facebook MUSE word embeddings as features to a bi-directional Long Short Term Memory (bi-LSTM) network combined with a classifier, such as a Conditional Random Field (CRF) or a Random Forest. Using the CoNLL shared task corpora as our training data, we plan to train a bi-LSTM to extract features from sentences and implement a classifier to tag words with a Named Entity Inner-Outer-Beginning (IOB) tag.

## Dataset
* CoNLL 2003 Corpus - Includes Parts-of-Speech and Named Entities
	* The Conference on Computational Natural Language Learning (CoNLL) dataset is a four-column dataset with the following features: word, part-of-speech, sentence parse token, named entity
* Facebook Research MUSE Word Embeddings
	* Word embeddings are vector representations of words that capture meaning in relation to other words
	* These Facebook word embeddings are from 30 languages aligned into a single vector space


## Related Work
We will use the paper “Neural Architectures for Named Entity Recognition” by Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer as a guide as we create our MNER model. Though the authors of this paper train models from corpora in various languages, they do not attempt to use cross-lingual word embeddings as a means to classify named entities in multiple languages using a single model. 

## Hypothesis
We believe that using features derived from a bi-LSTM on multilingual word embeddings as inputs to a classifier will achieve moderate NER results in a single training language. We hope that generalizing this process to multiple languages will give similar results in untrained languages. We suspect that languages with well-represented vocabularies and grammatical structures similar to the training language will have the best results of the unseen languages.

## Results
We achieved strong results for testing in several languages after training a Bi-LSTM-CRF model on a combined corpus of English, Spanish, and German IOB NER tags. We achieved the following metrics for our full model:

* English: Accuracy: 0.99, Precision: 0.88, Recall: 0.74, F1: 0.79
* German: Accuracy: 0.98, Precision: 0.72, Recall: 0.44, F1: 0.52
* Spanish: Accuracy: 0.98, Precision: 0.82, Recall: 0.68, F1: 0.72

Read more about our results in our [report](./report.pdf).

## Usage

We created a pretrained model that can be tested by running `python ner_parse.py` and typing a sentence in English, Spanish, German, or Italian when prompted.



## References 

Sang, Erik F Tjong Kim. Intro duction to the CoNLL Shared Task Language-Independent Named Entity Recognition. CNTS. University of Antwerp. ACL. USA. 2002. [http://www.aclweb.org/anthology/W02-2024](http://www.aclweb.org/anthology/W02-2024)

Sang, Erik F. Tjong Kim \& Fien De Meulder. Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition. CNTS. University of Antwerp. ACL. USA. 2003. [http://delivery.acm.org/10.1145/1120000/1119195/p142-tjong\_kim\_sang.pdf](http://delivery.acm.org/10.1145/1120000/1119195/p142-tjong\_kim\_sang.pdf)

Lample, Guillaume \& Miguel Ballesteros \&
Sandeep Subramanian \& Kazuya Kawakami \& Chris Dyer. Neural Architectures for Named Entity Recognition. NLP Group. Carnegie Mellon University. DARPA. USA. 2016. [https://arxiv.org/pdf/1603.01360.pdf](https://arxiv.org/pdf/1603.01360.pdf)

Munro, Robert \& Christopher D. Manning. Accurate Unsupervised Joint Named-Entity Extraction from Unaligned Parallel Text. Stanford NLP Group. Stanford University. USA. 2012.
[https://nlp.stanford.edu/pubs/MunroManning2012ner.pdf](https://nlp.stanford.edu/pubs/MunroManning2012ner.pdf)

Chang, Angel X. \& Valentin I. Spitkovsky \& Christopher D. Manning \& Eneko Agirre. A comparison of Named-Entity Disambiguation and Word Sense Disambiguation. Stanford NLP Group. Stanford University. IXA. USA. 2016. [https://nlp.stanford.edu/pubs/chang2016entity.pdf](https://nlp.stanford.edu/pubs/chang2016entity.pdf)


