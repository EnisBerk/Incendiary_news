
# Repository for Incendiary News Detection paper submitted to FLAIRS-32:


Notebooks according to their names:  
* word2vec: word vectors as features with different classifiers.  
* BBC_BBC : we are training with BBC positive samples and testing with BBC again.  
* BBC_CNN : we are training with BBC positive samples and testing with CNN positive samples.  


Other files:

* ./data/clean_data.p : is a pickle file, includes all datasets cleaned by cleantext.py

* [word2vec.txt](https://storage.googleapis.com/deep_learning_enis/Incendiary_news/word2vec.txt) : stores vectors for each word exists in the corpus, generated with [fasttext](https://github.com/facebookresearch/fastText). It is tab separated and each line have a word followed with 300 floating point numbers. 

* cleantext.py : This code cleans corpus and generates clean_data.p pickle. Cleaning includes removing website urls, source names, words related to crawling process of specific sources, characters that exists only specific resources. Also articles with less than 100 characters are removed from corpus. For details please check the file. 
After cleaning data, it also create list of all words to be used with fasttext for word2vec and saves to ./data/wordlist.txt

Original data collected is stored in data folder:
* article_text_pos_1063_57531.json        Incendiary News
* article_text_neg_bbc_iter1_07272.json   Non-Incendiary News from BBC, first iteration
* iter2_text_neg_bbc_12981.json           Non-Incendiary News from BBC, second iteration
* iter1_article_text_neg_CNN_08109.json   Non-Incendiary News from CNN, first iteration

Authors:
* [@EnisBerk](https://github.com/EnisBerk)
* [Prof. Elena Filatova](https://ef2020.commons.gc.cuny.edu/)


Requirements:  
python=3.7.1  
scikit-learn=0.20.0  
nltk=3.3.0  
numpy=1.15.2  

#my_tagger.yaml and pos_tagger.py files from [turkish-pos-tagger](https://github.com/onuryilmaz/turkish-pos-tagger/tree/a889bc2e633561f5050035cd1ffaf91b3ef38fe5)  
#[turkish-stemmer-python](https://github.com/otuncelli/turkish-stemmer-python/tree/1f60006c023152e46e5704065cdc51e68d63240a)

## How to get notebooks working:

Add following lines to beginning of the notebook, downloands all required files 
```

!git clone https://github.com/EnisBerk/Incendiary_news.git
%cd Incendiary_news

!git clone https://github.com/otuncelli/turkish-stemmer-python.git
%cd turkish-stemmer-python/
!git reset --hard 1f60006c023152e46e5704065cdc51e68d63240a
%cd ../

!git clone https://github.com/onuryilmaz/turkish-pos-tagger.git
%cd turkish-pos-tagger
!git reset --hard a889bc2e633561f5050035cd1ffaf91b3ef38fe5
%cd ../

!cp -r turkish-pos-tagger/* ./
!cp -r turkish-stemmer-python/* ./

!rm -r turkish-pos-tagger
!rm -r turkish-stemmer-python

!curl -O https://storage.googleapis.com/deep_learning_enis/Incendiary_news/word2vec.txt
!cp word2vec.txt ./data/
```

