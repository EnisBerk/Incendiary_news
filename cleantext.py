
import json
import nltk
from TurkishStemmer import TurkishStemmer
import pickle

# there are four files with data
# they are dictionaries, keys are links to news and text of the news is value

# Incendiary News
text_store=open("./data/article_text_pos_1063_57531.json","r")
article_text_dict_positive = json.load(text_store)
text_store.close()
# '3815', "text"

# Non-Incendiary News from BBC, first iteration
text_store=open("./data/article_text_neg_bbc_iter1_07272.json","r")
iter1_BBC_text_dict_neg = json.load(text_store)
text_store.close()

#  Non-Incendiary News from BBC, second iteration
text_store=open("./data/iter2_text_neg_bbc_12981.json","r")
iter2_BBC_text_dict_neg = json.load(text_store)
text_store.close()

# Non-Incendiary News from CNN, first iteration
text_store=open("./data/iter1_article_text_neg_CNN_08109.json","r")
iter1_CNN_neg_text = json.load(text_store)
text_store.close()





#those stop words to be removed
my_stopwords= ['acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri',
				 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü',
				 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 
				 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 
				 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü',
				 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye',
				 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm',
				 've', 'veya', 'ya', 'yani']

# those exists in specific sources and they are features that are not related to hate-speech so they will be removed 
my_stopwords.extend(['facebook', 'telif','hakkı', 'telif hakkı','dr','bbc','bır','dan', 'den',
					'karsı','twitter',"caption","image",
					"image caption","tiklayin",'copyright',
					 'yayın','sayfa','no',
					 "tüm","hakları",'saklıdır','copyright'
					"sayfa","world","world service","yeni akit","gazetesihalkalı",
					"tüm hakları saklıdır","hakları saklıdır","hakları",'saklıdır',
					"yayın","sayfa","no",'ve',"cnn","aa","destekyeniakitcomtr","httpyeniakitcomtr"])

# some websites tends to use special characters too much, remove them all
# removethose='“’‘”•.,\'\"!;@?())'
# removethose+='*'+ '0'+ ':'+ ']'+ '_'+'$'+ '['+'{'+ '}'+ '»'
# removethose+='©'+'-'+"1234567890"+"&"+"—"+"/"+"|"+"="+">"+"…"+"%"+"′"

# removethose=['“', '’', '‘', '”', '•', '.', ',', "'", '"', '!', 
#              ';', '@', '?', '(', ')', ')', '*', '0', ':', ']',
#              '_', '$', '[', '{', '}', '»', '©', '-', '1', '2', 
#              '3', '4', '5', '6', '7', '8', '9', '0', '&', '—', 
#              '/', '|', '=', '>', '…', '%', '′','€', '¥','£','›',
#              '¼','<','¨','‏','­','–','#','+']

# keep only those turkish letters
lowercase=' abcdefghijklmnoprstuvyzğöıüşç'

stemmer = TurkishStemmer()

# this function lowers text, 
# then removes characters such as @ or » because
# those are used in some resources of news more than others and
# also corrects some characters such as 'i̇' which
# does not exists in Turkish but found in the text
# due to encoding or scraping errors
def pre_process_clean(text):
	text=text.lower()
	text=text.strip()
	text=text.replace('i̇','i').replace('î','i').replace('â','a').replace('á','a')
	text=text.replace('ū','ü').replace('û','u')
	text=text.replace('è','e').replace('é','e').replace('ê','e')
	result=""
	for thechar in text:
		if (thechar in lowercase):
			result+=thechar
	return result

def pre_process_stopwords(text):
	text=nltk.word_tokenize(text.lower().strip())
	text=filter(lambda x: x not in my_stopwords,text )
	text=" ".join(text)
	return text


for a_dict in [article_text_dict_positive,iter1_BBC_text_dict_neg,
										iter2_BBC_text_dict_neg,iter1_CNN_neg_text]:
	for key,value in a_dict.items():
		a_dict[key]=pre_process_stopwords(pre_process_clean(value))


def remove_short_articles(article_dict,char_lim=201):
	toremove=[]
	for article_text in article_dict.items():
		if len(article_text[1])<char_lim:
			toremove.append(article_text[0])

	for keys in toremove:
		del article_dict[keys]
	return article_dict

article_text_dict_positive=remove_short_articles(article_text_dict_positive)
iter1_BBC_text_dict_neg=remove_short_articles(iter1_BBC_text_dict_neg)
iter2_BBC_text_dict_neg=remove_short_articles(iter2_BBC_text_dict_neg)
iter1_CNN_neg_text=remove_short_articles(iter1_CNN_neg_text)


all_articles={"article_text_dict_positive":article_text_dict_positive,
				"iter1_BBC_text_dict_neg":iter1_BBC_text_dict_neg,
				"iter2_BBC_text_dict_neg":iter2_BBC_text_dict_neg,
				"iter1_CNN_neg_text":iter1_CNN_neg_text}

pickle.dump( all_articles, open( "./data/clean_data.p", "wb" ) )
all_text=[]

# create list of all words to be used with fasttext for word2vec
for artic_dict in all_articles:
	for artic in all_articles[artic_dict].values():
		all_text.append(artic)

all_text=" ".join(all_text)
all_text=nltk.word_tokenize(all_text.lower().strip())
text_freq=nltk.FreqDist(all_text)
str_to_file="\n".join(text_freq.keys())
myfile=open("./data/wordlist.txt","w")
myfile.write(str_to_file)
myfile.close()
