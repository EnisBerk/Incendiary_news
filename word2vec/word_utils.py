import nltk
import pickle
from datetime import datetime
import os
import sys
# sys.path.append("..")
sys.path.append('../website/crawlers/')
# from website.crawlers.article import Article
# RSS_links --> collect news  links (time,title,url) --> download news, parse text --->
# save as article object ---> update_words list ---> inference with ML


def clean_text(text):
    # text=text.replace("\n", " ")
    return text
# news_articles{url:article_object}
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

# stemmer = TurkishStemmer()

# this function lowers text,
# then removes characters such as @ or » because
# those are used in some resources of news more than others and
# also corrects some characters such as 'i̇' which
# does not exists in Turkish but found in the text
# due to encoding or scraping errors
def pre_process_clean(text):
    text=text.lower()
    # text=text.strip()
    text=text.replace('i̇','i').replace('î','i').replace('â','a').replace('á','a')
    text=text.replace('ū','ü').replace('û','u')
    text=text.replace('è','e').replace('é','e').replace('ê','e')
    text=text.replace("\n", " ")
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
if __name__ == "__main__":
    # TODO if a text file older than word list, pass that one
    path="../website/crawlers/database/"
    database_files=os.listdir(path)

    database_files=list(filter(lambda x: "new_articles" in x,database_files ))
    all_text=""
    for file in database_files:
        news_articles=pickle.load( open( path+file, "rb" ) )
        the_text=[pre_process_stopwords(pre_process_clean(article_obj.text)) for article_obj in news_articles.values()]
        the_text=" ".join(the_text)
        the_text=clean_text(the_text)
        all_text+=the_text

    wordlist_files=os.listdir("./data/")
    wordlist_files=list(filter(lambda x: "wordlist" in x,wordlist_files ))
    wordlist_files.sort(reverse=True)
    last_one=wordlist_files[0]
    with open("./data/"+last_one) as last_word_list_file:
        last_word_list=[line.strip() for line in last_word_list_file.readlines()]

    all_text=nltk.word_tokenize(all_text.lower(),language='turkish')
    text_freq=nltk.FreqDist(all_text)
    new_words=list(text_freq.keys())
    new_words.extend(last_word_list)
    new_words=list(set(new_words))
    str_to_file="\n".join(new_words)
    time_stamp=datetime.strftime(datetime.now(),"_%y_%m_%d_%H_%M_%S")
    myfile=open("./data/wordlist"+time_stamp+".txt","w")
    myfile.write(str_to_file)
    myfile.close()
