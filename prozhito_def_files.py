import re
import pandas as pd
import nltk
from nltk import punkt
from nltk import *
from pymystem3 import Mystem
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy2

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

dp=pd.read_csv('source/persons_daries.csv', sep=',' , low_memory=False)
sns.color_palette("tab10")
sns.set(style="whitegrid")

morph = pymorphy2.MorphAnalyzer()
m = Mystem(disambiguation=False)

def extract_by_id(id,data):
    try:
        extract = data.loc[data['prozhito_id']==int(id)]['text']
        if len(extract)==0:
            return None
        
        return pd.DataFrame(extract.apply(lambda x:re.sub(r'<[^>]+>', '', x)))
    
    except:
        print('ошибка')
        return None
    

def len_sent(tdf):
    tdf['sent_count'] = tdf['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
    tdf['word_count'] = tdf['text'].apply(lambda x:len(nltk.word_tokenize(x)))
    tdf['word_per_sent'] = tdf['word_count']/tdf['sent_count']
    word_per_sent = tdf['word_per_sent'].mean()
    return round(word_per_sent) 
# print(f'-------СРЕДНЕЕ КОЛИЧЕСТВО СЛОВ В ПРЕДЛОЖЕНИИ РАВНО ~ {round(word_per_sent)}--------') 

def plot_dirty(tdf, ax):
    count_vect_total = CountVectorizer(ngram_range=(1,1), min_df=5)

    corpus_total = [x for x in tdf['text'].fillna(' ') if len(str(x)) > 0]

    corpus_total_fit = count_vect_total.fit_transform(corpus_total)
    total_counts = pd.DataFrame(corpus_total_fit.toarray(), columns=count_vect_total.get_feature_names_out()).sum()
    ngram_total_df = pd.DataFrame(total_counts, columns=['counts'])

    
    ngram_total_df = ngram_total_df.sort_values(by='counts', ascending=False)
    
    plot_dirty = sns.barplot(x="counts",
                             y=ngram_total_df.head(20).index,
                             ax=ax,
                             data=ngram_total_df.head(20),
                             hue=ngram_total_df.head(20).index,
                             legend=False)
    
    ax.set_title('TОП СЛОВ БЕЗ ЧИСТКИ')
    return plot_dirty

def lemmatize(wrds, m):
    res = []
    for wrd in wrds:
        p = m.parse(wrd)[0]
        res.append(p.normal_form)
        
    return res

def tokenize(text, stoplst):
    without_stop_words = []
    txxxt = nltk.word_tokenize(text)
    for word in txxxt:
        if len(word) == 1:
            continue
        if word.lower() not in stoplst:
            without_stop_words.append(word)
    return without_stop_words


def plot_clean(tdf,stop_words, ax):
    text = str(tdf['text'].values)#.astype('str')
    tokens = word_tokenize(text)
    tokens = lemmatize(tokens, morph)
    filtered_tokens = [word for word in tokens if word not in stop_words]

    count_vect_total = CountVectorizer(ngram_range=(1,2),min_df=5)
    corpus_total = [x for x in filtered_tokens if (str(x) not in stop_words) and (len(x)>3)]


    corpus_total_fit = count_vect_total.fit_transform(corpus_total)#(corpus_total)

    total_counts = pd.DataFrame(corpus_total_fit.toarray(),columns=count_vect_total.get_feature_names_out()).sum()
    ngram_total_df = pd.DataFrame(total_counts,columns=['counts'])

    ngram_total_df = ngram_total_df.sort_values(by=['counts'],ascending=False)
    
    plot_clean = sns.barplot(x="counts",
                            y=ngram_total_df.head(20).index,
                            data=ngram_total_df.head(20),
                            
                            hue=ngram_total_df.head(20).index,
                            legend=False, ax=ax)
    ax.set_title('TОП СЛОВ ПОСЛЕ ЧИСТКИ')
                    
    return plot_clean


def uniq_words_share(tdf,stoplst):
    txt = str(tdf['text'].values)
    tokenized_text = tokenize(txt, stop_words)
    tokens = lemmatize(tokenized_text, morph)
    words = len(tokens)
    uniq_words = len(set(tokens))
    return (uniq_words/words) * 100



def process_and_visualize(stop_words, tdf):
    text = str(tdf['text'].values)
    
    def pre_process(text):
        text = re.sub(r"</?.*?>", " <> ", str(text))
        text = re.sub(r"(\\d|\\W)+", "", text)
        text = re.sub(r'[^а-яА-Я\s]+', ' ', text)
        text = text.lower()
        text = re.sub(r"\r\n", " ", text)
        text = re.sub(r"\xa0", " ", text)
        sub_str = 'sfgff'
        text = text[:text.find(sub_str)]
        return text

    morph = pymorphy2.MorphAnalyzer()

    cleaned_text = pre_process(text)
    tokenized_text = tokenize(cleaned_text, stop_words)
    lemmatized_text = lemmatize(tokenized_text, morph)

    Fdist = FreqDist(lemmatized_text)

    not_most_common = Fdist.most_common()[-21:-1]
    rare_words = [i[0] for i in not_most_common]

    

    return ', '.join(str(w) for w in rare_words)




def plot_pos(tdf, ax):
    pos_counter_bk = Counter()
    bk = str(tdf['text'].values)

    for sentence in sent_tokenize(bk, language="russian"):
        doc = m.analyze(sentence) 
        for word in doc: 
                if "analysis" not in word or len(word["analysis"]) == 0: 
                    continue

                gr = word["analysis"][0]['gr']
                pos = gr.split("=")[0].split(",")[0]
                pos_counter_bk[pos] += 1 

    pos_tags = []
    counts = []

    for pos, count in pos_counter_bk.most_common():
        pos_tags.append(pos)
        counts.append(count)

    percents = [count / sum(counts) * 100 for count in counts]
    bk_plt = ax.bar(pos_tags, percents, )
    ax.set_title('TОП ЧАСТЕЙ РЕЧИ в %')
    plt.xticks(rotation=30)
    
    return bk_plt

  


def plot_top_names(tdf, ax):
    morph = pymorphy2.MorphAnalyzer()

    def names_extr(wrds):
        res = []
        for wrd in wrds:
            p = morph.parse(wrd)[0]
            if 'Name' in p.tag:
                res.append(wrd)
        return res

    text = str(tdf['text'].values)

    without_stop_words = tokenize(text, stop_words)
    without_stop_words = lemmatize(without_stop_words, morph)
    is_name = names_extr(without_stop_words)
    Fdist = FreqDist(is_name)

    df = pd.DataFrame(list(Fdist.items()), columns=['Name', 'Count']).sort_values(by = 'Count',ascending=False)
    df = df.head(20)

    top_names_plot = sns.barplot(x="Count",
                             y="Name",
                             data=df,
                             legend=False, ax=ax)
    ax.set_title('ТОП ИМЁН')

    return top_names_plot





def plot_all_graphs(fig, axs, id):
    ### apply:
    tdf = extract_by_id(id,dp)
    if isinstance(tdf, pd.DataFrame)==False:
        if tdf==None:   
            return None, None
    

    messages=messages = [f'Обработано записей дневника: {len(tdf)}', 
                         f"Среднее кол-во слов в предложении ~ {len_sent(tdf)},",
                         f"Доля уникальных слов составляет ~ {round(uniq_words_share(tdf,stop_words))}%",
                         f"Список 20-ти редко встречающихся слов: {process_and_visualize(stop_words, tdf)}"]



    plot_clean(tdf,stop_words, axs[0, 0])
    # Вызовите остальные функции и передайте им соответствующий объект ax
    plot_dirty(tdf, axs[0, 1])
    plot_pos(tdf, axs[1, 0])
    plot_top_names(tdf, axs[1, 1])
    #fig.tight_layout()
    
    return fig, messages


if __name__ == '__main__':
    id='2525'
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18.5, 14), layout="constrained")
    fig, messages = plot_all_graphs(fig, axs, id)
    plt.savefig('prozhito.png')
