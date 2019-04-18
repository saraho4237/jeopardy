import json
import pandas as pd
import gensim
from nltk.stem.wordnet import WordNetLemmatizer

def clean_data(json_obj):
    # Convert .json to pd.DataFrame
    reports = []
    with open(json_obj) as jep_q:
        for i in jep_q:
            reports.append(json.loads(i))
    df=pd.DataFrame()
    for col in ["category","air_date","question","value","answer","round","show_number"]:
        lst=[reports[0][i][col] for i in range(len(reports[0]))]
        df[col]=lst
    # Remove Media Questions
    idx=[]
    for i,q in enumerate(df["question"]):
        if "www.j-archive.com/media/" in q:
            idx.append(i)
    df.drop(idx,axis=0,inplace=True)
    # Clean up episode date
    df["episode_air_date"]=[d.date() for d in pd.to_datetime(df["air_date"])]
    df["month"]=[date.month for date in df["episode_air_date"]]
    df["year"]=[date.year for date in df["episode_air_date"]]
    df.drop(["air_date"],axis=1,inplace=True)

    df=df.reset_index()
    df.drop(["index"],axis=1,inplace=True)
    original_questions=df["question"].copy()
    #Remove Punctuation and quotes from questions
    df['question'] = df['question'].str.replace('"','')
    df['question'] = df['question'].str.replace("'",'')
    df['question'] = df['question'].str.replace('[^\w\s]', '')
    documents=df["question"]

    return(df,documents,original_questions)

def make_stop_words():
     #Make stop words
     stop_words=[word for word in gensim.parsing.preprocessing.STOPWORDS]
     more=["dont","clue","include","come","call","say","see","youre","know","name","type","like","mean","term","youve","word","year","years","later"]
     more_stop_words=set(stop_words+more)
     return(more_stop_words)

def lemmatize_words(text):
    return (WordNetLemmatizer().lemmatize(text, pos='v'))

def tokenize_words(text):
    stops=make_stop_words()
    result = []
    for token in gensim.utils.simple_preprocess(text):
        token=lemmatize_words(token)
        if (token not in stops) and (len(token) > 3):
            result.append(token)
    return (result)
