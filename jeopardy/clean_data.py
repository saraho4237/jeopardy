import json
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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

    return(df)

def vectorize_data(df,col):
     documents=df[col]
     #Tokenize
     tokenized_docs = [word_tokenize(content) for content in documents]
     #Remove stop words
     stop_words=['___','____','_____',"a","an","the","also","am","among","another","appear","are","arent","at","b","be","because","belong","best","both","br","bring","c"]
