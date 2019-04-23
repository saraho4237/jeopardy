import pandas as pd
from preprocess_data import clean_data
import pickle as pickle

with open('doc_top_df.pkl','rb') as f:
    model = pickle.load(f)

df,documents,original_questions=clean_data("JEOPARDY_QUESTIONS1.json")

model['answer']=df["answer"].iloc[model["idx"]]
model['question']=df["question"].iloc[model["idx"]]
model['category']=df["category"].iloc[model["idx"]]
model['value']=df["value"].iloc[model["idx"]]
model['episode_air_date']=df["episode_air_date"].iloc[model["idx"]]
model['round']=df["round"].iloc[model["idx"]]
model.drop(["idx"],axis=1,inplace=True)

with open("jeopardy_df.pkl", 'wb') as f:
        pickle.dump(model, f)
