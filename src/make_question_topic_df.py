import pickle as pickle
from preprocess_data import clean_data
import pandas as pd

with open('../my_app/data/doc_top_df.pkl','rb') as f:
    model = pickle.load(f)

df,documents,original_questions=clean_data("../JEOPARDY_QUESTIONS1.json")
model["question"]=original_questions[model["idx"]]
model.drop(["idx"],axis=1,inplace=True)

with open('../my_app/data/quest_top_df.pkl','wb') as f:
     pickle.dump(model,f)
