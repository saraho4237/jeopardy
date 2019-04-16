import pandas as pd
import matplotlib.pyplot as plt
from clean_data import jep_json_to_df, drop_media_questions, make_date_obj
from visualizations import make_bar

if __name__ == "__main__":
    jep_df=jep_json_to_df("JEOPARDY_QUESTIONS1.json")
    drop_media_questions(jep_df)
    make_date_obj(jep_df)
    


#doc to vec
#word to vec
