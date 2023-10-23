from fastapi import FastAPI
from pymongo import MongoClient
import pandas as pd
import pickle
import re
import json



app = FastAPI()


mongo_uri = 'mongodb://your_username:your_password@localhost:27017/flipkart_db'
client = MongoClient(mongo_uri)

model = pickle.load(open('log_reg_training.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_training.pkl', 'rb'))


db = client['flipkart_db']
collection = db['flipkart']
data = list(collection.find())

input_collection = db['flipkart_input']
input_data = list(input_collection.find())


df = pd.DataFrame(data)
df2 = pd.DataFrame(input_data)


search_query = {
                    'written english'              : 0,
                    'camera'                       : 1,
                    'sony cybershot'               : 2,
                    'dslr canon'                   : 3,
                    'mathematics'                  : 4,
                    'data structures algorithms+'  : 5, 
                    'nike-deodrant'                : 6, 
                    'spoken english'               : 7,
                    'best-seller books'            : 8,
                    'tommy watch'                  : 9,
                    'c programming'                : 10,
                    'physics'                      : 11,
                    'chemistry'                    : 12,
                    'camcorder'                    : 13,
                    'dell laptops'                 : 14,
                    'titan watch'                  : 15,
                    'calvin klein'                 : 16, 
                    'timex watch'                  : 17,
                    'axe deo'                      : 18,
                    'chromebook'                   : 19,
}



def clean_text(text):
    text = re.sub(r'\"', ' ', text)
    text = re.sub(r'\,.', ' ', text)
    text = re.sub(r'\(', ' ', text)
    text = re.sub(r'\)', ' ', text)
    text = re.sub(r'\-', ' ', text)
    text = re.sub(r'\...', ' ', text)
    text = re.sub(r'\&', ' ', text)
    return text


def strip_dots(text):
    return text.split('...')[0]


def query(word:str, model, vectorizer):
    df = pd.DataFrame([word], columns=['product_names'])
    df['product_names'] = df['product_names'].apply(clean_text)
    model_value = model.predict(vectorizer.transform(df['product_names'])).tolist()
    for key, value in search_query.items():
        if value == model_value[0]:
            return json.dumps(key)
    return f'No search query for this request'




@app.get("/{search_query}")
async def read_query(search_query: str):
    strip_dots(search_query)
    result = query(search_query, model, vectorizer)
    return {"result": result}



if __name__=='__main__':
    uvicorn.run(app,host='localhost',port=8000)