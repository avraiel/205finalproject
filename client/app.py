from flask import Flask, request, render_template
import psycopg
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import torch
import torch.nn.functional as F


import os
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)



# PostgreSQL connection settings (customize these)
DB_CONFIG = {
    'host': 'localhost',
    'dbname': 'vectordb',
    'user': 'postgres',
    'password': os.getenv('PSQL_PASSWORD'), # TODO: PUT YOUR PASSWORD IN AN ENV FILE
    'port': '5432'
}

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embedding(prompt, model):
    models = {
        'mpnet' : 'sentence-transformers/all-mpnet-base-v2',
        'openai' : 'text-embedding-3-small',
        'jcblaise' : 'jcblaise/roberta-tagalog-base',
        'dost' : 'dost-asti/RoBERTa-tl-cased',
    }


    embedding = None
    if model == "openai":
        client = OpenAI(api_key = os.getenv('OPENAI_KEY'))
        response = client.embeddings.create(
                input = prompt,
                model=models[model]
            )
        # print(len(response))
        embedding = response.data[0].embedding
        embedding = list(embedding) # weird formatting to make sure the formats are uniform
    
    elif model == "mpnet":
        tokenizer = AutoTokenizer.from_pretrained(models[model])
        external_model = AutoModel.from_pretrained(models[model])    
        with torch.no_grad():   
            encoded_input = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
            model_output = external_model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            embedding = F.normalize(sentence_embeddings, p=2, dim=1)
            embedding = embedding.tolist()[0]


    elif model == "jcblaise" or model == "dost":
        tokenizer = AutoTokenizer.from_pretrained(models[model])
        external_model = AutoModelForMaskedLM.from_pretrained(models[model])
        encoded_input = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
        with torch.no_grad():
            model_output = external_model(**encoded_input, output_hidden_states=True)
            embeddings = model_output.hidden_states[-1]
            embedding = torch.mean(embeddings, dim=1)
            embedding = embedding.tolist()[0]

    return embedding


def get_db_connection():
    conn = psycopg.connect(**DB_CONFIG)
    return conn

@app.route('/', methods=['GET', 'POST'])
def index():
    average = None
    result = None
    model = None
    prompt = ""
    lite_checked = False
    lite = ""
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        model = request.form.get('model')
        lite =  "lite" if request.form.get('lite') else ""
        lite_checked = True if request.form.get('lite') else False
        table = model+lite

        prompt_embedding = get_embedding(prompt, model)
        with get_db_connection() as conn:
            print("Made a connection!")
            
            

            with conn.cursor() as cur:
                pass
                cur.execute(f"""
                    SELECT d.id, d.title, d.website, d.date, d.embedding <=> '{prompt_embedding}' AS distance, d.url, d.raw
                    FROM {table} d
                    ORDER BY d.embedding <=> '{prompt_embedding}'
                    LIMIT 5;
                """)
                result = cur.fetchall()
                average = 0
                for entry in result:
                    average += entry[4]
                average /= 5
            conn.close()
            print("Closed connection..")

    print(prompt)
    print(model)
    print(lite)
    return render_template('index.html', average=average, result=result, model=model, prompt=prompt, lite=lite_checked)


if __name__ == '__main__':
    app.run(debug=True)
