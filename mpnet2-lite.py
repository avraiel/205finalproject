import time
start_time = time.time()

import json
import copy

from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import torch.nn.functional as F

import psycopg
import os
from dotenv import load_dotenv
load_dotenv()

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# We want sentence embeddings for
"""ONE SINGULAR INPUT"""

file_path = 'test.json'
def clean_json(json_object):
    for obj in json_object:
        obj['body'] = ' '.join(obj['body'])
    return json_object

with open(file_path, 'r') as file:
    dataset = json.load(file)
dataset_clean = clean_json(copy.deepcopy(dataset))

def remove_metadata(json_object):
    for obj in json_object:
        del obj['img_path']
        del obj['img_url']
    return json_object

# embedding does not factor in img_path and img_url
dataset_clean = remove_metadata(dataset_clean)

# dataset_clean = dataset_clean[0:10]
dataset_clean = dataset_clean[0:2439]





def chunkify(text, max_length):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=max_length*0.2
    )
    chunks = [x for x in text_splitter.split_text(text)]
    
    return chunks



# Load model from HuggingFace Hub
total_articles = 0
total_entries = 0
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
print(tokenizer.model_max_length)
with psycopg.connect("dbname=vectordb user=postgres", password=os.getenv('PSQL_PASSWORD')) as conn:
    with conn.cursor() as cur:
        cur.execute("""
                CREATE TABLE IF NOT EXISTS mpnetlite (
                    id bigserial PRIMARY KEY,
                    embedding vector(768),
                    raw varchar(1000),
                    title varchar(500),
                    website varchar(255),
                    category varchar(255),
                    url varchar(500),
                    date varchar(255),
                    author varchar(255))
                """)
        print("Created the mpnet table!")
        for entry in dataset_clean:
            title = entry['title']
            website = entry['website']
            category = entry['category']
            url = entry['title']
            date = entry['title']
            author = entry['title']

            stringed = json.dumps(entry)
            chunked = chunkify(stringed, tokenizer.model_max_length)
            encoded_input = tokenizer(chunked, padding=True, truncation=False, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            chunk_idx = 0
            for sentence in sentence_embeddings:
                raw = chunked[chunk_idx]
                cur.execute("""
                    INSERT INTO mpnetlite (embedding, raw, title, website, category, url, date, author)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""", 
                            (sentence.tolist(), raw, title, website, category, url, date, author)
                    )
                chunk_idx += 1

            total_articles += 1
            total_entries += len(chunked)
            print(f"{total_entries} entries --- {total_articles} articles --- {time.time() - start_time} seconds")
            if total_entries%1000 == 0:
                print(f"{total_entries} entries --- {total_articles} articles --- {time.time() - start_time} seconds")
    conn.commit()

print(total_entries)
print(len(dataset_clean))

# Split Text if too long
# sentences = chunkify(sentences[0], tokenizer.model_max_length)
# print(sentences)
# for x in sentences:
#     print(len(x))
# print([len(x)] for x in sentences)

# Tokenize sentences


"""Loading takes 7.73 seconds, more than converting them to embedding"""
print("--- %s seconds ---" % (time.time() - start_time))

# # Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)

# # Perform pooling
# sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# # Normalize embeddings
# sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# print("Sentence embeddings:")
# # print(sentence_embeddings)
# print(len(sentence_embeddings))

# print("--- %s seconds ---" % (time.time() - start_time))

