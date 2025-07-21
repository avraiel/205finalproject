import time
start_time = time.time()

import json
import copy

from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
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


# Sentences we want sentence embeddings for
# sentences = ['This is an example sentence', 'Each sentence is converted']
"""ONE SINGULAR INPUT"""
sentences = ['{"body": "Kung hindi rin lang paniniwalaan ang datos sa war on drugs, mas makakabubuti pang buwagin na lang ang Philippine National Police pati na rin ang gobyerno, ayon kay Senador Ronald \\u201cBato\\u201d Dela Rosa. \\u201cIf you do not trust PNP numbers, you dissolve the PNP. If you do not trust the government, tanggalin ang gobyerno. Let the human rights, sila ang mag-rule ng ating bansa,\\u201d pahayag ni Dela Rosa sa panayam ng ANC Headstart. \\u201c\\u2019Pag ganoong wala tayong tiwala sa ating government instrumentality, edi dissolve natin lahat, pati gobyerno,\\u201d sabi ng dating Philippine National Police (PNP) chief. Ginawa ni Dela Rosa ang pahayag kasunod ng muling paggiit ni United Nations High Commissioner for Human Rights Michelle Bachelet na kailangang may managot sa pagkamatay ng 25,000 katao sa kampanya ng Duterte administration laban sa iligal na droga. Ang nasabing bilang ay mas mataas sa datos ng gobyerno na 6,663 na drug-related death. Pero para kay Del Rosa, kalokohan lang umano nakuhang datos ni Bachelet. \\u201cSino siya (Bachelet) para magsabi nang ganoon dito sa atin? Kung Nakapunta na ba siya dito, naobserbahan ba niya kung ano\\u2019ng nangyayari dito?\\u201d tanong natin sa UN Human Rights chief. \\u201cKasi ang pinagbasehan kasi niyang report, ridiculous na ginawa ng ating mga kababyan. Isipin mo 27,000 ang napatay sa war on drugs samantalang ang official record ng PNP, 10,000 plus lang,\\u201d sambit pa nito. \\u201cEh talagang magre-reak siya ng ganoon kung makatanggap siya ng ganoong report. Hindi naman siya makapunta dito para mag observe, obserbahan niya pag nandito siya,\\u201d ayon pa kay Dela Rosa. Welcome naman sa senador na magpadala ang UN Human Right Council na kanilang kinatawan sa Pilipinas para ma-assess ang drug campaign ng gobyerno. \\u201cI welcome that move para malaman nila ang katotohanan. But it\\u2019s not for me to decide. It\\u2019s an executive decision, kung iaallow silang pumunta dito,\\u201d sabi ni Dela Rosa.", "title": "Ayaw n\\u2019yo paniwalaan drug war data? PNP, gobyerno buwagin na \\u2013 Bato", "website": "AbanteTNT", "category": "News", "date": "Jul 8, 2020 @ 14:30", "author": "Dindo Matining", "url": "https://tnt.abante.com.ph/ayaw-nyo-paniwalaan-drug-war-data-pnp-gobyerno-buwagin-na-bato/", "img_url": "https://tnt.abante.com.ph/wp-content/uploads/2020/07/Dela-Rosa.jpg", "img_path": "a14690aff7cba59b358e8c97b684c58f29537e0716caa21bfd511da8996b078c.jpg"}']
# sentences = ['{"body": "Kung hindi rin lang paniniwalaan ang datos sa war on drugs, mas makakabubuti pang buwagin na lang ang Philippine National Police pati na rin ang gobyerno, ayon kay Senador Ronald \\u201cBato\\u201d Dela Rosa. \\u201cIf you do not trust PNP numbers, you dissolve the PNP. If you do not trust the government, tanggalin ang gobyerno. Let the human rights, sila ang mag-rule ng ating bansa,\\u201d pahayag ni Dela Rosa sa panayam ng ANC Headstart. \\u201c\\u2019Pag ganoong wala tayong tiwala sa ating government instrumentality, edi dissolve natin lahat, pati gobyerno,\\u201d sabi ng dating Philippine National Po']
file_path = 'test.json'
def clean_json(json_object):
    for obj in json_object:
        obj['body'] = ' '.join(obj['body'])
    return json_object

with open(file_path, 'r') as file:
    dataset = json.load(file)
dataset_clean = clean_json(copy.deepcopy(dataset))

# dataset_clean = dataset_clean[0:20]
dataset_clean = dataset_clean[0:2439]

def chunkify(text, max_length):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=max_length*0.2
    )
    chunks = [x for x in text_splitter.split_text(text)]
    
    return chunks


# Load model from HuggingFace Hub
model = "jcblaise/roberta-tagalog-base"
total_articles = 0
total_entries = 0
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForMaskedLM.from_pretrained(model)
print(tokenizer.model_max_length)
with psycopg.connect("dbname=vectordb user=postgres", password=os.getenv('PSQL_PASSWORD')) as conn:
    with conn.cursor() as cur:
        cur.execute("""
                CREATE TABLE IF NOT EXISTS jcblaise (
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
        print("Created the jcblaise table!")
        for entry in dataset_clean:
            title = entry['title']
            website = entry['website']
            category = entry['category']
            url = entry['title']
            date = entry['title']
            author = entry['title']

            stringed = json.dumps(entry)
            chunked = chunkify(stringed, 512)
            encoded_input = tokenizer(chunked, padding=True, truncation=False, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input, output_hidden_states=True)
                embeddings = model_output.hidden_states[-1]
                sentence_embeddings = torch.mean(embeddings, dim=1).squeeze()

            chunk_idx = 0
            for sentence in sentence_embeddings:
                raw = chunked[chunk_idx]
                cur.execute("""
                    INSERT INTO jcblaise (embedding, raw, title, website, category, url, date, author)
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

