# %%
from itertools import combinations

import spacy

nlp = spacy.load("en_core_web_sm")
text = (
    "funny comedy music laugh humor song songs jokes musical hilarious"
)
doc = nlp(text)

for token1, token2 in combinations(doc, 2):
    print(
        f"similarity between {token1} and {token2} is {token1.similarity(token2)}"
    )


# %%
import pandas as pd
from gensim.models import Word2Vec
from tqdm.autonotebook import tqdm


data = pd.read_csv("train.csv")

sentences = []
for review in tqdm(data ["review"]):
    doc = nlp(review, disable = ["tagger", "ner"])
    for sent in doc.sents:
        sentences.append(
            [token.text for token in sent
            if not token.is_punct])
    

#%%  
# train a Word2vec model
model = Word2Vec(sentences)


# %%
for token1, token2 in combinations(text.split(), 2):
    print(
        f"similarity between {token1} and {token2} is {model.wv.similarity(token1, token2)}"
    )

# %%
