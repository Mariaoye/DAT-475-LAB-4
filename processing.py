# %%
from PIL.Image import ORDERED
import pandas as pd
import spacy
from tqdm.autonotebook import tqdm


nlp = spacy.load("en_core_web_sm")
data = pd.read_csv("train.csv")


#%%
def get_adjs(doc):
    result = [
        token.text for token in doc
        if token.pos_ == "ADJ"
    ]
    return result


def get_ents(doc):
    result = [
        ent.text for ent in doc.ents
        if ent.label_ in ["PERSON", "NORP", "FAC","ORG"]
    ]
    return result


def get_number_of_sents(doc):
    result = len(list(doc.sents))
    return result


# %%
from collections import Counter


number_of_sents, pos_adj_counter, neg_adj_counter, ent_counter = (
    [],
    Counter(),
    Counter(),
    Counter(),
)

for review, sentiment in tqdm(data.to_records(index=False)):
    doc = nlp(review)
    adjs = get_adjs(doc)
    ents = get_ents(doc)

    # what to do with number_of_sents

    number_of_sents.append(
        get_number_of_sents(doc)
        )

    # what to do with pos_adj_counter and neg_adj_counter
    for adj in adjs:
        counter = pos_adj_counter if sentiment == "positive" else neg_adj_counter
        counter[adj.lower()] += 1

        #OR

    # if sentiment == "positive":
        #counter = pos_adj_counter
    #else:
        #counter = neg_adj_counter
        #counter[adj] += 1


    # what to do with ent_counter
    for ent in ents:
        ent_counter[ent.lower()] += 1


# %%
data["number_of_sents"] = number_of_sents
import seaborn as sns

sns.boxplot(x="sentiment", y="number_of_sents", data=data)


# %%
top20_ent = ent_counter.most_common()[:20]

y = [t[0] for t in top20_ent]
x = [t[1] for t in top20_ent]

sns.barplot(x=x, y=y)


# %%
import matplotlib.pyplot as plt

_, axes = plt.subplots(1, 2, figsize=(16, 8))

top20_pos_adj = pos_adj_counter.most_common()[:20]
y = [t[0] for t in top20_pos_adj]
x = [t[1] for t in top20_pos_adj]
sns.barplot(x=x, y=y, ax=axes[0])


top20_neg_adj = neg_adj_counter.most_common()[:20]
y = [t[0] for t in top20_neg_adj]
x = [t[1] for t in top20_neg_adj]
sns.barplot(x=x, y=y, ax=axes[1])
