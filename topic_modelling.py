# %%
from matplotlib.pyplot import text
import pandas as pd
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load("en_core_web_sm")
data = pd.read_csv("train.csv")


def show_topic(model, feature_names, top):
    for index, distribution in enumerate(model.components_):
        sorted_word_indices = distribution.argsort()[::-1][:top]
        print(f"Topic {index}:")
        print(" ".join([feature_names[i] for i in sorted_word_indices]))

#%%
# CountVectorizer 

def custom_tokenizer(text):
    return [
        token.text
        for token in nlp(text, disable=["tagger" , "parser", "ner"])
        if not token.is_stop and not token.is_punct
    ]


tf_vectorizer = CountVectorizer(     # set up your CountVectorizer
    max_df= .8,
    min_df= 10, #25000 * .01 = 250
    max_features=1000,
    tokenizer=custom_tokenizer,
)
tf = tf_vectorizer.fit_transform(data["review"])

lda = LatentDirichletAllocation(
    # set up your LatentDirichletAllocation
    n_components=30,
    learning_method="online",
    learning_offset=50,
    max_iter=5,
    random_state=2020,
)
lda.fit(tf)


tf_feature_names = tf_vectorizer.get_feature_names()
top = 10
show_topic(lda, tf_feature_names, top)


# %%
