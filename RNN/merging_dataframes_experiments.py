import json
import stanza
import stanza_tokenizer_v2
import string
import random
import pandas as pd
import numpy as np
from collections import Counter
from myUtils import rare_words_to_unknown

# GET ALL MECO WORDS PREPROCESSED
with open("words_en_meco_manually_fixed.txt", 'r') as f:
    meco_documents = f.read()

# split documents based on new paragraphs
meco_docs = meco_documents.split('\n\n')[:-1]

# remove the \n from each document
meco_docs = [text.replace('\\n', ' ') for text in meco_docs]

# extra preprocessing to make the merging with RNN outputs easier
meco_docs = [text.replace('-', ' ') for text in meco_docs]
meco_docs = [text.replace("\'s", '') for text in meco_docs]

# meco_docs is a list of documents, we create a list of stanza documents
docs_stanza = [stanza.Document([], text=doc) for doc in meco_docs]

# process each document using stanza
en_data_docs = stanza_tokenizer_v2.custom_tokenizer_lemmatizer_preserving_documents(docs_stanza, 
                                                              'en', 
                                                              'tokenize, pos, lemma')

# lowercase (only for En, Hi does not have capitalization)
en_data_lower = []
for en_data_lemmas in en_data_docs:
    en_data_lower.append([[lemma.lower() for lemma in sent] for sent in en_data_lemmas])

# remove punctuation
en_data_no_punct = []
for en_data_lemmas in en_data_lower:
    en_data_no_punct.append([[lemma.translate(str.maketrans('', '', string.punctuation)) for lemma in sent] for sent in en_data_lemmas])

# remove empty strings left over by removing the punctuation
en_data_clean = []
for en_data_lemmas in en_data_no_punct:
    en_data_clean.append([list(filter(None, lemmas)) for lemmas in en_data_lemmas])

# remove 's' words
meco_lemmas = []
for docs in en_data_clean:
    
    for sent in docs:
        current_sent = []
        for wrd in sent:
            if wrd != 's':
                current_sent.append(wrd)
        meco_lemmas.append(current_sent)

# SAVE THE MECO LEMMAS
with open("meco_lemmas.txt", 'w', encoding='utf-8') as f:
        json.dump(meco_lemmas, f)

# Flatten the list
en_flat = []
for lst in en_data_clean:
    en_flat.extend([word for sent in lst for word in sent])
en_flat.remove('s')


en_meco_df = pd.read_csv("df_en_meco.csv")

# en_meco_df['IncOne'] = (en_meco_df["participant"]==en_meco_df["participant"].shift())
# en_meco_df['IncOne'] = (
#     np.where(en_meco_df.IncOne, 
#         np.where( en_meco_df['total_word_idx'].eq(en_meco_df['total_word_idx'].shift()+1), 
#                   'True' , en_meco_df['total_word_idx']-en_meco_df['total_word_idx'].shift() ),
#     ''))

# remove words with NA values
en_meco_df = en_meco_df[en_meco_df['word'].notna()]

df_for_merging_en = en_meco_df[en_meco_df['participant'] == 3][['total_word_idx', 'text_id', 'sent_id_and_idx', 'word_idx', 'word']]
df_for_merging_en['processed_word'] = en_flat
df_for_merging_en.to_csv('en_merged.csv')


# GET ALL HINDI-POTSDAM WORDS PREPROCESSED
with open("hindi1_words.txt", 'r', encoding='utf-8') as f:
    hindi1_documents = f.read()


# split sentences based on new paragraphs
hindi_docs = hindi1_documents.split('\n')[:-1]

# hindi_docs is a list of sentences, we create a list of stanza documents
docs_stanza = [stanza.Document([], text=doc) for doc in hindi_docs]

# process each document using stanza
hi_data_lemmas = stanza_tokenizer_v2.custom_tokenizer_lemmatizer_preserving_sentences(docs_stanza, 
                                                              'hi', 
                                                              'tokenize, pos, lemma')

list_of_punctuation_to_remove = string.punctuation + '।'
# remove punctuation
hi_data_lemmas = [[lemma.translate(str.maketrans('', '', list_of_punctuation_to_remove)) for lemma in sent] for sent in hi_data_lemmas]

# remove empty strings left over by removing the punctuation
hi_data_lemmas = [list(filter(None, lemmas)) for lemmas in hi_data_lemmas]

hi_potsdam_flat = [word for sent in hi_data_lemmas for word in sent]

hi_potsdam_df = pd.read_csv("df_hi_potsdam.csv")

df_for_merging_hi = hi_potsdam_df[hi_potsdam_df['participant'] == 1][['sent_id_and_idx', 'word_idx', 'word']]
df_for_merging_hi['processed_word'] = pd.Series(hi_potsdam_flat)
df_for_merging_hi.to_csv('hi_merged.csv')

# SAVE THE POTSDAM LEMMAS
with open("potsdam_lemmas.txt", 'w', encoding='utf-8') as f:
        json.dump(hi_data_lemmas, f)
