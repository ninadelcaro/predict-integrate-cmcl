
import string
import re
import joblib
import json
import stanza
import stanza_tokenizer_v2


with open('enwiki-sample_wikiextractor.txt', 'r', encoding='utf-8') as f:
    en_data = f.read()

#---------------------------DATA PREPROCESSING-------------------------------#
# split documents based on new paragraphs
en_data = en_data.split('\n\n')

# remove newline characters
en_data = [text.replace('\n', ' ') for text in en_data]

# remove page title
en_data = [re.sub("[^:]+: ", "", text, 1) for text in en_data]

# en_data is a list of documents, we create a list of stanza documents
docs_stanza = [stanza.Document([], text=doc) for doc in en_data]

# gets first n articles
n = round(len(docs_stanza) / 10) # 12830 articles, len(docs_stanza) = 128296
docs_stanza_subset = docs_stanza[:]

# tokenization, POS tags and lemmatization is done by stanza using a pipeline
# additionally, hindi also receives the mwt processor in the pipeline
en_data_lemmas = stanza_tokenizer_v2.custom_tokenizer_lemmatizer_preserving_sentences(docs_stanza_subset, 
                                                              'en', 
                                                              'tokenize, pos, lemma')

# lowercase (only for En, Hi does not have capitalization)
en_data_lemmas = [[lemma.lower() for lemma in sent] for sent in en_data_lemmas]

# remove punctuation
en_data_lemmas = [[lemma.translate(str.maketrans('', '', string.punctuation)) for lemma in sent] for sent in en_data_lemmas]

# remove empty strings left over by removing the punctuation
en_data_lemmas = [list(filter(None, lemmas)) for lemmas in en_data_lemmas]

# save the data
joblib.dump(en_data_lemmas, 'en_lemma_list_all_sentences.sav')

# code for loading the data
#en_preprocessed = joblib.load('en_lemma_list_first_200k.sav')

# alternative way for saving the data - as a plaintext file
with open('en_lemma_list_all_using_json_sentences.txt', 'w', encoding='utf-8') as f:
    json.dump(en_data_lemmas, f)
    
# code for loading the data
# with open('en_lemma_list_first_200k_using_json_sentences.txt', 'r') as filehandle:
#     basic_list = json.load(filehandle)
