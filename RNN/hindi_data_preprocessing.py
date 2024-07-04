import string
import joblib
import json
import stanza
import stanza_tokenizer_v2
import re
import datetime

# hi model was not yet downloaded on the servers, so execute this line once:
#stanza.download('hi')

start = datetime.datetime.now()
start1 = datetime.datetime.now()
# load the data
with open('hiwiki_final.txt', 'r', encoding='utf-8') as f:
    hi_data = f.read()
print("Done with reading Hindi:", datetime.datetime.now() - start) # 4 seconds

#---------------------------DATA PREPROCESSING-------------------------------#
start = datetime.datetime.now()
# # remove newline characters: 
# hi_data = hi_data.replace('\n', ' ')
# print("Done with removing newlines:", datetime.datetime.now() - start) # 2 seconds

# n = 1000000
# hi_data_subset = hi_data[:n]

# split documents based on new paragraphs
hi_data = hi_data.split('\n\n')

# remove newline characters
hi_data = [text.replace('\n', ' ') for text in hi_data]

# remove page title
hi_data = [re.sub("[^:]+: ", "", text, 1) for text in hi_data]

# en_data is a list of documents, we create a list of stanza documents
docs_stanza = [stanza.Document([], text=doc) for doc in hi_data]

# gets first n articles
n = round(len(docs_stanza) / 8) # 12830 articles, len(docs_stanza) = 128296
docs_stanza_subset = docs_stanza

# tokenization, POS tags and lemmatization is done by stanza using a pipeline
# additionally, hindi also receives the mwt processor in the pipeline
start = datetime.datetime.now()
hi_data_lemmas = stanza_tokenizer_v2.custom_tokenizer_lemmatizer_preserving_sentences(docs_stanza_subset, 
                                                       'hi', 
                                                       'tokenize, pos, lemma')
print("Done with lemmatizing Hindi:", datetime.datetime.now() - start) # 19 seconds

list_of_punctuation_to_remove = string.punctuation + '।'
# remove punctuation
hi_data_lemmas = [[lemma.translate(str.maketrans('', '', list_of_punctuation_to_remove)) for lemma in sent] for sent in hi_data_lemmas]

# remove empty strings left over by removing the punctuation
hi_data_lemmas = [list(filter(None, lemmas)) for lemmas in hi_data_lemmas]
# save the data - flattens the list unfortunately, but json works
joblib.dump(hi_data_lemmas, 'hi_data_lemmas.sav')

# code for loading the data
# hi_preprocessed = joblib.load('hi_lemma_list_first_200k.sav')

# alternative way for saving the data - as a plaintext file
with open('hi_data_lemmas.txt', 'w', encoding='utf-8') as f:
    json.dump(hi_data_lemmas, f)

print("All done within: ", datetime.datetime.now() - start1)
# code for loading the data
# with open('hi_lemma_list_test_using_json_sentences.txt', 'r') as filehandle:
#     hi_preprocessed_2 = json.load(filehandle)
