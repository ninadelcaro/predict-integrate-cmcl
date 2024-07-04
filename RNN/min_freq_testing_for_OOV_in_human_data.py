import json
import stanza
import stanza_tokenizer_v2
import string
import random
from collections import Counter
from myUtils import rare_words_to_unknown

# GET ALL MECO WORDS PREPROCESSED
with open("words_en_meco_manually_fixed.txt", 'r') as f:
    meco_documents = f.read()

# split documents based on new paragraphs
meco_docs = meco_documents.split('\n\n')[:-1]

# remove the \n from each document
meco_docs = [text.replace('\\n', ' ') for text in meco_docs]

# meco_docs is a list of documents, we create a list of stanza documents
docs_stanza = [stanza.Document([], text=doc) for doc in meco_docs]

# process each document using stanza
en_data_lemmas = stanza_tokenizer_v2.custom_tokenizer_lemmatizer_preserving_sentences(docs_stanza, 
                                                              'en', 
                                                              'tokenize, pos, lemma')

# lowercase (only for En, Hi does not have capitalization)
en_data_lemmas = [[lemma.lower() for lemma in sent] for sent in en_data_lemmas]

# remove punctuation
en_data_lemmas = [[lemma.translate(str.maketrans('', '', string.punctuation)) for lemma in sent] for sent in en_data_lemmas]

# remove empty strings left over by removing the punctuation
en_data_lemmas = [list(filter(None, lemmas)) for lemmas in en_data_lemmas]


# REMOVE RARE WORDS FROM PREPROCESSED WIKI DATA
with open("en_data.txt", 'r') as f:
    en_sentences = json.load(f)

random.seed(2342)
random.shuffle(en_sentences)

min_frequency = 100
en_sent = rare_words_to_unknown(en_sentences[:260190], min_frequency) # there will be 260190 sentences in hindi train and validation sets


# FIND OUT PERCENTAGE OF OOV WORDS IN MECO 
en_sentences_flat = [word for sent in en_sent for word in sent]
en_meco_flat = [word for sent in en_data_lemmas for word in sent]

number_lemmas_meco = len(en_meco_flat)

# create a word frequency dictionary
en_freq_dict = Counter(en_meco_flat)

en_set = set(en_sentences_flat)
en_meco_set = set(en_meco_flat)

word_intersection = en_set.intersection(en_meco_set)
words_only_in_meco = en_meco_set.difference(en_set)

meco_words_in_vocab = len(word_intersection)
wiki_en_size_of_vocab = len(en_set)

number_of_present_meco_words_en = 0
for word in en_freq_dict:
    if word in en_set:
        number_of_present_meco_words_en += en_freq_dict[word]
print(number_of_present_meco_words_en / number_lemmas_meco)
# RESULTS
# number of lemmas in meco: 2109
# number of unique words in meco: 787
# min_frequency = 0 -> 781 meco words in vocabulary, 181381 size of vocabulary, 99.5% preserved text
# min_frequency = 20 -> 715 meco words in vocabulary, 12588 size of vocabulary, 94.8% preserved text
# min_frequency = 24 -> 706 meco words in vocabulary, 11059 size of vocabulary, 94.4% preserved text
# min_frequency = 25 -> 700 meco words in vocabulary, 10738 size of vocabulary, 94.0% preserved text
# min_frequency = 30 -> 689 meco words in vocabulary, 9528 size of vocabulary, 93.1% preserved text
# min_frequency = 50 -> 658 meco words in vocabulary, 6769 size of vocabulary, 90.7% preserved text
# min_frequency = 100 -> 589 meco words in vocabulary, 4248 size of vocabulary, 86.4% preserved text
# the above is for a sample of 260190 sentences, below we do it for the whole data
# min_frequency = 0 -> 785 meco words in vocabulary, 553586 size of vocabulary
# min_frequency = 10 -> 781 meco words in vocabulary, 80171 size of vocabulary 
# min_frequency = 200 -> 700 meco words in vocabulary, 10374 size of vocabulary
# min_frequency = 500 -> 642 meco words in vocabulary, 5611 size of vocabulary


# SAVE THE CHOSEN SENTENCES AS THE ONES TO TRAIN AND TEST WITH 
with open("en_train_test_data.txt", 'w', encoding='utf-8') as f:
        json.dump(en_sentences[:289101], f)



# BEGIN TESTS FOR HINDI

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


# REMOVE RARE WORDS FROM PREPROCESSED WIKI DATA
with open("hi_data.txt", 'r') as f:
    hi_sentences = json.load(f)

random.seed(2342)
random.shuffle(hi_sentences)

min_frequency = 0
hi_sent = rare_words_to_unknown(hi_sentences[:260190], min_frequency) # there will be 260190 sentences in hindi train and validation sets


# FIND OUT PERCENTAGE OF OOV WORDS IN MECO 
hi_sentences_flat = [word for sent in hi_sent for word in sent]
hi_potsdam_flat = [word for sent in hi_data_lemmas for word in sent]

number_lemmas_potsdam = len(hi_potsdam_flat)

# create a word frequency dictionary
hi_freq_dict = Counter(hi_potsdam_flat)

hi_set = set(hi_sentences_flat)
hi_potsdam_set = set(hi_potsdam_flat)

word_intersection = hi_set.intersection(hi_potsdam_set)
words_only_in_potsdam = hi_potsdam_set.difference(hi_set)

potsdam_words_in_vocab = len(word_intersection)
wiki_hi_size_of_vocab = len(hi_set)

number_of_present_potsdam_words_hi = 0
for word in hi_freq_dict:
    if word in hi_set:
        number_of_present_potsdam_words_hi += hi_freq_dict[word]
print(number_of_present_potsdam_words_hi / number_lemmas_potsdam)

# RESULTS
# number of lemmas in Potsdam: 1199
# number of unique words in Potsdam: 398
# min_frequency = 0 -> 349 Potsdam words in vocabulary, 206393 size of vocabulary, 94.1% preserved text
# min_frequency = 10 -> 312 Potsdam words in vocabulary, 20267 size of vocabulary, 89.7% preserved text
# min_frequency = 20 -> 306 Potsdam words in vocabulary, 12555 size of vocabulary, 88.7% preserved text
# min_frequency = 24 -> 303 Potsdam words in vocabulary, 11076 size of vocabulary, 88.4% preserved text
# min_frequency = 50 -> 276 Potsdam words in vocabulary, 6600 size of vocabulary, 84.6% preserved text

