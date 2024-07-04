import json
import random
from os.path import join
from myUtils import init_seed, rare_words_to_unknown
from s2i import String2IntegerMapper

with open("en_train_test_data.txt", 'r') as f:
    en_sentences = json.load(f)

with open("hi_data.txt", 'r') as f:
    hi_sentences = json.load(f)
    
# shuffle the sentences to obtain random sentences from different articles

#Initialize the seed
init_seed(2342)
random.shuffle(en_sentences)

#Initialize the seed
init_seed(2342)
random.shuffle(hi_sentences)


# get the train set and test sets
en_test_set = en_sentences[round(len(en_sentences) * 0.9):]
en_train_set = en_sentences[:round(len(en_sentences) * 0.9)]
hi_test_set = hi_sentences[round(len(hi_sentences) * 0.9):]
hi_train_set = hi_sentences[:round(len(hi_sentences) * 0.9)]

# Load string to int mappers
hi_mappings = String2IntegerMapper.load(join("final_models/model_hi_0delay_seed_2342_epoch_45", "w2i"))
en_mappings = String2IntegerMapper.load(join("final_models/model_en_0delay_seed_2342_epoch_41", "w2i"))

en_test_set = [[word if word in en_mappings.s2i else '<UNK>' for word in sent]for sent in en_test_set]
en_train_set = [[word if word in en_mappings.s2i else '<UNK>' for word in sent]for sent in en_train_set]
hi_test_set = [[word if word in hi_mappings.s2i else '<UNK>' for word in sent]for sent in hi_test_set]
hi_train_set = [[word if word in hi_mappings.s2i else '<UNK>' for word in sent]for sent in hi_train_set]

# do the following only for Wiki test data, not for data accompanying eyetracking measures
en_test_set = rare_words_to_unknown(en_test_set, 24)
hi_test_set = rare_words_to_unknown(hi_test_set, 24)


with open("en_meco_lemmas.txt", 'r') as f:
    en_meco = json.load(f)

with open("hi_potsdam_lemmas.txt", 'r') as f:
    hi_potsdam = json.load(f)


en_meco = [[word if word in en_mappings.s2i else '<UNK>' for word in sent]for sent in en_meco]
hi_potsdam = [[word if word in hi_mappings.s2i else '<UNK>' for word in sent]for sent in hi_potsdam]

# number of sentences
print("Number of sentences:")
print("English train set: ", len(en_train_set))
print("English test set: ", len(en_test_set))
print("English eye-tracking set: ", len(en_meco))
print("Hindi train set: ", len(hi_train_set))
print("Hindi test set: ", len(hi_test_set))
print("Hindi eye-tracking set: ", len(hi_potsdam))

# number of lemmas in total
en_train_flat = [word for sent in en_train_set for word in sent]
en_test_flat = [word for sent in en_test_set for word in sent]
en_meco_flat = [word for sent in en_meco for word in sent]
hi_train_flat = [word for sent in hi_train_set for word in sent]
hi_test_flat = [word for sent in hi_test_set for word in sent]
hi_potsdam_flat = [word for sent in hi_potsdam for word in sent]

print("-----------------------------------------------------------")
print("Number of lemmas/tokens:")
print("English train set: ", len(en_train_flat))
print("English test set: ", len(en_test_flat))
print("English eye-tracking set: ", len(en_meco_flat))
print("Hindi train set: ", len(hi_train_flat))
print("Hindi test set: ", len(hi_test_flat))
print("Hindi eye-tracking set: ", len(hi_potsdam_flat))

# number of unique lemmas
en_train_unique = set(en_train_flat)
en_test_unique = set(en_test_flat)
en_meco_unique = set(en_meco_flat)
hi_train_unique = set(hi_train_flat)
hi_test_unique = set(hi_test_flat)
hi_potsdam_unique = set(hi_potsdam_flat)

print("-----------------------------------------------------------")
print("Number of unique lemmas/tokens - vocabulary size:")
print("English train set: ", len(en_train_unique))
print("English test set: ", len(en_test_unique))
print("English eye-tracking set: ", len(en_meco_unique))
print("Hindi train set: ", len(hi_train_unique))
print("Hindi test set: ", len(hi_test_unique))
print("Hindi eye-tracking set: ", len(hi_potsdam_unique))

# mean sentence length
en_train_mean_sent_len = sum([len(sent) for sent in en_train_set]) / len(en_train_set)
en_test_mean_sent_len = sum([len(sent) for sent in en_test_set]) / len(en_test_set)
en_meco_mean_sent_len = sum([len(sent) for sent in en_meco]) / len(en_meco)
hi_train_mean_sent_len = sum([len(sent) for sent in hi_train_set]) / len(hi_train_set)
hi_test_mean_sent_len = sum([len(sent) for sent in hi_test_set]) / len(hi_test_set)
hi_potsdam_mean_sent_len = sum([len(sent) for sent in hi_potsdam]) / len(hi_potsdam)

print("-----------------------------------------------------------")
print("Mean sentence length:")
print("English train set: ", en_train_mean_sent_len)
print("English test set: ", en_test_mean_sent_len)
print("English eye-tracking set: ", en_meco_mean_sent_len)
print("Hindi train set: ", hi_train_mean_sent_len)
print("Hindi test set: ", hi_test_mean_sent_len)
print("Hindi eye-tracking set: ", hi_potsdam_mean_sent_len)

# range of sentence length
en_train_range_sent_len = [min([len(sent) for sent in en_train_set]), max([len(sent) for sent in en_train_set])]
en_test_range_sent_len = [min([len(sent) for sent in en_test_set]), max([len(sent) for sent in en_test_set])]
en_meco_range_sent_len = [min([len(sent) for sent in en_meco]), max([len(sent) for sent in en_meco])]
hi_train_range_sent_len = [min([len(sent) for sent in hi_train_set]), max([len(sent) for sent in hi_train_set])]
hi_test_range_sent_len = [min([len(sent) for sent in hi_test_set]), max([len(sent) for sent in hi_test_set])]
hi_potsdam_range_sent_len = [min([len(sent) for sent in hi_potsdam]), max([len(sent) for sent in hi_potsdam])]

print("-----------------------------------------------------------")
print("Range of sentence length:")
print("English train set: ", en_train_range_sent_len)
print("English test set: ", en_test_range_sent_len)
print("English eye-tracking set: ", en_meco_range_sent_len)
print("Hindi train set: ", hi_train_range_sent_len)
print("Hindi test set: ", hi_test_range_sent_len)
print("Hindi eye-tracking set: ", hi_potsdam_range_sent_len)

# type-token ratios
en_train_ttr = len(en_train_unique) / len(en_train_flat)
en_test_ttr = len(en_test_unique) / len(en_test_flat)
en_meco_ttr = len(en_meco_unique) / len(en_meco_flat)
hi_train_ttr = len(hi_train_unique) / len(hi_train_flat)
hi_test_ttr = len(hi_test_unique) / len(hi_test_flat)
hi_potsdam_ttr = len(hi_potsdam_unique) / len(hi_potsdam_flat)

print("-----------------------------------------------------------")
print("Type-token ratios:")
print("English train set: ", en_train_ttr)
print("English test set: ", en_test_ttr)
print("English eye-tracking set: ", en_meco_ttr)
print("Hindi train set: ", hi_train_ttr)
print("Hindi test set: ", hi_test_ttr)
print("Hindi eye-tracking set: ", hi_potsdam_ttr)