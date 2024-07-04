import os, sys, inspect
import random
import torch
import numpy as np
from collections import Counter
# Add src folder to path
SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
sys.path.insert(0, os.path.join(SCRIPT_FOLDER, os.pardir))
from s2i import String2IntegerMapper



def init_seed(seed):

    #set the seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

def sentences2indexs(sentences, beginning, ending, word_mappings):
    """
        Creates the dictionary of mappings and returns the mapped sentences.
        sentences: list of list of strings
        separator: string to separate sentences
    """
    
    all_sentences_indexed = []
    for i,sentence in enumerate(sentences):
        sentence_idxs = []
        #words = sentence.split(" ")
        sentence.insert(0, beginning)
        sentence.append(ending)
        for word in sentence:
            #Remove spaces at beginning or ending of words
            #word=word.strip()
            # Add word to mappings
            word_mappings.add_string(word)
            # Get index of the word
            idx = word_mappings[word]
            sentence_idxs.append(idx)
        all_sentences_indexed.append(sentence_idxs)

    return all_sentences_indexed

def prepare_sequences(all_sentences_indexed): #add delay as a parameter to this function
    input_seq, target_seq = [], []
    # Prepare input and target idx sequences (i.e remove last and first items, respectively)
    for i in range(len(all_sentences_indexed)):
        # Remove last item for input sequence (we don't need to predict anything after this one)
        input_seq.append(all_sentences_indexed[i][:-1])

        # Remove first item for target sequence
        target_seq.append(all_sentences_indexed[i][1:])


    return input_seq, target_seq


def prepare_tensors_sgd(inputs, targets, delay, device):
    """
    Prepare data for training of minibatch of size 1 ('SGD')
    We don't need padding/packing.

    :param inputs
    :param targets
    :return:
    """

    inputs_t, targets_t = [], []
    for i in range(len(inputs)): #for each sentence
        sentence_length=len(inputs[i])
        input_t = torch.zeros((sentence_length + delay,), dtype=torch.long, device=device)
        input_t[:sentence_length] = torch.tensor(inputs[i], device=device)

        inputs_t.append(input_t)#convert sentence to tensor
        targets_t.append(torch.tensor(targets[i], device=device))
    return inputs_t, targets_t

def prepare_tensors_sgd_with_backward_direction(inputs, targets, delay, device):
    """
    Prepare data for training of minibatch of size 1 ('SGD')
    We don't need padding/packing.

    :param inputs
    :param targets
    :return:
    """

    inputs_t, targets_t = [], []
    for i in range(len(inputs)): #for each sentence
        sentence_length=len(inputs[i])
        input_t = torch.zeros((sentence_length*2 - 1 + delay,), dtype=torch.long, device=device)
        # first part of the input is just the sentence
        input_t[:sentence_length] = torch.tensor(inputs[i], device=device)
        # second part of the input is the sentence going backwards
        input_t[:sentence_length] = torch.tensor(inputs[i], device=device)

        inputs_t.append(input_t)#convert sentence to tensor
        targets_t.append(torch.tensor(targets[i], device=device))
    return inputs_t, targets_t


def rare_words_to_unknown(sentences, min_frequency):
    
    # flatten the sentences list
    sentences_flat = [word for sent in sentences for word in sent]
    
    # create a word frequency dictionary
    freq_dict = Counter(sentences_flat)
    
    # save only words that appear more than min_frequency times
    frequent_enough_words = {k:v for (k,v) in freq_dict.items() if v >= min_frequency}
    
    # change the rare words in sentences to UNK
    processed_sentences = [[word if word in frequent_enough_words else '<UNK>' for word in sent]for sent in sentences]

    return processed_sentences


def input_to_indexs(sentences, delay, min_frequency):


    # Create mapping dictionary
    word_mappings = String2IntegerMapper()

    # Define null, and beginning and ending sentence symbol
    beginning = "<s>"
    ending = "</s>"
    null = "<NULL>"

    #Add to dictionary
    word_mappings.add_string(null) #we reserve 0 for null)
    word_mappings.add_string(beginning)
    word_mappings.add_string(ending)

    processed_sentences = rare_words_to_unknown(sentences, min_frequency)
    
    # Convert sentences into word indexes
    all_sentences_indexed = sentences2indexs(processed_sentences, beginning, ending, word_mappings)

    return word_mappings, all_sentences_indexed

