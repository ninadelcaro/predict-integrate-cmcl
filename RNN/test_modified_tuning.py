import os, sys, inspect, argparse
from os.path import join, isfile
from os import remove
import json
import torch
import pandas as pd

# Add src folder to path
# SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
#     os.path.split(inspect.getfile(inspect.currentframe()))[0]))
# sys.path.insert(0, join(SCRIPT_FOLDER, os.pardir))
from rnn import RNNModel
from s2i import String2IntegerMapper
#from tagger import Tagger
from metrics import get_perword_metrics_sentence
from myUtils import prepare_tensors_sgd, prepare_sequences, sentences2indexs, init_seed, rare_words_to_unknown

# TODO: change the POS tagger 

def load_saved_model(path, device):
    # Load hyperparameters
    hyperparams = json.load(open(join(path, "hyperparams.json")))

    # Create base model
    rnn = RNNModel(hyperparams, device)
    rnn.to(device)

    # Load saved parameters (weights)
    params_state_dict = torch.load(join(path, "model"), map_location=device)
    missing_keys = rnn.load_state_dict(params_state_dict)

    return rnn

# def sentences2indexs_clean(sentences, beginning, ending, word_mappings):
#     """
#         Creates the dictionary of mappings and returns the mapped sentences.
#         sentences: list of list of strings
#         separator: string to separate sentences
#     """
    
#     all_sentences_indexed = []
#     for i,sentence in enumerate(sentences):
#         sentence_idxs = []
#         #words = sentence.split(" ")
#         sentence.insert(0, beginning)
#         sentence.append(ending)
#         for word in sentence:
#             #Remove spaces at beginning or ending of words
#             #word=word.strip()
#             # if word in word_mappings:
#             if word_mappings[word] != None:
#                 idx = word_mappings[word]
#                 sentence_idxs.append(idx)
#         all_sentences_indexed.append(sentence_idxs)

#     return all_sentences_indexed

def forward_test(lang):
    header = "lang;previous_word;actual_word;predicted_word;correct;entropy;entropytop10;surprisal;target_in_top10"
    perplexities = []
    for i, input in enumerate(inputs_t):
        target_words = targets_t[i]
        # print("Testing sentence %i: %s" % (i, sentences[i]))
        #print("Input ", input)
        #print("Target ", target_words)
        # output size: n_words_sentence x n_words_vocabulary
        output, hidden = model.forward(input.unsqueeze(0))
        output = output.squeeze(0)  # we remove the batch dimension
        
        if args.delay > 0:
            output = output[args.delay:, :]  # shift: we ignore the first outputs (delay)
        words = torch.argmax(output, axis=1)
        #print("Model Output ", words)
        
        perplexity_per_sentence = torch.nn.NLLLoss()(output, target_words)
        perplexities.append(perplexity_per_sentence)

        sentence = sentences[i]

        # Get proportion of correct word predictions (argmax) in a sentence, and other metrics
        sentence_metrics = get_perword_metrics_sentence(output, target_words, sentence)

        # Save results
        # print(sentence_metrics)
        # lens = [len(x) for x in sentence_metrics.values()]
        # if (len(set(lens)) != 1):
        #     print("Problem with sentence ", sentence, lens)
        
        sentence_metrics["lang"]=lang
        sentence_metrics["perplexity_per_sentence"] = torch.exp(perplexity_per_sentence).item()
        partialdf = pd.DataFrame.from_dict(sentence_metrics)
        partialdf.to_csv(outputf, mode='a', header=header, index=False)
        header = False

    perplexity_aggregated = torch.exp(torch.stack(perplexities).mean())
    print("Final perp: ", perplexity_aggregated)
    print("Results saved in %s"%outputf)

if __name__ == "__main__":

    # print(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Seed for training neural network.")
    parser.add_argument("--language", type=str, help="Language to train on.")
    parser.add_argument("--delay", type=int, help="If smaller than 1, then normal RNN. Otherwise, delayed-RNN.")
    parser.add_argument("--min_frequency",  type=int, help="Words with a frequency smaller than this are set to unknown.")
    parser.add_argument("--path_to_model", type=str, help="Saved model.")
    parser.add_argument("--input_data", type=str, help="File with sentences to test on.")
    # parser.add_argument("--train_set", type=str, help="File with sentences it was trained on.")
    
    # comment this line (this is just for debugging)
    # parser.set_defaults(seed=84, language="eng", delay=2, direction="forward", input_data="../data/eng_0_60_cds_utterances_cleaned_first_2.txt", path_to_model="../saved_models/model_eng_forward_2delay_seed_84_epoch_480")

    args = parser.parse_args()

    # Check command line options
    assert (args.language in args.input_data)

    # Set output file and remove if already exists
    outputf = join(args.path_to_model, 'accuracies_test.csv')
    if isfile(outputf):
        remove(outputf)

    # Initialize the seed
    init_seed(args.seed)

    # Set gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data to test (for now let's just take any of the small files we have)
    with open(args.input_data, 'r') as f:
        sentences = json.load(f)
    
    
    # # TEMPORARY:
    # sentences = [["my", 'name', 'is', 'nina'], ["name", "is", "bond"]]
    # # with open(args.train_set, 'r') as f:
    # #     train_sentences = json.load(f)
    # # # get the test set
    # # #train_sentences = sentences[:round(len(sentences) * 0.8)]
    # # train_vocab = []
    # # for sent in train_sentences:
    # #     for word in sent:
    # #         if word not in train_vocab:
    # #             train_vocab.append(word)
    # # test_sentences = sentences[round(len(sentences) * 0.995) + 1 :]
    # # sentences = [[word for word in sent if word in train_vocab]for sent in test_sentences]


   
    # Load saved model
    model = load_saved_model(args.path_to_model, device)
    # Load string to int mappers
    mappings = String2IntegerMapper.load(join(args.path_to_model, "w2i"))
    
    # set out of vocabulary words to unknown token
    processed_sentences = [[word if word in mappings.s2i else '<UNK>' for word in sent]for sent in sentences]
    
    # do the following only for Wiki data, not for data accompanying eyetracking measures
    rare_words_removed_sentences = rare_words_to_unknown(processed_sentences, args.min_frequency)
    sentences = rare_words_removed_sentences

    # Convert sentences to indexs
    beginning = "<s>"
    ending = "</s>"
    all_sentences_indexed = sentences2indexs(sentences, beginning, ending, mappings)
    inputs, targets = prepare_sequences(all_sentences_indexed)
    # Put them as tensors
    inputs_t, targets_t = prepare_tensors_sgd(inputs, targets, args.delay, device)

    # Run one forward pass of the model with the data, and collect the output
    forward_test(args.language)
