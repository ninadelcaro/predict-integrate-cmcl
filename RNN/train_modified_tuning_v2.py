# from Raquel
import sys, os, inspect
from os.path import join
import argparse
import time
import torch
import numpy as np
import random
import math
import json
import matplotlib.pyplot as plt
#SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
#    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
#PROJECT_FOLDER = join(SCRIPT_FOLDER, os.pardir)
#sys.path.insert(0, PROJECT_FOLDER)
from rnn import RNNModel
from convergence import ConvergenceCriterion, ConvergenceCriteria
from myUtils import prepare_tensors_sgd, input_to_indexs, init_seed, prepare_sequences, rare_words_to_unknown, sentences2indexs


# TODO: check if convergence criteria needs changing

def log_cuda_usage(fh):

    fh.write("Using cuda: ")
    fh.write(torch.cuda.get_device_name(0))
    fh.write('\nMemory Usage:\n')
    fh.write('Allocated: %f GB\n'%round(torch.cuda.memory_allocated(0)/1024**3,1))
    fh.write('Cached:   %f GB\n'%round(torch.cuda.memory_cached(0)/1024**3,1))


def train_and_tune(hyperparameters, train_set, validation_set, device, dict_size):

    # transform train_set into list(list(int)) where int are 
    # encoded words, and list(int) are sentences
    word_mappings, train_sentences_indexed = input_to_indexs(train_set, args.delay, args.min_frequency)
    
    # initialize the model and send it to the device
    rnn = RNNModel(hyperparameters, device)
    rnn.to(device)

    #Instantiate an optimizer
    optimizer = torch.optim.SGD(rnn.parameters(), lr=hyperparameters["lr"])

    #Create a loss function
    loss_function = torch.nn.NLLLoss()

    # prepare validation set in the same way the test set would be
    # set out of vocabulary words to unknown token
    processed_valid_sentences = [[word if word in word_mappings.s2i else '<UNK>' for word in sent]for sent in validation_set]
    # set rare words to unknown token
    rare_words_removed_valid_sentences = rare_words_to_unknown(processed_valid_sentences, args.min_frequency)
    # Convert sentences to indexes
    beginning = "<s>"
    ending = "</s>"
    valid_sentences_indexed = sentences2indexs(rare_words_removed_valid_sentences, beginning, ending, word_mappings)
    inputs_valid, targets_valid = prepare_sequences(valid_sentences_indexed)
    # make the input and targets be tensors
    inputs_valid_t, targets_valid_t = prepare_tensors_sgd(inputs_valid, targets_valid, args.delay, device)

    #We update after every sentence
    # batch_size = 1
    
    #Initialize other vars
    options = {}
    options["max_epochs_stagnated"] = 10
    # min_change = options["min_change"] = 0.0025
    options["min_change"] = 0.0025
    criteria = [ConvergenceCriterion("loss", options)]
    convergence_criteria = ConvergenceCriteria(criteria, "or")
    
    # empty lists to store the losses of the train and validation sets per epoch
    loss_all_epochs = []
    loss_valid_all_epochs = []
    
    converged=False
    epoch=0

    #Start training - each iteration is an epoch
    while not converged:
        #print("another epoch")
        epoch+=1
        be=time.perf_counter()

        # running_loss = 0.0

        #Shuffle the sentences so that every epoch gets presented
        # with sentences in a diffferent order 
        random.shuffle(train_sentences_indexed)

        # Get sequences of inputs and targets
        inputs, targets = prepare_sequences(train_sentences_indexed)

        # Get tensors of inputs and targets, and add delay if required
        inputs_t, targets_t = prepare_tensors_sgd(inputs, targets, args.delay, device)

        losses_sentences = []

        #We present each sentence separately
        for i, input in enumerate(inputs_t):
            target = targets_t[i]

            # Clean the gradient data so that it doesn't accumulate (SGD)
            optimizer.zero_grad()

            #Run forward pass for this sentence
            #output shape: sentence_length*vocabulary_size
            output, hidden = rnn.forward(input.unsqueeze(0)) #we add the batch dimension (which is always 1)

            #Compute the loss
            #the loss function expects dimensions n_inputs * output_size
            output=output.squeeze(0) #we remove the batch dimension
            if args.delay > 0:
                output = output[args.delay:, :] #shift: we ignore the first outputs (delay)

            loss = loss_function(output, target)

            # Backprop (compute the gradient)
            loss.backward()

            # Update the weights of the neural network
            optimizer.step()
            
            # keep track of loss values to plot them
            # running_loss += loss.item() * target.size(0) #target?output? #RGA it's the same

            losses_sentences.append(loss.item())

        #losses_sentences=train_step_sgd(rnn, optimizer, loss_function, inputs_t, targets_t)

        loss_this_epoch = np.mean(losses_sentences)
        loss_all_epochs.append(loss_this_epoch)

        losses_valid_sentences = []
        
        # present each sentence separately
        for i, input in enumerate(inputs_valid_t):
            with torch.no_grad():
                target_words = targets_valid_t[i]
                output, hidden = rnn.forward(input.unsqueeze(0))
                output = output.squeeze(0)  # we remove the batch dimension
                
                if args.delay > 0:
                    output = output[args.delay:, :]  # shift: we ignore the first outputs (delay)

                loss_valid = loss_function(output, target_words)

                losses_valid_sentences.append(loss_valid.item())

        loss_valid_epoch = np.mean(losses_valid_sentences)
        loss_valid_all_epochs.append(loss_valid_epoch)
        
        #Check whether it's time to stop training
        convergence_criteria.update_state(epoch, loss_all_epochs, None, None, None)
        converged=convergence_criteria.converged()

        time_epoch=time.perf_counter()-be

        #Report
        line="Epoch: {}   Loss: {:.4f} Time: {:.3f}\n".format(epoch, loss_this_epoch, time_epoch)
        with open(flog, "a") as fh:
            fh.write(line)
#            if device.type == 'cuda':
#                log_cuda_usage(fh)

        if epoch%10 == 0:
            print(line)

        if epoch%5 == 0:
            print("Saving intermediate model at epoch %i..."%epoch)
            #model_folder = rnn.save_model(path_to_saved_models, word_mappings, epoch, args)

    #Save the final model
    #model_folder = rnn.save_model(path_to_saved_models, word_mappings, epoch, args)
    #print("Model saved at %s"%model_folder)
    return loss_all_epochs[-1], loss_valid_all_epochs[-1], epoch


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",  type=int, help="Seed for training neural network.")
    parser.add_argument("--language",  type=str, help="Language to train on.")
    parser.add_argument("--delay",  type=int, help="If smaller than 1, then normal RNN. Otherwise, delayed-RNN.")
    parser.add_argument("--min_frequency",  type=int, help="Words with a frequency smaller than this are set to unknown.")
    parser.add_argument("--hidden_dim",  type=int, help="Size of hidden layer.")
    parser.add_argument("--embedding_dim", type=int, help="Size of word embeddings.")
    parser.add_argument("--lr",  type=float, help="Learning rate for training neural network.")
    parser.add_argument("--input_data", type=str, help="File with sentences to train on.")
    #ToDo comment this line for safer mode (it's just useful for debugging purposes)
    #parser.set_defaults(seed=8, language="eng", delay=0, hidden_dim=250, embedding_dim=100, lr=0.01, input_data="../data/eng_0_60_cds_utterances_cleaned_first_10.txt")
    args = parser.parse_args()

    #Check parameters
    assert(args.language in args.input_data)

    #Initialize the seed
    init_seed(args.seed)

    #Set gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    with open(args.input_data, 'r') as f:
        sentences = json.load(f)

    # shuffle the sentences to obtain random sentences from different articles
    random.shuffle(sentences)

    # get the train, validation and test set
    test_set = sentences[round(len(sentences) * 0.9):]
    validation_set = sentences[round(len(sentences) * 0.8):round(len(sentences) * 0.9)]
    train_set = sentences[:round(len(sentences) * 0.8)]

    # save test set to use with test file
    test_filename = "final_test_set_%s_%i_delay_%i_seed_%i_min_freq.txt"%(args.language, args.delay, args.seed, args.min_frequency)
    with open(test_filename, 'w', encoding='utf-8') as f:
        json.dump(test_set, f)

    #Prepare folder to save trained models
    #path_to_saved_models=join(PROJECT_FOLDER, "saved_models")
    path_to_saved_models = 'saved_models'
    #Set logging file
    flog="training_log_%s_%idelay_seed_%i"%(args.language, args.delay, args.seed)
    with open(flog,"w") as fh:
        fh.write("Training on file {}\n".format(args.input_data))
        if device.type == 'cuda':
            log_cuda_usage(fh)

    # prepare file for keeping track of hyperparameters and losses
    with open("hyperparameter_values_%s_%i_delay_%i_seed_%i_min_freq.txt"%(args.language, args.delay, args.seed, args.min_frequency),"w") as f:
        pass
    
    #Prepare input data (mappings and lists of indices)
    word_mappings, all_sentences_indexed = input_to_indexs(sentences, args.delay, args.min_frequency)
    dict_size = len(word_mappings.s2i)

    #We update after every sentence
    batch_size = 1

    #TODO: make the grid random
    hidden_dim = [400, 600] #[int(random.uniform(200, 700)) for i in range(2)] # original (200, 2000)
    embedding_dim = [250, 350]#[int(random.uniform(200, 500)) for i in range(2)] # original (90, 1500)
    n_rnn_layers = [1]
    lr = [0.001] # original: random.sample([0.1, 0.01, 0.001, 0.0001], 2) 

    grid_params = []    
    for i in hidden_dim:        
        for j in embedding_dim: 
            for l in n_rnn_layers:
                for m in lr:
                    grid_params.append({"hidden_dim": i, 
                                        "embedding_dim": j, 
                                        "n_rnn_layers": l, 
                                        "lr": m})
    
    losses_train = []
    losses_valid = []
    for hyp in grid_params:
        print("one setting:", hyp)
        hyp['output_size'] = dict_size
        loss_train, loss_valid, epochs = train_and_tune(hyp, train_set, validation_set, device, dict_size)
        losses_train.append(loss_train)
        losses_valid.append(loss_valid)
        with open("hyperparameter_values_%s_%i_delay_%i_seed_%i_min_freq.txt"%(args.language, args.delay, args.seed, args.min_frequency), "a") as f:
            f.write("Train loss: {}   Validation loss: {}   Hyperparameters: {}   Epochs: {}\n".format(loss_train, loss_valid, hyp, epochs))

    # find best model according to loss on val and according to perplexity 
    best_loss_idx = losses_valid.index(min(losses_valid))
    print("Best validation loss and the hyperparameters, training loss, and num of epochs: ", min(losses_valid), grid_params[best_loss_idx], losses_train[best_loss_idx], epochs)
    
    # #Plot the losses
    # plt.plot(loss_all_epochs)
    # plt.ylabel("Loss")
    # plt.xlabel("Epoch")
    # plt.title("Train losses {}".format(args.language))
    # plt.savefig(join(model_folder, "loss.{}.png".format(args.language)))


