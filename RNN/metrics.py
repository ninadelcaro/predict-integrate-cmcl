import numpy as np
import torch


def pred_to_prob(preds):
    '''
    :param preds: prediction from forward step (output tensor from logsoftmax)
                  shape: nwords, vocabulary_size
    :return: numpy vector with probabilities (not log probs)
    '''
    probs = np.exp(np.array(preds.data.cpu()))
    return probs


def targets_probs(predicted_probs, targets):
    """
        Returns the probability of the targets.
    """
    try:
        probs_targets = predicted_probs[np.arange(len(predicted_probs)), targets.cpu()]
    except IndexError:
        probs_targets = predicted_probs[targets.cpu()]
    return probs_targets


def get_n_correct(output, targets):
    """
        Returns a list of 0s and 1s indicating which of the target
        words were correctly predicted. 
    """

    # output shape: 1 x nwords_sentence x voc_Size
    predicted_probs = pred_to_prob(output)  # convert output to actual probabilities
    predicted_idxs = np.argmax(predicted_probs, axis=1) # get what words were predicted
    correct = predicted_idxs == np.array(targets.cpu())
    correct = [int(x) for x in correct]
    return correct


def target_in_topk(output, targets, k):
    # output shape: [batch x] nwords_sentence x voc_Size
    # targets shape: nwords_sentence
    predicted_probs = pred_to_prob(output)  # convert output to actual probabilities
    # get indices of words with higher probability mass
    topk = torch.topk(torch.from_numpy(predicted_probs), k)
    topk = topk.indices.numpy()
    targets = targets.cpu().numpy()
    correct = [int(targets[i] in topk[i]) for i in range(len(targets))]
    return correct


def compute_accuracy_sentence(output, target_words):
    correct = get_n_correct(output, target_words)
    accuracy_sentence = sum(correct) / len(correct)
    return accuracy_sentence * 100


def compute_entropy_output(outputs):
    # output is in log probs
    # output shape: [batch] x words_in_sentence x vocsize

    # turn log probs into probs
    # predicted_probs: numpy array, nwords_sentence x vocsize
    predicted_probs = pred_to_prob(outputs.squeeze(0))

    # compute entropy
    # #h = entropy(predicted_probs, axis=1)
    # equivalent implementation for our case:
    try: 
    	h = -np.sum(predicted_probs * np.log(predicted_probs), axis=1)
    except:
        h = -np.sum(predicted_probs * np.log(predicted_probs), axis=0)
    return h


def compute_surprisal_output(output, targets):
    # output is in log probs
    # output shape: [batch], words_in_sentence, vocsize

    # turn log probs into probs
    # predicted_probs: numpy array, nwords_sentence x vocsize
    predicted_probs = pred_to_prob(output.squeeze(0))

    # get probabilities of each target word
    probs_targets = targets_probs(predicted_probs, targets)

    # compute surprisal for each target
    surprisal = - np.log(probs_targets)

    return surprisal


def compute_entropy_over_topk(output, k):
    """
        :param outputs: the output of the forward pass (output size: n_words_sentence x n_words_vocabulary )
    """

    # predicted_probs: numpy array, nwords_sentence x vocsize
    predicted_probs = pred_to_prob(output.squeeze(0))

    # FZ: convert np array into tensor
    predicted_probs = torch.from_numpy(predicted_probs)

    # FZ: function torch.topk: Returns the k largest elements of the given input tensor along a given dimension.
    # If dim is not given, the last dimension of the input is chosen.
    topk_tensor = torch.topk(predicted_probs, k)

    # FZ: convert tensor into np array
    topk_np = topk_tensor.values.numpy()

    # entropy of top10
    try:
        h = -np.sum(topk_np * np.log(topk_np), axis=1)
    except:
        h = -np.sum(topk_np * np.log(topk_np), axis=0)
    return h


def get_perword_metrics_sentence(output, targets, sentence):
    """
    actual_word; next_word;correct_prediction; target_probability
    <s>;this; 0; 0.01
    this; is; 1; 0.8
    is; an; 1; 0.67
    example;</s>; 0; .4
    :param output: the output of the forward pass (output size: n_words_sentence x n_words_vocabulary )
    :param targets:
    :return:
    """

    # Compute whether prediction is correct (i.e. has maximum probability mass)
    correct = get_n_correct(output, targets)

    # Compute entropy on the output layer
    entropy = compute_entropy_output(output)

    # Compute entropy on the output layer, just among the items with higher probability (top10%)
    tenpc = round(output.size()[1]*0.1)
    entropytop10 = compute_entropy_over_topk(output, tenpc)

    # Compute surprisal of the target word
    surprisal = compute_surprisal_output(output, targets)

    # More lenient accuracy: see if target is in top10%
    word_in_top10 = target_in_topk(output, targets, tenpc)

    # Gather all the information
    metrics_dict = {"actual_word": sentence[1:], \
                 "correct": correct, \
                 "previous_word": sentence[:-1], \
                 "predicted_word": torch.argmax(output.cpu(), axis=1), \
                 "entropy": entropy, \
                 "entropy_top10": entropytop10, \
                 "surprisal": surprisal, \
                 "target_in_top10": word_in_top10
                 }

    return metrics_dict
