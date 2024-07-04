__author__ = 'Raquel G. Alhama'
import numpy as np
npa=np.array

class ConvergenceState:
    def __init__(self):
        self.all_losses = None
        self.output_probabilities = None
        self.target_probabilities = None

    def update(self, n_epoch, all_losses, output_probabilities, target_probabilities, training_accuracy):
        self.all_losses = all_losses
        self.output_probabilities = output_probabilities
        self.target_probabilities = target_probabilities
        self.n_epoch = n_epoch
        self.training_accuracy = training_accuracy

class ConvergenceCriteria:
    def __init__(self, list_of_criteria, andor):
        self.list_of_criteria = list_of_criteria
        self.andor = andor
        self.state=ConvergenceState()

    def update_state(self, epochs, all_losses, output_probabilities, target_probabilities, training_accuracy):
        self.state.update(epochs, all_losses, output_probabilities, target_probabilities, training_accuracy)

    def converged(self):
        if len(self.list_of_criteria) == 1:
            criterion = self.list_of_criteria[0]
            return criterion(self.state)
        elif self.andor == "and":
            for criterion in self.list_of_criteria:
                if not criterion(self.state):
                    return False
            return True
        elif self.andor == "or":
            for criterion in self.list_of_criteria:
                if criterion(self.state):
                   return True
            return False
        else:
            raise Exception("Unknown logical operator to combine multiple criteria: %s?"%self.andor)



class ConvergenceCriterion:

    def __init__(self, criterion_type, options):
        self.criterion_type=criterion_type
        self.__n_epochs_stagnant=0

        if criterion_type == "halfprob":
            self.__hasConverged = self.convergence_criterion_halfprob
            self.tolerance=options["tolerance"]
        if criterion_type == "argmax":
            self.__hasConverged = self.convergence_criterion_argmax
        if criterion_type == "loss":
            self.max_epochs_stagnated = options["max_epochs_stagnated"]
            self.min_change = options["min_change"]
            self.__hasConverged = self.convergence_criterion_loss
        if criterion_type == "max_epochs":
            self.last_epoch = options["last_epoch"]
            self.__hasConverged = self.convergence_criterion_maxepochs
        if criterion_type == "min_accuracy":
            self.training_accuracy = options["min_accuracy"]
            if self.training_accuracy > 1 and self.training_accuracy <=100:
                self.training_accuracy = self.training_accuracy / 100
            self.__hasConverged = self.convergence_criterion_minaccuracy


    def __call__(self, state):
        if self.criterion_type == "halfprob":
            return self.__hasConverged(state.target_probabilities)
        elif self.criterion_type == "argmax":
            return self.__hasConverged(state.output_probabilities)
        elif self.criterion_type == "loss":
            return self.__hasConverged(state.all_losses)
        elif self.criterion_type == "max_epochs":
            return self.__hasConverged(state.n_epoch)
        elif self.criterion_type == "min_accuracy":
            return self.__hasConverged(state.training_accuracy)
        else:
            raise Exception("No function defined for criterion type: ",self.criterion_type)

    def convergence_criterion_halfprob(self, target_probs):
        """
        Returns true when all the output probabilities for all (minus tolerated) targets accumulate at least 0.5 probability.
        :param target_probs: vector with target probabilities.
        :param tolerance: Percentage of target samples allowed to not reach the half probability criterion.
        :return:
        """
        if self.tolerance is not None:
            ntolerated=self.tolerance*len(target_probs)
        nbelow=0
        for p in target_probs.values():
            if p < 0.5:
                if self.tolerance is None:
                    return False
                else:
                    nbelow+=1
                    if nbelow > ntolerated:
                        return False
        return True

    def convergence_criterion_argmax(self, preds, w2i):
        """
        Returns True if the target has the maximum probability (regardless of the amount).
        :param preds: torch output vector
        :param w2i: word to index
        :return:
        """
        for word, pred in preds.items():
            probs=np.exp(npa(pred.data)).reshape((-1))
            if np.argmax(probs) != w2i[word]:
                return False
        return True


    def convergence_criterion_loss(self, all_losses):
        ''' Returns True if the model has stagnated (i.e. the number of epochs where the change is smaller than self.min_change is bigger than self.max_epochs_stagnated)'''
        if len(all_losses) < self.max_epochs_stagnated:
            return False

        prev = np.mean(npa(all_losses[-2]))
        act = np.mean(npa(all_losses[-1]))
        diff_proportion = 1 - act/prev
        if abs(diff_proportion) < self.min_change:
            self.__n_epochs_stagnant +=1
        else:
            self.__n_epochs_stagnant = 0

        return self.__n_epochs_stagnant >= self.max_epochs_stagnated

    def convergence_criterion_maxepochs(self, act_epoch):
        ''' Returns True if we have reached the maximum number of epochs.  '''
        return act_epoch >= self.last_epoch

    def convergence_criterion_minaccuracy(self, training_accuracy):
        mean_acc = training_accuracy["correct"].mean()
        return mean_acc >= self.training_accuracy


    #@deprecated
    @staticmethod
    def convergence_criterion_mean_loss(all_losses, n):
        if len(all_losses) < n:
            return False
        #Check last n epochs
        for i in range(n, 1,-1):
            prev = np.mean(npa(all_losses[-i]))
            act = np.mean(npa(all_losses[-(i-1)]))
            if (1 - act/prev) > 0.05:
                return False
        return True
