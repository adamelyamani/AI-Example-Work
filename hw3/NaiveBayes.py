from collections import defaultdict
import numpy as np


def file_reader(file_path, label):
    list_of_lines = []
    list_of_labels = []

    for line in open(file_path):
        line = line.strip()
        if line=="":
            continue
        list_of_lines.append(line)
        list_of_labels.append(label)

    return (list_of_lines, list_of_labels)


def data_reader(source_directory):
    positive_file = source_directory+"Positive.txt"
    (positive_list_of_lines, positive_list_of_labels)=file_reader(file_path=positive_file, label=1)

    negative_file = source_directory+"Negative.txt"
    (negative_list_of_lines, negative_list_of_labels)=file_reader(file_path=negative_file, label=-1)

    neutral_file = source_directory+"Neutral.txt"
    (neutral_list_of_lines, neutral_list_of_labels)=file_reader(file_path=neutral_file, label=0)

    list_of_all_lines = positive_list_of_lines + negative_list_of_lines + neutral_list_of_lines
    list_of_all_labels = np.array(positive_list_of_labels + negative_list_of_labels + neutral_list_of_labels)

    return list_of_all_lines, list_of_all_labels


def evaluate_predictions(test_set,test_labels,trained_classifier):
    correct_predictions = 0
    predictions_list = []
    prediction = -1
    for dataset,label in zip(test_set, test_labels):
        probabilities = trained_classifier.predict(dataset)
        if probabilities[0] >= probabilities[1] and probabilities[0] >= probabilities[-1]:
            prediction = 0
        elif  probabilities[1] >= probabilities[0] and probabilities[1] >= probabilities[-1]:
            prediction = 1
        else:
            prediction=-1
        if prediction == label:
            correct_predictions += 1
            predictions_list.append("+")
        else:
            predictions_list.append("-")

    print("Total Sentences correctly: ", len(test_labels))
    print("Predicted correctly: ", correct_predictions)
    print("Accuracy: {}%".format(round(correct_predictions/len(test_labels)*100,5)))

    return predictions_list, round(correct_predictions/len(test_labels)*100)


class NaiveBayesClassifier(object):
    def __init__(self, n_gram=1, printing=False):
        self.prior = []
        self.conditional = []
        self.V = []
        self.n = n_gram

    def word_tokenization_dataset(self, training_sentences):
        training_set = list()
        for sentence in training_sentences:
            cur_sentence = list()
            for word in sentence.split(" "):
                cur_sentence.append(word.lower())
            training_set.append(cur_sentence)
        return training_set

    def word_tokenization_sentence(self, test_sentence):
        cur_sentence = list()
        for word in test_sentence.split(" "):
            cur_sentence.append(word.lower())
        return cur_sentence

    def compute_vocabulary(self, training_set):
        vocabulary = set()
        for sentence in training_set:
            for word in sentence:
                vocabulary.add(word)
        V_dictionary = dict()
        dict_count = 0
        for word in vocabulary:
            V_dictionary[word] = int(dict_count)
            dict_count += 1
        return V_dictionary

    def train(self, training_sentences, training_labels):

        # See the HW_3_How_To.pptx for details

        # Get number of sentences in the training set
        N_sentences = len(training_sentences)

        # This will turn the training_sentences into the format described in the HW_3_How_To.pptx
        training_set = self.word_tokenization_dataset(training_sentences)

        # Get vocabulary (dictionary) used in training set
        self.V = self.compute_vocabulary(training_set)

        # Get set of all classes
        all_classes = set(training_labels)

        #-----------------------#
        #-------- TO DO (begin) --------#
        # Note that, you have to further change each sentence in training_set into a binary BOW representation, given self.V

        D = len(self.V)
        BOW = np.zeros((N_sentences, D))
        # Represent each sentence as a BOW and store in a NUMPY matrix
        for x in range(N_sentences):
            sentence = training_set[x]
            for y in sentence:
                if y in self.V:
                    index = self.V[y]
                    BOW[x][index] = 1

        # Compute the conditional probabilities and priors from training data, and save them in:
        # self.prior
        # self.conditional
        # You can use any data structure you want.
        # You don't have to return anything. self.conditional and self.prior will be called in def predict():

        # Create zero NUMPY vectors to initialize self.conditional and self.prior
        self.conditional = np.zeros((len(all_classes), D))
        self.prior = np.zeros(len(all_classes))
        # For each class calculate self.prior and self.conditional
        for y in all_classes:

            N_sentences_in_c = 0
            # Adjusts index for when the class is '-1'
            if y == -1:
                index = 2
            else:
                index = y
            # Counts the total number of sentences that share the current class
            for w in range(N_sentences):
                if training_labels[w] == y:
                    N_sentences_in_c = N_sentences_in_c + 1
            # Represent self.prior as the ratio of sentences in the current class to total sentences
            self.prior[index] = N_sentences_in_c / N_sentences

            Total_N_in_c = 0
            # Counts the total number of words in the BOW that exist in a sentence of the current class
            for a in range(D):
                for b in range(N_sentences):
                    if training_labels[b] == y:
                        if BOW[b][a] == 1:
                            Total_N_in_c = Total_N_in_c + 1
                            break
            # Represent self.conditional values for each word as the ratio of occurrences in sentences that shares the
            # current label to the total number of words in the BOW that exist in a sentence of the current class
            for x in range(D):
                N_in_c = 0
                for z in range(N_sentences):
                    if training_labels[z] == y:
                        if BOW[z][x] == 1:
                            N_in_c = N_in_c + 1
                self.conditional[index][x] = N_in_c / Total_N_in_c

        # -------- TO DO (end) --------#

    def predict(self, test_sentence):

        # The input is one test sentence. See the HW_3_How_To.pptx for details

        # Your are going to save the log probability for each class of the test sentence. See the HW_3_How_To.pptx for details
        label_probability = {
            0: 0,
            1: 0,
            -1: 0,
        }

        # This will tokenize the test_sentence: test_sentence[n] will be the "n-th" word in a sentence (n starts from 0)
        test_sentence = self.word_tokenization_sentence(test_sentence)

        #-----------------------#
        #-------- TO DO (begin) --------#
        # Based on the test_sentence, please first turn it into the binary BOW representation (given self.V) and compute the log probability
        # Please then use self.prior and self.conditional to calculate the log probability for each class. See the HW_3_How_To.pptx for details

        D = len(self.V)
        BOW = np.zeros(D)
        # Represent the sentence as a BOW
        for y in test_sentence:
            if y in self.V:
                index = self.V[y]
                BOW[index] = 1

        # Initialize the three probabilities in a NUMPY zero vector
        probs = np.zeros(3)

        for c in range(3):
            probs[c] = np.log(self.prior[c])
            # Sum the conditional probabilities for the test sentence for each class
            for x in range(D):
                if BOW[x] == 1:
                    probs[c] = probs[c] + np.log(self.conditional[c][x])
            # Return dictionary values
            if c == 2:
                index = -1
            else:
                index = c
            label_probability[index] = probs[c]

        # Return a dictionary of log probability for each class for a given test sentence:
        # e.g., {0: -39.39854137691295, 1: -41.07638511893377, -1: -42.93948478571315}
        # Please follow the PPT to first perform log (you may use np.log) to each probability term and sum them.

        # -------- TO DO (end) --------#

        return label_probability


if __name__ == '__main__':
    train_folder = "data-sentiment/train/"
    test_folder = "data-sentiment/test/"

    training_sentences, training_labels = data_reader(train_folder)
    test_sentences, test_labels = data_reader(test_folder)

    NBclassifier = NaiveBayesClassifier(n_gram=1)
    NBclassifier.train(training_sentences, training_labels)

    results, acc = evaluate_predictions(test_sentences, test_labels, NBclassifier)
