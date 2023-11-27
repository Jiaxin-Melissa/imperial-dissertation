# Training a convolutional network for authorship attribution on the AI-SOCO challenge
# Jiaxin Li, 2021
# Note: This file was previously called conv_net_gpu_tune7.py

import collections
import zipfile
import argparse
import random
import torch
import os
import csv
import numpy as np
import collections
import torch
import random


####################################################################################################
####################################################################################################

# Extract the hyperparameters from the command line arguments
parser = argparse.ArgumentParser()

# hidden_size is the size of the vectors created by the convolution
parser.add_argument("--hidden_size", type=int, default=random.choice([700, 900, 1200]))
# The point of the "default=random.choice(...)" part is that, when no argument is provided, the code will randomly choose one of the options.
# This is used in hyperparameter search, where I call the code multiple times to try multiple different choices of these hyperparameters.


# embedding_size is the size of the embeddings assigned to the characters
parser.add_argument("--embedding_size", type=int, default=random.choice([50, 100, 200]))

# kernel_size is the size of the convolutional kernel (how many adjacent characters are merged in the convolution)
parser.add_argument("--kernel_size", type=int, default=random.choice([2, 3, 4, 5, 6, 7, 10]))

# batchSize indicates how many program codes are processed together. Generally higher batch sizes lead to faster training, but take more GPU memory.
parser.add_argument("--batchSize", type=int, default=random.choice([8, 16, 24, 32]))

# learning_rate indicates the speed with which the model's parameters are updated
parser.add_argument("--learning_rate", type=float, default=random.choice([0.00005, 0.0001, 0.0002, 0.001, 0.002]))

# dropout_rate indicates how many units of the hidden vector are randomly set to zero on every training pass
parser.add_argument("--dropout_rate", type=float, default=random.choice([0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5]))

# where the data are
parser.add_argument('--data_dir', default='data_dir')

# whether or not to use the ReLU activation function in the network
parser.add_argument("--relu", type=int, default=random.choice([0, 1]))

# most_frequent: add this many most frequent words to the vocabulary
parser.add_argument("--most_frequent", type=int, default=random.choice([100, 500, 1000, 1000, 1000, 2000, 5000]))

args = parser.parse_args()

# Create a unique integer for this model run. This will be used for storing the accuracies.
id_for_this_run = random.randint(100000,1000000)

####################################################################################################
####################################################################################################


# This is the vocabulary of frequent words. For every word, it will store how often it appeared in the data.
tokenVocabulary = collections.defaultdict(int)



# Source of the load_labels function: random_baseline.py in the AI-SOCO starter code available at https://competitions.codalab.org/competitions/25148.
def load_labels(split):
    with open(os.path.join(args.data_dir, '{}.csv'.format(split)), 'r') as fp:
        reader = csv.reader(fp)
        problems = list(reader)
    problems = problems[1:]
    return problems


# Source of the load_data function: tfidf_knn_baseline.py.sh in the AI-SOCO starter code available at https://competitions.codalab.org/competitions/25148.
def load_data(split, split_csv):
    problems = list()
    counter = 0
    with zipfile.ZipFile(os.path.join(args.data_dir, "ZIPPED_DATA.zip"), "r") as zipped:
      for row in split_csv:
        counter += 1
        if counter % 1000 == 0:
            print("Read", counter, "program codes from the", split, "set.")
        with zipped.open(split +"/" + row[1]) as fp:
            code = fp.read().decode('UTF-8').replace('\n', " \n ").replace('\r\n', " \r\n ")
            problems.append(code)
            for x in code.split(" "): # .
                if len(x) > 1:
                   tokenVocabulary[x]+=1
    return problems


# The following three lines are taken from tfidf_knn_baseline.py.sh in the AI-SOCO starter code available at https://competitions.codalab.org/competitions/25148.
train_csv = load_labels('train')
train_labels = list(zip(*train_csv))[0]
train_labels = list(map(int, train_labels))

# The following three lines are taken from tfidf_knn_baseline.py.sh in the AI-SOCO starter code available at https://competitions.codalab.org/competitions/25148.
dev_csv = load_labels('dev')
dev_labels = list(zip(*dev_csv))[0]
dev_labels = list(map(int, dev_labels))

print("Loading training data")
train_data = load_data('train', train_csv)
print("Loading dev data")
dev_data = load_data('dev', dev_csv)

####################################################################################################
####################################################################################################


# When reading the data (calling load_data), the code collected counts for all words.
# Now, sort the words by their frequency
tokenVocabulary = sorted(list(tokenVocabulary.items()), key=lambda x:x[1], reverse=True)
# and select the most frequent ones
tokenVocabulary_selected = [x[0] for x in tokenVocabulary[:args.most_frequent]]


# itos will be the *list* of all characters and words for which the model will have embeddings.
# It consists of:
# First, special symbols PAD (assigned the number 0), to be added to shorter codes to make sure all codes have the same length
# Second, special symbol OOV (assigned the number 1), to be used in place of unknown characters that didn't appear in the training set
# Third, all characters appearing in the entire training data, sorted
# Fourth, all the special whitespace characters
# Fifth, the selected words
itos = ["PAD", "OOV"] + sorted(list(set(list("".join(train_data)))) + [f"WHITE{i}" for i in range(1,21)]) + tokenVocabulary_selected

# stoi is a *dict* mapping every string to its position in itos
stoi = dict(list(zip(itos, range(len(itos)))))



####################################################################################################
####################################################################################################

# Build the convolutional network. Its components are
# First, the embeddings (one embedding vector for every word or character in itos/stoi)
# Second, the convolution, which transforms embedding vectors of adjacent characters/words into a single vector
# Third, the output, with transforms thye vector for the whole code into scores for all the different possible authors

print("Constructing Embedding Matrix")
embeddings = torch.nn.Embedding(len(stoi), args.embedding_size)
print("Constructing Convolutional Net")
conv = torch.nn.Conv1d(in_channels=args.embedding_size, out_channels=args.hidden_size, kernel_size=args.kernel_size)
print("Constructing Output Classifier")
output = torch.nn.Linear(args.hidden_size, 1000)

# ceLoss is the loss function (given the scores and the correct authors, it computes the loss)
ceLoss = torch.nn.CrossEntropyLoss()

# dropout randomly sets some activations in the network to zero during training to prevent overfitting
dropout = torch.nn.Dropout(p=args.dropout_rate)


# Now, create the optimizer, which will adapt the parameters to decrease the loss function during training
def parameters():
    for x in [embeddings, conv, output]:
        for y in x.parameters():
            yield y

optim = torch.optim.Adam(lr=args.learning_rate, params=parameters())

####################################################################################################
####################################################################################################



# Put together the source  codes and the labels indicating their authors
dev = list(zip(dev_data, dev_labels))
# Sort the codes by length. The purpose of this is to avoid wasting 
#dev = sorted(dev, key=lambda x:len(x[0]))

data = list(zip(train_data, train_labels))
#data = sorted(data, key=lambda x:len(x[0]))

# This helper function is used to shuffle the training set
def shuffled(x):
    random.shuffle(x)
    return x


def prepareInputNumeric(inputs):
    # Turn program codes into lists of integers, by encoding every character/word into an integer
    # inputs: a list of program codes (each one is a string)

    # This is what should be returned: a list of lists of integers
    inputs_format = []
     #loop through all program codes
    for j in range(len(inputs)):
        inp = inputs[j] # this is the j-th program code
        processed = [] # this is the output: a list of integers
        while len(inp) > 0: # loop as long as the code has not been fully processed
            whitespaces = 0 # count how many whitespaces there are in a row
            while len(inp) > 0 and inp[0] == " ": # remove all whitespaces at the beginning of the code
                inp = inp[1:]
                whitespaces+=1
            if whitespaces > 0: # there were some whitespaces
                processed.append(f"WHITE{whitespaces}") # add a symbol for these whitespaces to the output
            if " " in inp: # look whether there is another whitespace in the remaining program code
                q = inp.index(" ")
            else:
                q = len(inp)
            nextWord = inp[:q] # the next word goes from the start of the program to the next whitespace, or to the end (if there is no whitespace)
            if nextWord in stoi: # the next word is in the vocabulary
                processed.append(nextWord) # put the next word into the output
            else:
                # put each of the next word's characters individually into the output
                for character in nextWord:
                    processed.append(character)
            # remove the first word from the program code
            inp = inp[q+1:]

        # So far, the output is a list of characters or strings
        # Now, transform each of them into an integer using the stoi (string-to-int) dictionary
        # In case a string isn't contained in the vocabulary, instead represent as 1 (which is reserved for out-of-vocabulary strings)
        processed = [stoi.get(q, 1) for q in processed]
        inputs_format.append(processed)
    return inputs_format


def runModel(inputs):
    # First. transform the list of program codes (which are strings) into a list of lists of integers
    numeric = prepareInputNumeric(inputs)
    # Find out the longest length of any of the programs
    maxSize = max(len(y) for y in numeric)
    # Add zeros to shorter programs so that all of the lists have the same length
    numeric = [x + [stoi["PAD"] for _ in range(maxSize - len(x))] for x in numeric]
    # Now turn this into a tensor
    numeric = torch.LongTensor(numeric)
    # Convert the long (integer) tensor into an embedding tensor
    embedded = embeddings(numeric)
    # Run the convolutional network on the embeddings.
    # Here, it is necessary to call 'transpose' because the convnet in Pytorch expects a different order of the dimensions
    layer1 = conv(embedded.transpose(1,2))
    # Optionally, run the ReLU unit over the vector
    # The ReLU unit makes the network nonlinear (it replaces negative values by zero), which could make the network more powerful
    if args.relu > 0:
        layer1 = torch.nn.ReLU()(layer1)
    # Now apply max-pooling using torch.max(...)
    # I'm calling dim=2 because this computes the maximum over all the positions within a program code (i.e., the third dimension in the tensor).
    aggregated = dropout(torch.max(layer1, dim=2)[0])
    # Now, call the "output" module (a torch.Linear module) to transform the vector representation into scores for every possible author
    author_scores = output(aggregated)

    return author_scores

# accuracies stores the dev/validation accuracies achieved after every epoch
accuracies = []

# Loop over the epochs. I set 20 to be the maximum number of epochs.
for epoch in range(20):

    # Let the dropout module know that we're training, so it is activated.
  dropout.train(True)

  # Loop through the training set.
  # I first build indices for every batchSize-th index in the training set (e.g., 0, 8, 16, 24 if batchSize=8). This is done by: list(range(0, len(train_data), args.batchSize))
  # I next shuffle this using the shuffled(...) function defined above. The idea is  that shuffling the training set reduces overfitting.
  for i in shuffled(list(range(0, len(train_data), args.batchSize))):
    # Now, we iterate over i's. i will be some multiple of the batchSize, e.g., 0, 8, 16, ...
    # I take the batch starting at i and ranging up to i+batchSize
    inputs = train_data[i:i+args.batchSize] # the program codes (a list of strings)
    labels = train_labels[i:i+args.batchSize] # the labels (a list of integers indicating the authors)
    # Now run the model on the programs to get scores for every author
    author_scores = runModel(inputs)
    # Put the labels (a list of integers) on the GPU
    labels = torch.LongTensor(labels)
    # Run the author scoes and the labels through the loss function
    loss = ceLoss(author_scores, labels)
    # Run the optimizer to improve the parameters
    optim.zero_grad() # delete data from previous iteration
    loss.backward() # find out how to change each parameter to decrease loss
    optim.step() # change the parameters
    if random.random() < 0.1: # From time to time, print the loss and the accuracies obtained so far
        print("Training epoch:", epoch, ". Cross-Entropy: ", float(loss), ". Dev accuraies so far: ",accuracies)

  # Let the dropout module know that we're testing, so it is de-activated
  dropout.train(False)
  overall = 0 # this counts how many codes the model has been evaluated on
  correct = 0 # this counts how often among these codes the model was right

  # As above for the training set, now loop through the development/validation set.
  for i in list(range(0, len(dev), args.batchSize)):
    inputs = dev_data[i:i+args.batchSize]
    labels = dev_labels[i:i+args.batchSize]
    # Now run the model on the programs to get scores for every author
    author_scores = runModel(inputs)
    # Put the labels (a list of integers) on the GPU
    labels = torch.LongTensor(labels)
    # Obtain model predictions: For every one of the source codes in the batch, find out which author receives the highest score.
    # I use dim=1 because this looks at the maximum over all authors, for every one of the codes.
    predictions = torch.argmax(author_scores, dim=1)
    # Now count for how many of the programs in the batch the prediction was equal to the label
    correct += float((predictions == labels).sum())
    # And count how many programs there were
    overall += len(inputs)
    if random.random() < 0.1 == 0: # From time to time, print the accuracy seen so fat
        print("Running on test set. Progress: ", i/len(dev), ". Accuracy: ", correct/overall)
  # Now store the accuracy
  accuracies.append(float(correct/overall))

  # Save the results to a file
  with open(f"output/{__file__}_eval_{id_for_this_run}.txt", "w") as outFile: # e.g., the filename might be output/trainConvolutionalModel.py_eval_87787.txt
    print(args, file=outFile) # store the hyperparameters
    print(" ".join([str(q) for q in accuracies]), file=outFile) # store the accuracies

  # If the model has run through dev/valid at least twice, and the last accuracy was not better than the second-to-last one (i.e., the model hasn't improved), stop training.
  if len(accuracies) > 1 and accuracies[-1] <= accuracies[-2]:
      break
  torch.save({"args" : args, "vocab" : itos, "model" : [x.state_dict() for x in [embeddings, conv, output]]}, "trained_model_"+str(id_for_this_run)+".model")
