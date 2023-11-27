# Getting predictions on the test set from the convolutional network
# Jiaxin Li, 2021

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
parser.add_argument("--hidden_size", type=int, default=random.choice([900]))
# The point of the "default=random.choice(...)" part is that, when no argument is provided, the code will randomly choose one of the options.
# This is used in hyperparameter search, where I call the code multiple times to try multiple different choices of these hyperparameters.


# embedding_size is the size of the embeddings assigned to the characters
parser.add_argument("--embedding_size", type=int, default=random.choice([200]))

# kernel_size is the size of the convolutional kernel (how many adjacent characters are merged in the convolution)
parser.add_argument("--kernel_size", type=int, default=random.choice([5]))

# batchSize indicates how many program codes are processed together. Generally higher batch sizes lead to faster training, but take more GPU memory.
parser.add_argument("--batchSize", type=int, default=random.choice([16]))

# learning_rate indicates the speed with which the model's parameters are updated
parser.add_argument("--learning_rate", type=float, default=random.choice([0.0001]))

# dropout_rate indicates how many units of the hidden vector are randomly set to zero on every training pass
parser.add_argument("--dropout_rate", type=float, default=random.choice([0.5]))

# where the data are
parser.add_argument('--data_dir', default='data_dir/TEST')

# whether or not to use the ReLU activation function in the network
parser.add_argument("--relu", type=int, default=random.choice([0, 1]))

# most_frequent: add this many most frequent words to the vocabulary
parser.add_argument("--most_frequent", type=int, default=random.choice([500]))

parser.add_argument("--model_path", type=str, default="model_pretrained_5grams.model")

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
    with zipfile.ZipFile(os.path.join(args.data_dir, "ai-soco.zip"), "r") as zipped:
      for row in split_csv:
        counter += 1
        if counter % 1000 == 0:
            print("Read", counter, "program codes from the", split, "set.")
        with zipped.open(split +"/" + row[1]) as fp:
            code = fp.read().decode('UTF-8').replace('\n', " \n ").replace('\r\n', " \r\n ")
            problems.append((code, row[1]))
            for x in code.split(" "): # .
                if len(x) > 1:
                   tokenVocabulary[x]+=1
    return problems



print("Loading test data")
test_csv = load_labels('test')
test_labels = list(zip(*test_csv))[0]
test_labels = list(map(int, test_labels))
test_data = load_data('test', test_csv)


####################################################################################################
####################################################################################################






# Load the previously trained model
# Note that, in order to run on the CPU, I pass the map_location information according to https://pytorch.org/tutorials/beginner/saving_loading_models.html
checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))


# Load all the configuration arguments concerning the hyperparameters of the model's size.
# No need to load other parameters such as learning rate as these are not relevant to making predictions after training/
args.hidden_size = checkpoint["args"].hidden_size
args.embedding_size = checkpoint["args"].embedding_size
args.kernel_size = checkpoint["args"].kernel_size
args.relu = checkpoint["args"].relu
args.most_frequent = checkpoint["args"].most_frequent


# Initialize the vocabulary from the trained model, to ensure the mapping from characters/words to embeddings is correct
itos = checkpoint["vocab"]
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

# Load the parameters of the model from the loaded model
for x, y in zip([embeddings, conv, output], checkpoint["model"]):
       x.load_state_dict(y)

# Unlike the training version, no need to create an optimizer here as this file here is only about testing, not training.

####################################################################################################
####################################################################################################

test = list(zip(test_data, test_labels))
test = sorted(test, key=lambda x:len(x[0]))



# This is for shuffling the dataset.
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

print("Hyperparameter configuration: ", args)

# Open a file to write the test set predictions to
with open("predictions.csv", "w") as outFile:
  # The first line of the predictions file has this form
  print("uid,pid", file=outFile)

  # Let the dropout module know that we're testing, so it is de-activated
  dropout.train(False)
  # As before for the training set, loop through the test set
  for j, i in enumerate(list(range(0, len(test_data), args.batchSize))):
    batch = test[i:i+args.batchSize]
    inputs, labels = zip(*batch)
    # Different from the train/dev sets, every test program comes with an ID that uniquely specifies it. Extract these IDs into a list of IDs, to put these into the predictions file.
    inputs, input_ids = zip(*inputs)
    # Now run the model on the programs to get scores for every author
    author_scores = runModel(inputs)
    # Obtain model predictions: For every one of the source codes in the batch, find out which author receives the highest score.
    # I use dim=1 because this looks at the maximum over all authors, for every one of the codes.
    predictions = torch.argmax(author_scores, dim=1)

    # Now turn predictions from a tensor on the GPU into a simple list of integers
    predictions_ = predictions.detach().cpu().numpy().tolist()

    # Iterate over the programs in the batch, and write the predictions (which author) together with the IDs (which program) to the output file.
    for uid, pid in zip(predictions, input_ids):
       print(f"{int(uid)},{int(pid)}", file=outFile)

    if random.random() < 0.1: # From time to time, print how far the model has gitten
       print("Progress: ", i/len(test_data))

