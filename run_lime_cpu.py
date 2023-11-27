# Generating explanations with LIME
# Jiaxin Li, 2021

import os
import csv
import argparse
import numpy as np
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
import time
import random
from lime import lime_text
from lime.lime_text import LimeTextExplainer


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
parser.add_argument("--kernel_size", type=int, default=random.choice([2]))

# batchSize indicates how many program codes are processed together. Generally higher batch sizes lead to faster training, but take more GPU memory.
parser.add_argument("--batchSize", type=int, default=random.choice([8]))

# learning_rate indicates the speed with which the model's parameters are updated
parser.add_argument("--learning_rate", type=float, default=random.choice([0.0001]))

# dropout_rate indicates how many units of the hidden vector are randomly set to zero on every training pass
parser.add_argument("--dropout_rate", type=float, default=random.choice([0.5]))


# whether or not to use the ReLU activation function in the network
parser.add_argument("--relu", type=int, default=random.choice([1]))

# most_frequent: add this many most frequent words to the vocabulary
parser.add_argument("--most_frequent", type=int, default=random.choice([1000]))

# the size of the explanation, i.e., how many n-grams should be extracted
parser.add_argument("--n_explanations", type=int, default=random.choice([10, 50, 100]))

# the path where the model checkpoint is stored
parser.add_argument("--model_path", type=str)

# where the source code is located
parser.add_argument("--input_code", type=str, default="example/raw_code.cpp")


args = parser.parse_args()

# Create a unique integer for this model run. This will be used for storing the accuracies.
id_for_this_run = random.randint(100000,1000000)

####################################################################################################
####################################################################################################


# This is the vocabulary of frequent words. For every word, it will store how often it appeared in the data.
tokenVocabulary = collections.defaultdict(int)





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



# This is for shuffling the dataset.
# Setting a fixed seed is useful for making results comparable across different runs (so the model will be run on the same program codes, in the same order)
def shuffled(x, seed=random.randint(0,100)):
    random.Random(seed).shuffle(x)
    return x


def prepareInputNumeric(inputs):
    # Turn program codes into lists of integers, by encoding every character/word into an integer
    # inputs: a list of program codes (each one is a string)

    # This is what should be returned: a list of lists of integers
    inputs_format = []
     #loop through all program codes
    for j in range(len(inputs)):
        inp = inputs[j]
        processed = []
        while len(inp) > 0:
            whitespaces = 0
            while len(inp) > 0 and inp[0] == " ":
                inp = inp[1:]
                whitespaces+=1
            if whitespaces > 0:
                processed.append(f"WHITE{whitespaces}")
            if " " in inp:
                q = inp.index(" ")
            else:
                q = len(inp)
            nextWord = inp[:q]
            if nextWord in stoi:
                processed.append(nextWord)
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

def extractListOfAllNGrams(inputs):
    numeric = prepareInputNumeric([inputs])[0]
    ngrams = ["_".join([str(w) for w  in (numeric[i:i+args.kernel_size])]) for i in range(len(numeric)-args.kernel_size)]
    ngrams = " ".join(list(set(ngrams)))
    return ngrams

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


# This is a version of the runModel version that runs on the transformed version of the codes used for LIME, where each code is represented as a list of N-Grams
def runModel_ngrams(inputs):
    inputs_numeric = []
    for inputs_ in inputs:
       inputs_numeric.append([[int(z) for z in x.split("_")] for x in inputs_.split(" ") if len(x) > 0])
    maxSize = max(len(y) for y in inputs_numeric)
    numeric_all = [x + [[stoi["PAD"] for _ in range(args.kernel_size)] for _ in range(maxSize - len(x))] for x in inputs_numeric]
    author_scores_over_batches = []

    # LIME constructs 5,000 versions (so 'inputs' has length 5,000), which is too much to run through the model in one go.
    # Therefore, go through it in batches of size 24
    for batch in range(0, len(numeric_all), 24):
       # Place the codes into a tensor and put it on the GPU
       numeric = torch.LongTensor(numeric_all[batch:batch+24])

       # Store the sizes of this tensor
       BATCH_SIZE, LENGTH, KERNEL_SIZE = numeric.size()

       # Run this through the embeddings module.
       # Here, I'm using a trick where I represent every ngram as its own code of length equal to the kernel size.
       # This way, I can then obtain the output of the convolution module specifically on this ngram.
       embedded = embeddings(numeric.view(BATCH_SIZE*LENGTH, KERNEL_SIZE))

       # Run the embeddings through the convolutional network
       layer1 = conv(embedded.transpose(1,2))

       # Optionally, apply the ReLU function
       if args.relu > 0:
           layer1 = torch.nn.ReLU()(layer1)
         
       # Now transform the tensor of the vectors back into the previous shape, so that vectors belonging to the same code are together.
       layer1 = layer1.view(BATCH_SIZE, LENGTH, args.hidden_size)

       # Apply max pooling
       aggregated = torch.max(layer1, dim=1)[0]
       # Apply dropout
       aggregated = dropout(aggregated)
       # Run through the output module to get scores for every author
       author_scores = output(aggregated)
       # Store these scores in a list. Call detach().cpu() to move it onto the CPU and make sure Pytorch doesn't waste GPU memory for this.
       author_scores_over_batches.append(author_scores.detach().cpu())
    # For every one of the size-24 batches, I have collected the author scores in author_scores_over_batches.
    # Now, concatenate these together into one big table of author scores
    author_scores = torch.cat(author_scores_over_batches, dim=0)
    return author_scores


# This is the function that LIME uses to call my model for predictions.
# 'inputs' is a list of 5,000 strings, each of which consists of some set of n-grams with whitespaces in between
def prediction_function(inputs):
    print("Running model to get predictions on ", len(inputs), " generated program codes.") # Tell the human that the model is getting predictions
    # Run the model
    author_scores = runModel_ngrams(inputs)
    # So far, the author scores are just numbers, but LIME needs these to be probabilities (a probability that a person wrote a code).
    # Therefore, run Softmax to transform the scores into probabilities
    probs = torch.nn.Softmax(dim=1)(author_scores)
    # Return the probabilities, call detach().cpu().numpy() to prevent Pytorch from wasting memory
    return probs.detach().cpu().numpy()

# This function takes a source code and uses LIME to create explanations
# inputs: a source code (a string)
def runModelInterpret(inputs):
    # Step 1: Initialize LIME
    explainer = LimeTextExplainer(class_names=list(range(1000)))

    # Step 2: Extract all ngrams from the code
    listOfNgramsInCode = extractListOfAllNGrams(inputs)

    # Step 3: Run LIME.
    # Here:
    #  listOfNgramsInCode contains all ngrams in the code
    #  prediction_function is a function defined above, which is called by LIME, and which runs my model on a set of ngrams
    #  num_features describes how many n-grams LIME should identify for its explanation
    exp = explainer.explain_instance(listOfNgramsInCode, prediction_function, num_features=args.n_explanations)
    # Turn the explanations provided by LIME into a Python list
    explanations = (exp.as_list())

    # What would the model predict on the full code?
    predictionFromFull = int(runModel_ngrams([listOfNgramsInCode]).max(dim=1)[1])

    explanationsNegative = [x for x in explanations if x[1] > 0] # The weights returned by LIME are inverted: negative weights correspond to positive, and vice versa
    explanationsPositive = [x for x in explanations if x[1] < 0]
    # Prrint the explanations, ordered in increasing order of weights
    print("Explanations for this author, ordered by their weight:")
    print()
    print("Negative Weights: Evidence AGAINST the author")
    for ngram, imp in sorted(explanationsNegative, key=lambda x:x[1]):
        ngram = [itos[int(q)] for q in ngram.split("_")]
        print(ngram)
    print()
    print("Positive Weights: Evidence FOR the author")
    for ngram, imp in sorted(explanationsPositive, key=lambda x:x[1]):
        ngram = [itos[int(q)] for q in ngram.split("_")]
        print(ngram)
    return None


# this is for monitoring how long creating the explanations takes
overallStartTime = time.time()


print("Reading program code from ", args.input_code)
with open(args.input_code, "r", newline='') as inFile:
    inputCode = inFile.read()

dropout.train(False)

interpretation = runModelInterpret(inputCode)

