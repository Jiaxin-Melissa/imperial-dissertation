Authorship attribution for C++ code on the data of the AI-SOCO 2020 competition.
Jiaxin Li, 2021

Prerequisites:

The code was developed using
* Python 3.8.10
* torch 1.9.0+cu111
* lime 0.2.0.1

Pretrained models are available on OneDrive (can be accessed with the link inside Imperial) :

`(kernel size=1) model_pretrained_1grams.model https://imperiallondon-my.sharepoint.com/:u:/g/personal/jl220_ic_ac_uk/EdfR85njMrlBh0rMOlH0hgkBlIg0Es1uW-f9RjWgO43xlg?e=QuZ45w`

`(kernel size=2) model_pretrained_2grams.model https://imperiallondon-my.sharepoint.com/:u:/g/personal/jl220_ic_ac_uk/Ea3-vyfkk6VGgeKVpXUT-DAB9o6SO1-vWTEx4y88dkGu1w?e=0EqrYc`

`(kernel size=5) model_pretrained_5grams.model https://imperiallondon-my.sharepoint.com/:u:/g/personal/jl220_ic_ac_uk/EZQoCVtPVOpOt5Gurtz_jfMBwX00HI2f-AvJouE8S_y-Tg?e=lm24lQ`

Using the scripts:

1. Training Model

`python model_train.py --kernel_size=5`

2. Creating test predictions

`python model_getTestPredictions.py --model_path=model_pretrained_5grams.model`

3. Evaluating LIME

`python lime_evaluate.py --model_path=model_pretrained_5grams.model`

For the random baseline, use:

`python lime_evaluate_baseline.py --model_path=model_pretrained_5grams.model`

4. Running model to classify a program code

`python model_Classify.py --input_code=example/raw_code.cpp --model_path=model_pretrained_5grams.model`

5. Running LIME to explain a prediction

`python3 run_lime.py  --input_code=example/raw_code.cpp --model_path=model_pretrained_1grams.model --n_explanations=10`


