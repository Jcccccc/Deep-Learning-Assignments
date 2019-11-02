# CSCI-566 Assignment 2

## The objectives of this assignment
* Implement the forward and backward passes as well as the neural network training procedure for Recurrent Neural Networks (RNNs).
* Learn the basic concepts of language modeling and how to apply RNNs.
* Implement popular a generative model, Generative Adversarial Networks (GANs) using TensorFlow.

## Work on the assignment
Please first clone or download as .zip file of this repository.

Working on the assignment in a virtual environment is highly encouraged.
In this assignment, please use Python `3.5` (or `3.6`).
You will need to make sure that your virtualenv setup is of the correct version of python.

Please see below for executing a virtual environment.
```shell
cd CSCI566-Assignment2
pip3 install virtualenv # If you didn't install it
virtualenv -p $(which python3) ./venv_cs566_hw2
source ./venv_cs566_hw2/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# install tensorflow (cpu version, recommended)
pip3 install tensorflow=='1.14.0'

# install tensorflow (gpu version)
# run this command only if your device supports gpu running
pip3 install tensorflow-gpu=='1.14.0'

# Work on the assignment

# Deactivate the virtual environment when you are done
deactivate
```

## Work with IPython Notebook
To start working on the assignment, simply run the following command to start an ipython kernel.
```shell
# add your virtual environment to jupyter notebook
python -m ipykernel install --user --name=venv_cs566_hw2

# port is only needed if you want to work on more than one notebooks
jupyter notebook --port=<your_port>

```
and then work on each problem with their corresponding `.ipynb` notebooks.
Check the python environment you are using on the top right corner.
If the name of environment doesn't match, change it to your virtual environment in "Kernel>Change kernel".

## Problems
In each of the notebook file, we indicate `TODO` or `Your Code` for you to fill in with your implementation.
Majority of implementations will also be required under `lib` with specified tags.

### Problem 1: RNNs for Language Modeling (60 points)
The IPython Notebook `Problem_1.ipynb` will walk you through implementing a recurrent neural network (RNN) from scratch.

### Problem 2: Generative Adversarial Networks  (40 points)
The IPython Notebook `Problem_2.ipynb` will help you through implementing a generative adversarial network (GAN).

## PLEASE DO NOT CLEAR THE OUTPUT OF EACH CELL IN THE .ipynb FILES
Your outputs on the `.ipynb` files will be graded. We will not rerun the code. If the outputs are missing, that will be considered as if it is not attempted.

## How to submit

Run the following command to zip all the necessary files for submitting your assignment.

```shell
sh collectSubmission.sh
```

This will create a file named `assignment2.zip`, please rename it with your usc student id (eg. 4916525888.zip), and submit this file through the [Google form](https://forms.gle/2bdjgDwXBsnR2ap38).
Do NOT create your own .zip file, you might accidentally include non-necessary
materials for grading. We will deduct points if you don't follow the above
submission guideline.

## Questions?
If you have any question or find a bug in this assignment (or even any suggestions), we are
more than welcome to assist. Please take a look at the FAQ section below before posting a question.

Again, NO INDIVIDUAL EMAILS WILL BE RESPONDED.

PLEASE USE **PIAZZA** TO POST QUESTIONS (under folder assignment2).


## FAQ

- **Can I reuse the virtualenv from Assignment 1?**\
You can reuse the vistual environment but maybe you need to install some missing packages using `pip3 install -r requirements.txt`. \
Maybe simpler is to create a new virtualenv, we give instructions above.

- **My RNN in Problem 1 is better than the LSTM?** \
Try experimenting with the number of training epochs (LSTM may train slower in the beginning) and the training prediction horizon (benefits of LSTMs get more apparent on longer prediction problems). \
Even then, it is possible that the training dataset is too simple to show large benefits of LSTMs.

- **When parsing the text data file for Problem 1 I get a `charmap codec can't decode` error.**\
This might be a platform dependent issue. In past years adding `encoding="utf8"` to the file `open` command helped in these cases.

- **What is the `meta` variable used for in Problem 1?**\
This variable is used to pass all values to the backward pass that are necessary to compute the gradients. You can use it as a dictionary to pass any desired values over to the `backward` function.

- **I experimented with the hyperparameters and tried many different combinations, which ones should I report?**\
The usual rule of thumb is to report results with the best hyperparameters you found. \
Exception is the prediction horizon parameter `T` in Problem 1, please **do not** report results for `T` smaller than the default value.

- **The function set_seed() produces an error?**\
Make sure your tensorflow version is not below 1.12.0.

- **My reconstruction loss for Problem 2.2 is higher than 32?** \
You should achieve a reconstruction loss lower than 32 finally.
