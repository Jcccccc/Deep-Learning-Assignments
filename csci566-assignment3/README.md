# CSCI-566 Assignment 3

## The objectives of this assignment
* Getting familiar with reinforcement learning and policy gradient methods!
* Setting up and interacting with environments, specifically [OpenAI Gym environments](https://github.com/openai/gym).
* Implementing REINFORCE: rollout storage, policy network and training loop.
* Testing RL on various environments and engineering reward function
* Implementing Actor Critic Architecture


## Work on the assignment
Please first clone or download as .zip file of this repository.

Working on the assignment in a virtual environment is highly encouraged.

In this assignment, we recommend you use Python `3.7` (we briefly tested `3.6.8` and it seemed to work too).
You will need to make sure that your virtualenv setup is of the correct version of python.
We will be using *PyTorch* in this assignment.

Please see below for executing a virtual environment.
```shell
cd CSCI566-Assignment3
pip3 install virtualenv # If you didn't install it
virtualenv -p $(which python3) ./venv_cs566_hw3
source ./venv_cs566_hw3/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Work on the assignment

# Deactivate the virtual environment when you are done
deactivate
```

## Work with IPython Notebook
To start working on the assignment, simply run the following command to start an ipython kernel.
```shell
# add your virtual environment to jupyter notebook
python -m ipykernel install --user --name=venv_cs566_hw3

# port is only needed if you want to work on more than one notebooks
jupyter notebook --port=<your_port>

```
and then work on the problem in `Policy_Gradients.ipynb` notebooks.
Check the python environment you are using on the top right corner.
If the name of environment doesn't match, change it to your virtual environment in "Kernel>Change kernel".

## Working on the Problem
In the notebook file `Policy_Gradients.ipynb`, we indicate `TODO` or `Your Code` for you to fill in with your implementation.
You only need to edit this notebook, and only inside the specified TODO blocks.

## PLEASE DO NOT CLEAR THE OUTPUT OF THE CELLS IN THE .ipynb FILES
Your outputs on the `.ipynb` files will be graded. We will not rerun the code. If the outputs are missing, that will will be considered as if it is not attempted, and such cases will be penalized.

## How to submit

Run the following command to zip all the necessary files for submitting your assignment.

```shell
sh collectSubmission.sh <USC_ID>
```

This will create a file named `<USC_ID>.zip` (eg. 4916525888.zip). Please submit this file through the [Google form](https://forms.gle/Q5AmgG1iXWqD1oQQ8).
If you have to create own .zip file, make sure to ONLY include the `Policy_Gradients.ipynb` file, and name your file as `<USC_ID>.zip`.

We will deduct points if you don't follow the above submission guideline.

## Questions?
If you have any question or find a bug in this assignment (or even any suggestions), we are
more than welcome to assist. Please take a look at the FAQ section below before posting a question.

Again, NO INDIVIDUAL EMAILS WILL BE RESPONDED.

PLEASE USE **PIAZZA** TO POST QUESTIONS (under folder hw3).


## FAQ

- **Can I reuse the virtualenv from previous assignments?**  
You can reuse the virtual environment but you should install the missing packages using `pip3 install -r requirements.txt`.  
Usually it is simpler to create a new virtualenv, as given in the instructions above.

- **Do I need to retain training outputs like videos in the ipython notebook?**  
**Yes!** Please do not manually clear any outputs from your notebooks (Note that sometimes the given code will clear the output for you, but you do not need to worry about it).
