# CSCI-566 Assignment 1

## The objectives of this assignment
* Implement the forward and backward passes as well as the neural network training procedure
* Implement the widely-used optimizers and training tricks including dropout
* Get familiar with TensorFlow by training and designing a network on your own
* Visualize the learned weights and activation maps of a ConvNet
* Use Grad-CAM to visualize and reason why ConvNet makes certain predictions

## Work on the assignment
Working on the assignment in a virtual environment is highly encouraged.
In this assignment, please use Python `3.6`.
You will need to make sure that your virtualenv setup is of the correct version of python.

Please see below for executing a virtual environment.
```shell
cd csci566-assignment1
pip3 install virtualenv # If you didn't install it

# replace your_virtual_env with the virtual env name you want
virtualenv -p $(which python3) your_virtual_env
source your_virtual_env/bin/activate

# install dependencies other than tensorflow
pip3 install -r requirements.txt
# or
pip3 install numpy jupyter ipykernel opencv-python matplotlib

# install tensorflow (cpu version, recommended)
pip3 install tensorflow

# install tensorflow (gpu version)
# run this command only if your device supports gpu running
pip3 install tensorflow-gpu

# work on the assignment
deactivate # Exit the virtual environment
```

## Work with IPython Notebook
To start working on the assignment, simply run the following command to start an ipython kernel.
```shell
# add your virtual environment to jupyter notebook
source your_virtual_env/bin/activate
python -m ipykernel install --user --name=your_virtual_env

# port is only needed if you want to work on more than one notebooks
jupyter notebook --port=your_port_number

```
and then work on each problem with their corresponding `.ipynb` notebooks.
Check the python environment you are using on the top right corner.
If the name of environment doesn't match, change it to your virtual environment in "Kernel>Change kernel".

## Problems
In each of the notebook file, we indicate `TODO` or `Your Code` for you to fill in with your implementation.
Majority of implementations will also be required under `lib` with specified tags.

### Problem 1: Basics of Neural Networks (40 points)
The IPython Notebook `Problem_1.ipynb` will walk you through implementing the basics of neural networks.

### Problem 2: Getting familiar with TensorFlow (30 points)
The IPython Notebook `Problem_2.ipynb` will help you with a better understanding of implementing a simple ConvNet in Tensorflow.

### Problem 3: Visualizations and CAM (30 points)
The IPython Notebook `Problem_3.ipynb` will gain you insights with what neural networks learn with the skills of visualizing them.

## How to submit

Run the following command to zip all the necessary files for submitting your assignment.

```shell
sh collectSubmission.sh
```

This will create a file named `assignment1.zip`, **please rename it with your usc student id (eg. 4916525888.zip)**, and submit this file through the [Google form](https://forms.gle/EZE5KVJ6PNt6TddZ7).
Do NOT create your own .zip file, you might accidentally include non-necessary materials for grading.
We will deduct points if you don't follow the above submission guideline.

## Questions?
If you have any question or find a bug in this assignment (or even any suggestions), we are more than welcome to assist.

Again, NO INDIVIDUAL EMAILS WILL BE RESPONDED.

PLEASE USE **PIAZZA** TO POST QUESTIONS (under folder assignment1).

## FAQ

- Install `opencv-python` using conda

`requirements.txt` specifies `opencv-python==4.0.0.21` as the recommended version, but it is not available on conda.
If you are using conda, you can install the latest stable opencv version on conda: https://anaconda.org/tstenner/opencv

- Cannot get 50% accuracy for TinyNet in Problem 1

You can try to vary the batch size, epochs, learning rate, and parameters of fc layers.

- What is a good starting learning rate?

There is a good article: https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2

- Keep getting a constant loss for Problem 2-1

If you are using `slim`, you should be aware of a ReLU layer following the fully connected layer.
To prevent this happening, you have to set the default activation function of slim's fully connected layer to `None`.

- The zip file to submit is too large

Make sure you do not include your virtual environment, checkpoints, or datasets.

- General debugging tip
1. Make sure your implementations matches the specified model layers perfectly.
2. Make sure the output of one layer (say `self.conv1`) is input to the next layer (say `self.relu1`). Often it's possible we mistype a layer's name.
3. Check the implementation of your optimizer and train_op. Since there is no learning, this is a likely source of error.
4. Print what you are passing to the feed dict to make sure it makes sense.
5. Put print statements at various places inside your implementation code to make sure every module is working as it should.

