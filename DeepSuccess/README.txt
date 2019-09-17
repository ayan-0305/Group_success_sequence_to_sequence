# Guidelines to run the code for the DeepSuccess model

################################################################

**Installation requirements**

1. Install the ""torch"" package for pyTorch. Follow the guidelines from: "https://pytorch.org"
2. Install the "numpy" package as well as other python packages imported in the file "DeepSuccess.py"
3. Use Python version 2.7 to run the codes

##############################################################


**Instructions to run the code**

1. Activate the environment by running the command: `source pyTorch/env_py/bin/activate`

2. Run `python DeepSuccess.py`. 

a) The file "DeepSuccess.py" is the main file containing the codes for data extraction, initializing the model parameters, model initialization, training and testing the model as well as print the results for the DeepSuccess model.

b) The file "data.py" contains the code for extracting the data relevant to success label, success metrics, feature vectors and category information of Meetup groups.

c) The file "model.py" contains the code for building the Encoder, Decoder and attention mechanism for the sequence-to-sequence model.

d) The file "utils.py" contains the code for defining the loss functions as well as training and testing of the model. It also defines functions for computing the Accuracy for classification as well as the Binning Accuracy for estimating success metrics. 

**Output**

The code will output the Accuracy, Precision and Recall values for success prediction and the Binning accuracy for estimating group size and event attendance.

