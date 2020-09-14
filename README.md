# Interpretability Methods for Machine Learning algorithms trained on Steel Industry data 

### IN PROGRESS

In this repo it is present the work done in order to create an interpretable model which classify if an image of a steel sheet has a defect or not. Starting from a binary classifier and endinf with a multi class lassifier (no defects and four type of defect). For each model evaluation, inspection and intepretation techniques have been used. <br>
The structure of the repo is the following: <br>
- *data*: folder where the images for training are present. there are also present two folder called *binaryData* and *multiData* which contain the metadata (.csv) used to get the images for training/validation/test
- *img*: folder where useful images are saved
- *model*: here all (cuda) parameters of the the binary and multi models are saved. In order to retrieve the model you need to initailize the same (pytorch) architecture, move in a cuda device and load the state dict.
- *notebooks_eda*: folder where some Explorative Data Analysis notebooks are. It is possibile to visualize some examples and the distribution of our data.
- *notebooks_interpretation*: main focus of this work. Here are present some notebooks which try to fulfil this goal.
- *notebooks_models*: folder where it is possible to evaluate the model, either singularly or by comaprison.
- *results*: where some .csv or .txt files are saved
- *scripts*: python scripts used to train models
- *src*: where python functions or classes are defined 



### 1. Data overview
    
- **Data Source**: Kaggle (https://www.kaggle.com/c/severstal-steel-defect-detection/overview) <br>
- **Company**: Severstal (https://en.wikipedia.org/wiki/Severstal; https://www.severstal.com/eng/about/) <br>
- **Data Description**: Images of dimension 3 x 1600 x 256, with informations about the presence of the defect, its type (multiple chips on the surface/Single vertical cracks/Multiple vertical cracks/Multiple large surface patches) and the location of it.

### 2. Binary Model
#### 2.a) Model used
Three models have been used to solve this task: VGG16_bn, Squeeze Net 1.1 and an ad-hoc CNN. 
#### 2.b) Model evaluation
The final model is a reduced version of the pre-trained Squeeze Net 1.1, fine tuned on our dataset and uagmented using gaussian noise. It reaches 0.941 acccuracy, 0.944 f1-score and an AUC of 0.983 on the test set. This metrics where around 0.18 points greater than a baseline model (Logistic Regression).
#### 2.c) Inspection and interpetability techniques
To inspect the behavior of the model a t-sne (for qualitative results) and a logistic regression (for quantitative results) has been used using the output of every layer of the model. It has been seen that as the signal passes the network it becomes more and more linearly separable. Moreover neuron inspection has been done to check if the model has learned interesting features. This analysis revealed that the model learned simple features (horizontal/vertical strips and colors) in the first layers and more complicated ones  (non linear geometry figures) in the last layers. 

In the intepretation part, specific techniques have been used to try to understand the decision policies of the models. Then, these techniques were compared by trying to check how well they could provide a mask of the defects whenever it was present, so that they could be used as a very raw segmentation model.

#### 2.d) Results
IN PROGRESS
