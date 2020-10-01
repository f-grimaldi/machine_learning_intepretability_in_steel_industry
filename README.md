# Interpretability Methods for Machine Learning algorithms trained on Steel Industry data

In this repo it is present the work done in order to create an interpretable and explainable model which classify if an image of has a defect or not (binary model) or also the type of defect in the image (if any) (multi class model). For each model evaluation, inspection and interpretation techniques have been used. <br>
The structure of the repo is the following: <br>
- *data*: folder where the images for training are present. there are also present two folder called *binaryData* and *multiData* which contain the metadata (.csv) used to get the images for training/validation/test
- *img*: folder where useful images are saved
- *model*: here all (cuda) parameters of the the binary and multi models are saved. In order to retrieve the model you need to initialize the same (pytorch) architecture, move in a cuda device and load the state dict.
- *notebooks_eda*: folder where some Explorative Data Analysis notebooks are. It is possible to visualize some examples and the distribution of our data.
- *notebooks_interpretation*: main focus of this work. Here are present some notebooks which try to fulfill this goal.
- *notebooks_models*: folder where it is possible to evaluate the model, either singularly or by comparison.
- *results*: .txt files of semantic segmentation evaluation metrics
- *scripts*: python scripts used to train models
- *src*: where python functions or classes are defined

### Major requirements

- Python (3.6) <br>
- torch <br>
- torchvision <br>
- numpy <br>
- matplotlib <br>
- sklearn <br>
- scipy <br>
- pandas <br>
- argparse <br>
- captum <br>
- shap <br>
- cv2 <br>
- PIL <br> 
- json <br>

### 1. Data overview

- **Data Source**: Kaggle (https://www.kaggle.com/c/severstal-steel-defect-detection/overview) <br>
- **Company**: Severstal (https://en.wikipedia.org/wiki/Severstal; https://www.severstal.com/eng/about/) <br>
- **Data Description**: Images of dimension 3 x 1600 x 256, with informations about the presence of the defect, its type (multiple chips on the surface/Single vertical cracks/Multiple vertical cracks/Multiple large surface patches) and the location of it.

### 2. Binary Model
#### 2.a) Model
The final model is a reduced version of the pre-trained Squeeze Net v1.1, with the last three FIRE modules deleted, fine tuned on our dataset and augmented using gaussian noise. It reaches 0.941 accuracy, 0.944 f1-score and an AUC of 0.983 on the test set. This metrics where around 0.18 points greater than a baseline model (Logistic Regression).

![Architecture of the final model](/imgs/ReducedSequeezeNetArch.png)
*FIG. Architecture of SqueezeNet adapted to out problem: the last three fire modules have been deleted*

#### 2.b) Inspection and interpretability techniques
Different inspection and interpretability techniques have been performed in order to explore the model decision policies. Starting from neuron feature visualization and local interpretability methods comparison and ending with local explanations of the test images.
![Example of GradCAM local explanantions](/imgs/HeatMap_Binary/binary_gradCAM.png)
*FIG. Different example of GradCAM application w.r.t the two classes. Red area excites the target neuron while blue ones inhibt it.*

#### 2.c) White-Stain Effect
A potentially spurious correlation has been found in the data. In fact, images with no defects had white stains in the surface and the model learned to distinguish these images by primarily looking for the presence of this stains. <br>
The effect was so strong that attaching a white stains in an image with defects would make the model get confused and misclassify the example as with no defetcs
![White-Stain Effect](/imgs/HeatMap_Binary/white_stain_effect1.png)
*FIG. Example of White-Stain Effect: in the left the original image is classified as with defects (red areas), while, when we attach a white stain over it (right), the model change its prediction and classify the image as with no defects. GradCAM enable us to check that the model is now looking more at the white-stains.*

### 3. Multi Class Model
#### 3.a) Model
For this problem, the same architecture of the binary model has been used. It reached an accuracy of 0.921, a balance accuracy of 0.904 a (weighted) F1-score of 0.924 and an AUC of 0.987.

#### 3.b) Inspection and interpretability techniques
The same interpretability techniques of the binary case have been used in order to explore the model policies. Similar results andd conclusions have been reached.
![Post-processed output of GuidedGradCAM used as a semantic segmentation model](/imgs/HeatMap_Multi/multi_dice.png)
*FIG. Example of post-processed output of GuidedGradCAM used as a semantic segmentation model: we can see that interpretability methods can roughly identify the areas of defects.*

### 4. Conclusion
The present work has showed how detect and classify defects in steel sheets is possible with discrete results and with efficient models, with few parameters. But, above all, we showed how explaining the model process and policies is not only possible but also important, since it enabled us to discovery unwanted and dangerous situations in the data (spurious correlation) and to provide a tool for human-machine interaction and communication, making the model an explainable and human-friendly recommendation system.
