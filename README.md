# SpaceWarp - Generative Local Edits - Computational Photography
SpaceWarp is a program that given an input image generates an aesthetically pleasing edit of the image as if it was captured in deep space. 


## Structure
At this time the structure of this branch is the following:

- Data modification: It is an helper directory where we put our data and for each training dataset, we renamed, copied and sometimes modified the images such that we can use them in our training_GAN notebook
- notebooks: This is where all the research regarding the data generation is, some of it is not used, this is mostly exploration about the 
- Data: Where we have 4 already compressed dataset such that you can skip the Data Loader part from the training_GAN notebook. Inside there is also a folder for the test data
- Results: Where you can see multiple different trained models and what they plot on the training set
- training_GAN: The notebook for loading and compressing the data first, then creation of the model and training of it
- predict_Notebook: the notebook which load a trained model and do a prediction on a single image
- report: the report that goes with the project
- README.md: this file

## How to use it

- Basically you can look into the exploratory notebooks to see how we got there
- Then you can create a datase using Automatic1111 webui : https://github.com/AUTOMATIC1111/stable-diffusion-webui
- Choose the images that you like and put them in a similar structure such a the ones in the Data_modification directory (helps to look at the notebook inside the training_x directories first)
- When you have your dataset you can run the training_GAN notebook to create your model
- Finally you can look at its prediction in the predict_Notebook
