# Mammogram Mass Analyzer
This is a free desktop computer aided diagnosis (CAD) tool that uses computer vision to detect and localize masses on full field digital mammograms.
It's a flask app that's running on the desktop. Internally there are two Yolov5L ensembled models that were trained on data from the VinDr-Mammo dataset. The model ensemble has a validation accuracy of 0.65 and a validation recall of 0.63.


<br>
<img src="https://github.com/vbookshelf/Mammogram-Mass-Analyzer/blob/main/images/sample_image.png" height="350"></img>
<i>Sample mammogram from the VinDr-Mammo dataset</i><br>
<br>

My aim was to create a proof of concept for a free desktop computer aided diagnosis (CAD) system that could be used as an aid when diagnosing breast cancer. Unlike a web app, this tool does not need an internet connection and there are no monthly costs for hosting and web server rental. I think a desktop tool could be helpful to radiologists in private practice and to medical non-profits that work in remote areas. 

This app can also be used as an example when building other computer vision desktop Flask apps. By reviewing the code you'll be able to see:
- How to integrate a Yolov5 model into a flask app
- How to use Ajax to implement asynchronous communication between Flask and the web page
- How to preload images so that the app feels smooth
- How to work with dicom files as input


<br>

## Demo

<br>
<img src="https://github.com/vbookshelf/Mammogram-Mass-Analyzer/blob/main/images/demo1.gif" height="400"></img>
<i>Demo showing what happens after the user selects three dicom files</i><br>
<br>

<br>


## 1 - Main features

- Free to use. Free to deploy. No monthly server rental costs like with a web app.
- Completely transparent. All code is accessible and therefore fully auditable.
- Runs locally without needing an internet connection
- Takes mammograms in dicom format as input
- Can analyze multiple mammograms simultaneously
- Uses the computer’s cpu. A gpu would make the app much faster, but it's not essential.
- Results are explainable because it draws bounding boxes around detected masses
- Patient data remains private because it never leaves the user’s computer
- Easy to customize because this is just a Flask app built using html, css and javascript.

<br>

## 2- Cons

- It’s not a one click setup. The user needs to have a basic knowledge of how to use the command line to set up a virtual environment, download requirements and launch a python app.
- The processing time per image is about 10 seconds because the inference is being done using the CPU.
- When diagnosing breast cancer radiologists look for masses, calcifications and architectural distortions. However, this app can only detect masses. The model was not trained to detect calcifications and architectural distortions because there was not enough data for these classes.
- The amount of positive samples in the training data was limited. The accuracy and recall could be improved with more training data.



<br>

## 3- How to run this app

### First download the project folder from Kaggle

I've stored the project folder (named mammogram-mass-analyzer-v0.0) in a Kaggle dataset.<br>
https://www.kaggle.com/datasets/vbookshelf/mammogram-mass-analyzer-v00

I suggest that you download the project folder from the Kaggle instead of from this GitHub repo. This is because the project folder on Kaggle includes the two trained models. The project folder in this repo does not include the trained models because GitHub does not allow files larger than 25MB to be uploaded.<br>
The models are located inside a folder called TRAINED_MODEL_FOLDER, which is located inside the yolov5 folder:<br>
mammogram-mass-analyzer-v0.0/yolov5/TRAINED_MODEL_FOLDER/

<br>

### Overview

This is a standard flask app. The steps to set up and run the app are the same for both Mac and Windows.

1. Download the project folder.
2. Use the command line to pip install the requirements listed in the requirements.txt file. (It’s located inside the project folder.) 
3. Run the app.py file from the command line.
4. Copy the url that gets printed in the console.
5. Paste that url into your chrome browser and press Enter. The app will open in the browser.

This app is based on Flask and Pytorch, both of which are pure python. If you encounter any errors during installation you should be able to solve them quite easily. You won’t have to deal with the Cuda related package dependency issues that happen when using Tensorflow.

<br>

### Detailed setup instructions

The instructions below are for a Mac. I didn't include instructions for Windows because I don't have a Windows pc and therefore, I could not test the installtion process on windows. If you’re using a Windows pc then please change the commands below to suit Windows. 

You’ll need an internet connection during the first setup. After that you’ll be able to use the app without an internet connection.

If you are a beginner you may find these resources helpful:

The Complete Guide to Python Virtual Environments!<br>
Teclado<br>
(Includes instructions for Windows)<br>
https://www.youtube.com/watch?v=KxvKCSwlUv8&t=947s

How To Create Python Virtual Environments On A Mac<br>
https://www.youtube.com/watch?v=MzuGMSw8la0&t=167s

<br>

```

1. Download the project folder, unzip it and place it on your desktop.
In this repo the project folder is named: mammogram-mass-analyzer-v0.0
Then open your command line console.
The instructions that follow should be typed on the command line. 
There’s no need to type the $ symbol.

2. $ cd Desktop

3. $ cd project_folder

4. Create a virtual environment. (Here it’s named myvenv)
This only needs to be done once when the app is first installed.
You'll need to have python3.8 available on your computer.
When you want to run the app again you can skip this step.
$ python3.8 -m venv myvenv

5. Activate the virtual environment
$ source myvenv/bin/activate

4. Install the requirements.
This only needs to be done once when the app is first installed.
When you want to run the app again you can skip this step.
$ pip install -r requirements.txt

5. Launch the app.
This make take a few seconds the first time.
$ python app.py

6. Copy the url that gets printed out (e.g. http://127.0.0.1:5000)

7. Paste the url into your chrome browser and press Enter. The app will launch in the browser. 

8. To stop the app type ctrl C in the console.
Then deactivate the virtual environment.
$ deactivate

```

There are sample mammograms in the sample_dicom_files folder. You can use them to test the app.

While the app is analyzing, please look in the console to see if there are any errors. If there are errors, please do what’s needed to address them. Then relaunch the app.

<br>

## 4- Model Training and Validation

The model card contains a summary of the training and validation datasets as well as the validation results. I've also included a confusion matrix and classification report. There's also some info about the app. Please refer to this document:<br>
https://github.com/vbookshelf/Mammogram-Mass-Analyzer/blob/main/mammogram-mass-analyzer-v0.0/Model-Card-and-App-Info.pdf

All the project jupyter notebooks are stored in the folder called "Notebooks". There are five notebooks. 
Each notebook was run in one of three locations: Locally on my laptop, on Kaggle and on VAST.<br>
https://github.com/vbookshelf/Mammogram-Mass-Analyzer/tree/main/mammogram-mass-analyzer-v0.0/Notebooks

Exp_06-LOCAL<br>
This contains the code to select the training and validation data from the original VinDr-Mammo dataset.

Exp_49-Kaggle<br>
This contains the code to create 10 folds. Only fold 0 was used for training and validation.

Exp_50-VAST<br>
The code for training and validating the first model.

Exp_51-VAST<br>
The code for training and validating the second model.

Exp_52-Kaggle<br>
The code for ensembling the two models and checking the performance of this ensemble on the Fold 0 validation data.



<br>

## 5- Licenses

All code that I have created is free to use under an MIT license.
 
However, please note that the VinDr-Mammo dataset that was used to train the model is licensed under a PhysioNet Restricted Health Data License 1.5.0. - this means that the data can be used for scientific research only. Therefore, the model that powers this app cannot be used for commercial purposes.<br>
PhysioNet Restricted Health Data License 1.5.0<br>
https://physionet.org/content/vindr-mammo/view-license/1.0.0/

The Ultralytics Yolov5 model is licensed under a GNU General Public License.<br>
https://github.com/ultralytics/yolov5/blob/master/LICENSE

<br>

## 6- Citations

Pham, H. H., Nguyen Trung, H., & Nguyen, H. Q. (2022). VinDr-Mammo: A large-scale benchmark dataset for computer-aided detection and diagnosis in full-field digital mammography (version 1.0.0). PhysioNet. https://doi.org/10.13026/br2v-7517.

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

<br>

## 7- Acknowledgements

Many thanks to Kaggle for the free GPU and other great resources they continue to provide.

I also want to thank the VinDr team for the dataset that they’ve generously made publicly available.

Many thanks to the team at Ultralytics for the Yolov5 model and pre-trained weights they’ve made freely available..

<br>

## 8- References

Introduction to Mammography<br>
https://www.youtube.com/watch?v=dEdR4iOdLh0

Ultralytics Yolov5<br>
https://github.com/ultralytics/yolov5

VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography<br>
https://arxiv.org/abs/2203.11205

VinDr-Mammo Dataset<br>
https://physionet.org/content/vindr-mammo/1.0.0/

The Complete Python Course | Learn Python by Doing in 2022<br>
Udemy<br>
https://www.udemy.com/course/the-complete-python-course/

Flask experiments<br>
https://github.com/vbookshelf/Flask-Experiments

W3.CSS Tutorial<br>
https://www.w3schools.com/w3css/defaulT.asp


<br>

## 9- Acronyms

These are a few acronyms I came across during tis project.


SFM - screen-film mammography<br>
FFDM - full-field digital mammography<br>
BI-RADS - Breast Imaging Reporting and Data System<br>
CADe/x - Computer-aided detection and diagnosis<br>
HMUH - Hanoi Medical University Hospital<br>
H108 - Hospital 108<br>
PACS - Picture Archiving and Communication Systems<br>
CC - craniocaudal (looking through the breast from above)<br>
MLO - mediolateral oblique (looking through the breast from the side)<br>
L - Patient's left<br>
R - Patient's right<br>

