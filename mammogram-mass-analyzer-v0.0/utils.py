
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import numpy as np
import pandas as pd
import os
import cv2
import shutil



import torch
import torchvision


def read_xray(path, voi_lut=True, fix_monochrome=True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way

    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data


def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image

    im = Image.fromarray(array)

    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)

    return im



# Draw the boxes on the image
def draw_bbox(image, xmin, ymin, xmax, ymax, text=None, line_thickness=20):
    """
    Set text=None to only draw a bbox without
    any text or text background.
    E.g. set text='Balloon' to write a
    title above the bbox.

    Output:
    Returns an image with one bounding box drawn.
    The title is optional.
    To draw a second bounding box pass the output image
    into this function again.

    """

    w = xmax - xmin
    h = ymax - ymin

    # Draw the bounding box
    # ......................

    start_point = (xmin, ymin)
    end_point = (xmax, ymax)
    bbox_color = (255, 255, 255)
    bbox_thickness = line_thickness

    image = cv2.rectangle(image, start_point, end_point, bbox_color, bbox_thickness)

    # Draw the background behind the text
    # ....................................

    # Only do this if text is not None.
    if text:
        # Draw the background behind the text
        text_bground_color = (0, 0, 0)  # black
        cv2.rectangle(image, (xmin, ymin - 150), (xmin + w, ymin), text_bground_color, -1)

        # Draw the text
        text_color = (255, 255, 255)  # white
        font = cv2.FONT_HERSHEY_DUPLEX
        origin = (xmin, ymin - 30)
        fontScale = 3
        thickness = 10

        image = cv2.putText(image, text, origin, font,
                            fontScale, text_color, thickness, cv2.LINE_AA)

    return image


# Convert the dicom files to png and store in png_images_dir
def convert_dicom_images_to_png(dicom_file_list):

    # Create png_images_dir.
    # Remember here we are outside the yolov5 folder.
    if os.path.isdir('yolov5/png_images_dir') == False:
        png_images_dir = os.path.join('yolov5', 'png_images_dir')
        os.mkdir(png_images_dir)

    # Prepare the images

    image_size_list = []

    for i, dicom_fname in enumerate(dicom_file_list):

        # Images
        # -------

        # load the image fname
        fname = dicom_fname.split('.')[0]

        # add the .png extension
        png_fname = fname + '.png'

        # Load a dicom image
        path = 'uploads/' + dicom_fname
        image = read_xray(path, voi_lut=True, fix_monochrome=True)

        height = image.shape[0]
        width = image.shape[1]

        image_size_list.append((height, width))

        # Don't resize the image
        # This step just enables the code to run. It doesn't change
        # the input image size.
        # image = resize(image, size=image_size, keep_ratio=False, resample=Image.LANCZOS)

        # Use this line when we are not resizing
        image = Image.fromarray(image)


        # Save the image in the folder
        # that we created.
        dst = os.path.join(png_images_dir, png_fname)
        image.save(dst)

    return image_size_list


def predict_on_all_png_images(model_list, device):

    # change the working directory
    os.chdir('yolov5')

    print(os.getcwd())

    # Yolo Model:
    # Make a prediction on all images in images_dir
    # The model only creates a txt file if it finds objects on an image.

    print('Starting prediction...')

    print(os.listdir('png_images_dir'))

    # Get the path to each trained model
    model_path_0 = f"TRAINED_MODEL_FOLDER/{model_list[0]}"
    model_path_1 = f"TRAINED_MODEL_FOLDER/{model_list[1]}"


    # Execute a shell command to run Yolov5.
    # Here we are ensembling the trained models.
    os.system(f'python detect.py --source "png_images_dir" --weights {model_path_0} {model_path_1} --device {device} --img 1024 --save-txt --save-conf --exist-ok')

    #print(os.listdir('runs/detect/exp/labels'))

    print('Prediction completed.')


def process_predictions(dicom_file_list, image_png_size_list, ABS_PATH_TO_STATIC):
    print('Checking dir...')
    print(os.getcwd())

    # Create pred_images_dir.
    # Remember here we are inside the yolov5 folder
    if os.path.isdir('pred_images_dir') == False:
        pred_images_dir = 'pred_images_dir'
        os.mkdir(pred_images_dir)


    txt_fname_list = os.listdir('runs/detect/exp/labels')

    #print(txt_fname_list)

    # We wull store the num preds for each dicom file
    # in a dict. The dicom file name is the key and
    # the num preds is the value.
    num_preds_dict = {}

    for i, item in enumerate(dicom_file_list):

        dicom_fname = item
        txt_fname = item.split('.')[0] + '.txt'
        png_fname = item.split('.')[0] + '.png'

        if txt_fname in txt_fname_list:

            print('Processing predicted bboxes.')

            # This is how to put the contents of a txt
            # file into a dataframe.

            path = f'runs/detect/exp/labels/{txt_fname}'

            # create a list of column names
            cols = ['class', 'x-center', 'y-center', 'bbox_width', 'bbox_height', 'conf-score']

            # put the file contents into a dataframe
            df_test_preds = pd.read_csv(path, sep=" ", header=None)

            # add the column names to the datafrae
            df_test_preds.columns = cols

            # Get the number of predicted bboxes
            num_bboxes = len(df_test_preds)

            # Add a key value pair to the dict
            num_preds_dict[dicom_fname] = num_bboxes

            # Remember that Yolo preds are normalized.
            # Need to convert them into dimensions for the comp submission.
            # The dimensions that we submit need to be based on the original comp image sizes.

            # image_png_size_list: [(h,w), ...]
            orig_image_h = image_png_size_list[i][0]
            orig_image_w = image_png_size_list[i][1]


            df_test_preds['w'] = df_test_preds['bbox_width'] * orig_image_w
            df_test_preds['h'] = df_test_preds['bbox_height'] * orig_image_h

            df_test_preds['x_cent'] = orig_image_w * df_test_preds['x-center']
            df_test_preds['y_cent'] = orig_image_h * df_test_preds['y-center']

            df_test_preds['xmin'] = df_test_preds['x_cent'] - (df_test_preds['w'] / 2)
            df_test_preds['ymin'] = df_test_preds['y_cent'] - (df_test_preds['h'] / 2)

            df_test_preds['xmax'] = df_test_preds['xmin'] + df_test_preds['w']
            df_test_preds['ymax'] = df_test_preds['ymin'] + df_test_preds['h']

            # Read the image
            path = os.path.join('png_images_dir', png_fname)
            image = cv2.imread(path)

            # Draw the bboxes on the image
            for i in range(0, len(df_test_preds)):
                xmin = int(df_test_preds.loc[i, 'xmin'])
                ymin = int(df_test_preds.loc[i, 'ymin'])
                xmax = int(df_test_preds.loc[i, 'xmax'])
                ymax = int(df_test_preds.loc[i, 'ymax'])

                image = draw_bbox(image, xmin, ymin, xmax, ymax, text=None, line_thickness=10)

            # save the image
            dst = os.path.join(pred_images_dir, png_fname)
            cv2.imwrite(dst, image)

        else:

            # Add a key value pair to the dict
            num_preds_dict[dicom_fname] = 0

            # Read the image
            path = os.path.join('png_images_dir', png_fname)
            image = cv2.imread(path)

            # save the image
            dst = os.path.join(pred_images_dir, png_fname)
            cv2.imwrite(dst, image)



    # Copy the pred_images_dir to the static folder so we can
    # display the images easily.


    # This is dependent on the current working directory.
    abs_path_to_pred_images_dir = os.path.abspath("pred_images_dir")

    src = abs_path_to_pred_images_dir
    dst = os.path.join(ABS_PATH_TO_STATIC, "pred_images_dir")

    shutil.copytree(src, dst)


    # Delete the exp folder
    # change the working directory
    print('Delete exp...')
    # os.chdir('yolov5')
    if os.path.isdir('runs/detect/exp') == True:
        shutil.rmtree('runs/detect/exp')

        print('exp folder deleted.')


    return num_preds_dict



def delete_user_submitted_data():

    """
    Note:
    This function does not delete the images in 'static/pred_images_dir'.
    The app needs the png images in this folder to display them on the main page.
    The 'static/pred_images_dir' folder gets deleted each time the user submits new files.

    """
    # Delete folders and their contents.
    # This is for data security.
    if os.path.isdir('uploads') == True:
        shutil.rmtree('uploads')

    # Delete folders and their contents.
    # This is for data security.
    if os.path.isdir('yolov5/png_images_dir') == True:
        shutil.rmtree('yolov5/png_images_dir')

    # Delete folders and their contents.
    # This is for data security.
    #if os.path.isdir('static/pred_images_dir') == True:
        #shutil.rmtree('static/pred_images_dir')

    # Delete folders and their contents.
    # This is for data security.
    if os.path.isdir('yolov5/pred_images_dir') == True:
        shutil.rmtree('yolov5/pred_images_dir')