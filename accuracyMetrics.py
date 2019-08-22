import numpy as np
import cv2
import os
import re

## Function to resize ground truth images or any image into 256 X 256 size
## Input is file path of image which needs to be altered
def prep_img_size(file_path):
    file_read = cv2.imread(file_path, 0)
    
    if file_read.shape != (256, 256):
        print("File dimension is " + str(file_read.shape) + " ... changing now")
        resized = cv2.resize(file_read, (256,256), interpolation = cv2.INTER_AREA)
        cv2.imwrite(file_path, resized) 
    else:
        print("File is already of the right dimension")
        
## Function returns a list of tuples, each tuple being a ground truth file and it's corresponding 
## predict file, eg 2.png and 2_predict.png
## Input param is directory path which contains all the images
def tuplize(dir_path):
    files = os.listdir(dir_path) # List of files in the directory

    gt_regex = re.compile(r'\d+\.png')
    gt_files = list(filter(gt_regex.search, files)) # Extract ground truth files

    predicted_files = list(filter(lambda x: x.endswith('_predict.png'), files)) # Extract predictions

    gtp_list = []
    # Find the couple from gt_files and predicted files, make a tuple and append to list
    for i in gt_files:
        file_name = str.split(i, '.')[0]
        predicted = list(filter((lambda x:re.match(r""+re.escape(file_name)+"_predict.png", x)), 
                           predicted_files))
        gtp_list.append((i,predicted[0]))
        
    
    return(gtp_list)

## Function to get overall Intersection over Union scores
## Input is directory path and list of tuples of gt and prediction
## Output is a score value
## Also called Jaccard Index
def iou_score(dir_path, gtp_tup_list):
    scores_lst = []
    for i in gtp_tup_list:
        target_file = dir_path + i[0]
        prediction_file = dir_path + i[1]

        target = cv2.imread(target_file, 0) # Read in gray scale
        prediction = cv2.imread(prediction_file, 0)
        
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection) / np.sum(union)
        
        scores_lst.append(iou_score) # Collect score for each tuple
        
    avg_score = sum(scores_lst) / len(scores_lst)
    avg_score = round(avg_score, 2)
    
    return(avg_score)


## Function to calculate dice coefficient.
## Very similar to IoU score, except in formula
def dice_coefficient(dir_path, gtp_tup_list):
    scores_lst = []
    for i in gtp_tup_list:
        target_file = dir_path + i[0]
        prediction_file = dir_path + i[1]

        target = cv2.imread(target_file, 0) # Read in gray scale
        prediction = cv2.imread(prediction_file, 0)
        
        intersection = np.logical_and(target, prediction)
        dice_coeff = 2 * np.sum(intersection) / (np.sum(target) + np.sum(prediction))
        
        scores_lst.append(dice_coeff)
    
    avg_score = sum(scores_lst) / len(scores_lst)
    avg_score = round(avg_score, 2)
    
    return(avg_score)



##############################################################################################
##### README
# Examples for how to use

## Testing prep_img_size function
# img_filepath = 'verify/'
# real = '0.png'
# target_file = img_filepath + real

# prep_img_size(target_file)

# ## Testing tuplize function
# gtp = tuplize(img_filepath)
# gtp


# ## IoU avg scores function testing
# scores = iou_score(img_filepath, gtp)
# scores