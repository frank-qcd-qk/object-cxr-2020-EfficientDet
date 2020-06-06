import numpy as np
import cv2
import os
import csv
import pandas as pd
import argparse
import time
#! EfficientDet
import torch
from torch.backends import cudnn
from efficientdetCore.backbone import EfficientDetBackbone
from efficientdetCore.efficientdet.utils import BBoxTransform, ClipBoxes
from efficientdetCore.utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
#! Preserve for future yolov4


#! Parser for submitter
parser = argparse.ArgumentParser(description='Run Frank submission')
parser.add_argument('input_csv', default='image_path.csv', metavar='INPUT_CSV',
                    type=str, help="input-data-csv-filename")
parser.add_argument('classification_csv', default='classification.csv', metavar='CLASSIFICATION_CSV',
                    type=str, help="output-classification-prediction-csv-path")
parser.add_argument('localization_csv', default='localization.csv', metavar='LOCALIZATION_CSV',
                    type=str, help='output-localization-prediction-csv-path')
args = parser.parse_args()

#! Global submission configurator
classificationWriter = None
localizationWriter = None
OBJECT_SEP = ';'
ANNOTATION_SEP = ' '

#! Detection global configuration
detectionThreashold = 0.1
predictionOverRide = False
predictionMultiboxOverride = False
image2DetectList = None


#!EfficientDet Configuration
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
threshold = 0.2
iou_threshold = 0.2
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True
obj_list = ['foreign_object']
color_list = standard_to_bgr(STANDARD_COLORS)
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
force_input_size = None  # set None to use default size
model = None
compound_coef = 3 #! Change me if change d value!


#! EfficientDet Model Loadings
def init_efficientDet_model():
    global model
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load("src/frankNet/frankNetD3.pth"))
    model.requires_grad_(False)
    model.eval()
    model = model.cuda()
    print("EfficientDetBackbone load complete. Model load complete. ")

def singleImageDetectionEfficientDet(imageIn):
    #imageIn = 'valid_image/08002.jpg' # No detection
    #imageIn = 'valid_image/08001.jpg' # Single Detection
    #imageIn = 'valid_image/08037.jpg' #Multiple Detection
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    ori_imgs, framed_imgs, framed_metas = preprocess(
        imageIn, max_size=input_size)
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
    with torch.no_grad():
        features, regression, classification, anchors = model(x)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)
    print(out)
    confidence = out[0]["scores"]
    print("confidence is:",confidence)
    bbox = out[0]["rois"]
    print("ROIS is:",bbox)
    print("hand over with detection...")
    return confidence,bbox

#! File IO
def fileIOInit():
    global localizationWriter, classificationWriter
    try:
        with open(args.classification_csv, 'w') as csvFile1:
            classificationWriter = csv.writer(csvFile1)
            classificationWriter.writerow(['image_path', 'prediction'])

        with open(args.localization_csv, 'w') as csvFile2:
            localizationWriter = csv.writer(csvFile2)
            localizationWriter.writerow(['image_path', 'prediction'])

    except Exception:
        raise IOError("Cannot initiate CSV file")

    print("File created successfully...")


def ingestList():
    global image2DetectList
    image2DetectList = pd.read_csv(
        args.input_csv, header=None, na_filter=False, names=['path'])


def writer(which, what):
    # which: 1 = classification, 2 = localization
    #what = [path,writing]
    if which == 1:
        with open(args.classification_csv, 'a') as csvFile1:
            classificationWriter = csv.writer(csvFile1)
            classificationWriter.writerow(what)
    else:
        with open(args.localization_csv, 'a') as csvFile2:
            localizationWriter = csv.writer(csvFile2)
            localizationWriter.writerow(what)




#! 1. classification confidence:
"""
image_path,prediction
valid_image/0001.jpg,0.82561
"""


def getClassificationEfficientDet(pathIn, confidencesIn):
    global predictionOverRide, predictionMultiboxOverride
    returner = [pathIn]
    confidenceList = []
    if len(confidencesIn) != 0:
        if predictionOverRide:
            returner.append(str(1.0))
        else:
            returner.append(max(confidencesIn))
    else:
        returner.append(str(0.0))
    writer(1, returner)


#! 2. classification bbox:
"""
image_path,confidence1 point1_x point1_y;confidence2 point2_x point2_y;confidence3 point3_x point3_y;confidence4 point4_x point4_y; ... confidenceK pointK_x pointK_y
valid_image/00004.jpg,0.25 500 700;0.51 320 1800;0.89 750 2200;0.49 1000 1200
"""


def getLocalizationEfficientDet(pathIn, confidenceIn,bboxIn):
    returner = [pathIn]
    if len(confidenceIn) != 0:
        string2Write = ""

        for aConfidence,aBbox in zip(confidenceIn,bboxIn):
            confidence = aConfidence
            bounds = aBbox
            x, y = int(bounds[0]+bounds[2]/2), int(bounds[1]+bounds[3]/2)
            string2Write += (str(round(confidence, 4)) +" "+str(x)+" "+str(y)+";")

        # remove dangling ";"
        string2Write = string2Write[:-1]
        returner.append(string2Write)
    else:
        returner.append('')
    writer(2, returner)


def looper():
    global image2DetectList
    for anImage in image2DetectList.itertuples(index=False):
        anImagePath = anImage[0]
        print("ID: ", anImagePath)
        localConfidence,localBbox = singleImageDetectionEfficientDet(anImagePath)

        getClassificationEfficientDet(anImagePath, localConfidence)
        getLocalizationEfficientDet(anImagePath,localConfidence, localBbox)

        print("", anImagePath, " Detection Complete. Total ",
              len(localConfidence), " foreign Object Observed.")
        # visualize(anImagePath,localBBox)


def metricsCalculator():
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    import matplotlib.pyplot as plt
    import subprocess
    print("++++++++++++++++++++++Metrics Report++++++++++++++++++++++")
    prediction_label = pd.read_csv(args.classification_csv, na_filter=False)
    pred = prediction_label.prediction
    pred = pred.replace(r'^\s*$', 0, regex=True)
    pred = pred.astype(float).values
    #print (pred)
    labels_dev = pd.read_csv('groundTruth.csv', na_filter=False)
    gt = labels_dev.annotation.astype(bool).astype(float).values
    # print(gt)
    # 1.
    acc = ((pred >= .5) == gt).mean()
    fpr, tpr, _ = roc_curve(gt, pred)
    roc_auc = auc(fpr, tpr)

    #2, VisualIze
    fig, ax = plt.subplots(
        subplot_kw=dict(xlim=[0, 1], ylim=[0, 1], aspect='equal'),
        figsize=(6, 6)
    )
    ax.plot(fpr, tpr, label=f'ACC: {acc:.03}\nAUC: {roc_auc:.03}')
    _ = ax.legend(loc="lower right")
    _ = ax.set_title('ROC curve')
    ax.grid(linestyle='dashed')
    print("Report AUC: ", roc_auc, " Report ACC: ", acc)
    plt.show()


def runner():
    init_efficientDet_model()
    fileIOInit()  # Checked running
    ingestList()  # Checked Running
    looper()
    metricsCalculator()

if __name__ == "__main__":
    runner()
    # metricsCalculator()
