import argparse
import os, sys
import scipy.io as scio
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

path = os.path.dirname(os.path.dirname('Model/attack_Model/'))
sys.path.append(path)
from attack import attack_model
from Load_Predict import load_and_predict
from tensorflow.keras.applications import inception_v3 as inc_net

from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.keras.applications import resnet50

from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import vgg16

from tensorflow.python.keras.applications.xception import Xception
from tensorflow.keras.applications import xception

N = 50  #Iterations of PGD algorithm
n = 100 #Sampling times of NES algorithm
m = 50
SIGMA = 0.001 #Perturbation adding size of NES algorithm
ETA = 0.05 #Perturbation adding size of PGD algorithm
EPSILON = 0.2 #Change control parameters of upper and lower boundaries of the image
top_k = 5   #Maximum probability of labels that can be queried
ColorMap = 4

LIME_Region = 8
Min_Pixel = 100

inet_model = inc_net.InceptionV3()
res_model = ResNet50()
vgg_model = VGG16()
xcep_model = Xception()


def Parser_Setting():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type = int, default = N)
    parser.add_argument('--n', type = int, default = n)
    parser.add_argument('--m', type = float, default = m)
    parser.add_argument('--sigma', type = float, default = SIGMA)
    parser.add_argument('--eta', type = float, default = ETA)
    parser.add_argument('--epsilon', type = float, default = EPSILON)
    parser.add_argument('--top_k', type = int, default = top_k)
    parser.add_argument('--colormap', type = int, default = ColorMap)
    parser.add_argument('--lime_region', type = int, default = LIME_Region)
    parser.add_argument('--min_pixel', type = int, default = Min_Pixel)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = Parser_Setting()

    ObjCat_101_Inception = attack_model(inc_net, inet_model, image_size = 299)
    ObjCat_101_ResNet = attack_model(resnet50, res_model, image_size = 224)
    ObjCat_256_VGG16 = attack_model(vgg16, vgg_model, image_size = 224)
    ILSVRC_Xception = attack_model(xception, xcep_model, image_size = 299)

    LP_ObjCat_Xception = load_and_predict(ILSVRC_Xception.net, ILSVRC_Xception.model, ILSVRC_Xception.image_size)
    LP_ObjCat_Inception = load_and_predict(ObjCat_101_Inception.net, ObjCat_101_Inception.model, ObjCat_101_Inception.image_size)
    
    path = "data/101_Object_InceptionV3/"
    dataFile = 'x_init.mat'
    x_init = scio.loadmat(dataFile)['init']

    # "choose your dataset and network": "ObjCat_101_Inception","ObjCat_101_ResNet","ObjCat_256_VGG16","ILSVRC_Xception"
    #data_result, x_adv_list = ObjCat_101_Inception.fgsm_model(args, x_init, ismask = "CAM", isOnlyLabel = False)
    data_result, x_adv_list = ObjCat_101_Inception.fgsm_model(args, x_init, ismask = "LIME", isOnlyLabel = False)
   
    data_result.to_csv(path + "store_lime_fgsm.csv", index=False)
    Mask_UI = path + 'x_adv_lime_fgsm.mat'
    scio.savemat(Mask_UI, {'adv': x_adv_list})


