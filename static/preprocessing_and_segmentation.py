import sys
import cv2
import numpy as np
import json
from pprint import pprint
import os

fullpath=os.path.dirname(os.path.realpath(__file__))

timestamp = sys.argv[1]
file_ext = sys.argv[2]

# print '.c/uploads/'+timestamp+'/input.'+file_ext
img1=cv2.imread(fullpath+'/uploads/'+timestamp+'/2.'+file_ext)
img1=cv2.resize(img1,(589,821))
# x123 - Noisefree
cv2.imwrite(fullpath+'/uploads/'+timestamp+'/preprocessed-resized.'+file_ext,img1)