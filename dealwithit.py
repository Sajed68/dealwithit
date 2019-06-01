import numpy as np
import cv2
import dlib
# from matplotlib import cm, pyplot as plt
from imutils.face_utils import visualize_facial_landmarks, shape_to_np
from imutils import rotate, rotate_bound
from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog
import sys


def create_gif(last, fps): #(inputPath, outputPath, delay, finalDelay, loop):
    # grab all image paths in the input directory
    #imagePaths = sorted(list(paths.list_images(inputPath)))
    
    # remove the last image path in the list
    #lastPath = imagePaths[-1]
    #imagePaths = imagePaths[:-1]
 
    # construct the image magick 'convert' command that will be used
    # generate our output GIF, giving a larger delay to the final
    # frame (if so desired)
    #cmd = "convert -delay {} {} -delay {} {} -loop {} {}".format(
    #    delay, inputPath.join('/*'), finalDelay, lastPath, loop,
    #    outputPath)
    cmd = "convert  -delay 1x"+str(30)+ " ./tmp/* -loop 0  animation.gif"
    os.system(cmd)
    last = './tmp/'+('%5d'%last).replace(' ', '0')+ '.jpg'
    cmd = "convert  animation.gif -delay 100 " + last + " -loop 0 animation.gif"
    os.system(cmd)


def prepare_glass(shape1):
    left_corner = np.zeros((2), np.int)
    left_corner[0] = shape1[17][0]
    left_corner[1] = shape1[36][1]
    right_corner = np.zeros((2), np.int)
    right_corner[0] = shape1[26][0]
    right_corner[1] = shape1[45][1]

    #VIS = cv2.line(I_RGB.copy(), tuple(left_corner.tolist()), tuple(right_corner.tolist()), (255, 10, 10), 3)

    #plt.imshow(VIS)
    #plt.show()

    eye_angle = np.degrees(np.arctan2(right_corner[1] - left_corner[1], right_corner[0] - left_corner[0]))


    resized_glass = glass.copy() 
    resized_mask = mask.copy() 

    rotated_glass = rotate_bound(resized_glass, eye_angle)
    rotated_mask = rotate_bound(resized_mask, eye_angle)

    length = shape1[26][0] - shape1[17][0]
    h, w, _ = rotated_glass.shape
    h = int(h * length / w)
    resized_glass = cv2.resize(rotated_glass, (length, h))
    resized_mask = cv2.resize(rotated_mask, (length, h))

    x,y = left_corner
    IM = I_RGB.copy()
    al = Image.fromarray(np.ones((I_RGB.shape[0],I_RGB.shape[1]), np.uint8)*255)

    fg = Image.fromarray(resized_glass)
    mk = Image.fromarray(resized_mask)
    #bg.putalpha(al)
    #fg.putalpha(mk)
    return fg, mk, al, h, x, y


def save_pics(shape):
    fg = [];mk=[];al=[]; h=[]; x=[]; y=[]; Range = []
    N = len(shape)
    for shape1 in shape:
        fg1, mk1, al1, h1, x1, y1 = prepare_glass(shape1)
        fg.append(fg1)
        mk.append(mk1)
        al.append(al1)
        h.append(h1)
        x.append(x1)
        y.append(y1)
        Range.append([i for i in range(-h1, y1-h1//2, 5)]+[y1-h1//2])
    
    L = [len(i) for i in Range]
    IDX = np.argsort(L)
    
    
    W, H, C = I.shape
    j = 0

    steps = 0
    finished = []
    
    for l in range(max(L)):
        bg = Image.fromarray(I_RGB)
        if finished != []:
            for k in finished:
                fg1 = fg[k]
                mk1 = mk[k]
                x1 = x[k]
                y1 = Range[k][-1]
                bg.paste(fg1, (x1, y1),mk1)
                
        for k in range(len(shape)):
            fg1 = fg[k]
            mk1 = mk[k]
            x1 = x[k]
            if steps < L[k]:
                y1 =  Range[k][steps]
            else:
                y1 = Range[k][-1]
                if finished.count(k) == 0:
                    finished.append(k)
            bg.paste(fg1, (x1, y1),mk1)

        #pic = np.array(bg.getdata()).reshape(W, H, C).astype(np.uint8)
        pic = bg.resize((H//2, W//2))
        pic.save('./tmp/'+('%5d'%j).replace(' ', '0')+ '.jpg')
        j += 1
        steps += 1

    bg = Image.fromarray(I_RGB)
    for k in range(N):
        fg1 = fg[k]
        mk1 = mk[k]
        x1 = x[k]
        y1 = Range[k][-1]
        bg.paste(fg1, (x1, y1),mk1)
    #bg.paste(fg, (x, y-h//2),mk)
    #pic = np.array(bg.getdata()).reshape(W, H, C).astype(np.uint8)
    pic = bg.resize((H//2, W//2))
    pic.save('./tmp/'+('%5d'%j).replace(' ', '0')+ '.jpg')
    return j, max(L)


# load face detector model
detector = dlib.get_frontal_face_detector()
model = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(model)

# load image:
address = tk.filedialog.askopenfile().name

I = cv2.imread(address)
if I is None:
    sys.exit()

gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

# opt:
I_RGB = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

#fig=plt.figure(figsize=(7,7), dpi= 80, facecolor='w', edgecolor='k')
#plt.imshow(I_RGB, cm.gray)
#plt.show()

# open glass and its mask:
glass = cv2.imread('sunglasses.png')
mask = cv2.imread('sunglasses_mask.png',0)

#plt.imshow(glass)
#plt.show()

# detect face:
faces = detector(gray)
numb_faces = len(faces)
shapes = []
for n in range(numb_faces):
    shape = predictor(gray, faces[n])
    #shape2 = predictor(gray, faces[1])

    shape = shape_to_np(shape)
    #shape2 = shape_to_np(shape2)
    #shape1 = shape2
    shapes.append(shape) # now I added

# opt:
#VIS = visualize_facial_landmarks(I_RGB, shape1)
#VIS = visualize_facial_landmarks(VIS, shape2)

#fig=plt.figure(figsize=(7,7), dpi= 80, facecolor='w', edgecolor='k')
#plt.imshow(VIS);
#plt.show()

j, fps = save_pics(shapes)

create_gif(j, fps)
