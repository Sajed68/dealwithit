print('This is "Deal With It" command line app', flush=True)
print('importing needed packages...', flush=True)
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
    print('try to save file ', NAME+ '.gif')    
    cmd = "convert  -delay 1x"+str(30)+ " ./tmpdealwithit/* -loop 0  " + NAME + ".gif"
    os.system(cmd)
    last = './tmpdealwithit/'+('%5d'%last).replace(' ', '0')+ '.jpg'
    cmd = "convert  " + NAME + ".gif -delay 100 " + last + " -loop 0 " + NAME + ".gif"
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
    Y = I_RGB.shape[1]
    K = cv2.putText(I_RGB.copy(), 'github.com/Sajed68/dealwithit', (100, Y - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, tuple(colors[i].tolist()))
    for l in range(max(L)):
        bg = Image.fromarray(K)
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
        #pic = bg.resize((H//2, W//2)) ##TODO
        pic = bg.resize((H//2, W//2))
        pic.save('./tmpdealwithit/'+('%5d'%j).replace(' ', '0')+ '.jpg')
        j += 1
        steps += 1

    bg = Image.fromarray(K)
    for k in range(N):
        fg1 = fg[k]
        mk1 = mk[k]
        x1 = x[k]
        y1 = Range[k][-1]
        bg.paste(fg1, (x1, y1),mk1)
    #bg.paste(fg, (x, y-h//2),mk)
    #pic = np.array(bg.getdata()).reshape(W, H, C).astype(np.uint8)
    pic = bg.resize((H//2, W//2)) ##TODO
    pic.save('./tmpdealwithit/'+('%5d'%j).replace(' ', '0')+ '.jpg')
    return j, max(L)

print('loading face detector model for dlib package...')
# load face detector model
detector = dlib.get_frontal_face_detector()
model = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(model)

print('waing for loading an image ...')
# load image:
address = tk.filedialog.askopenfile().name
NAME = address.split('/')[-1].split('.')[0]

I = cv2.imread(address)
if I is None:
    print('there is no face in this picture... :(')
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

print('detecting faces ...')
# detect face:
faces = detector(gray)
numb_faces = len(faces)
shapes = []
if numb_faces == 0:
    print("There's now faces here. good luck.")
    sys.exit()

##TODO select which faces:
print('Select which faces you want to deal with them?')

key = -1
Enter = ord('\n') # == 10
colors = np.random.randint(0, 256, size=(numb_faces, 3))  
selected_faces = []
selected = -1
number = 0
while key != Enter:
    K = I.copy()
    for i in range(numb_faces):
        x1, y1 = faces[i].tl_corner().x, faces[i].tl_corner().y
        pt1 = (x1, y1)
        x2, y2 = faces[i].br_corner().x, faces[i].br_corner().y
        pt2 = (x2, y2)
        if i == number:
            cv2.rectangle(K, pt1, pt2, tuple(colors[i].tolist()), 2)
        if selected_faces.count(i) != 0:
            K = cv2.putText(K, 'selected', (x1, y1), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, tuple(colors[i].tolist()))
    cv2.imshow('Select one of those faces..., then press Enter', K)
    key = cv2.waitKey()
    if key == 83:
        number += 1
        number %= numb_faces
    if key == ord(' '):
        if selected_faces.count(number) > 0:
            selected_faces.remove(number)
        else:
            selected_faces.append(number)
        selected_faces = list(set(selected_faces))
else:
    if selected_faces == []:
        selected_faces = list(range(numb_faces))
    else:
        faces = [faces[i] for i in selected_faces]
        numb_faces = len(faces)
        
cv2.destroyAllWindows()

print('processing ...')
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
print('saving pics in temp folder ...')
os.system('mkdir ./tmpdealwithit')
j, fps = save_pics(shapes)
print('creating gif using image magic ...')
create_gif(j, fps)
os.system('rm -d -r ./tmpdealwithit')
print("Deal with it ...\n I'm done, have nice day... :-)")
