import numpy as np
import cv2
import scipy.ndimage
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.externals import joblib
import imutils
import PIL.Image
import time
import pytesseract
import cvui
import os
from matplotlib import pyplot as plt
import time
import datetime
import yaml
from pypylon import pylon
import imageio
import csv
from numpy import genfromtxt
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
width=1360
height=768
frame1 = np.zeros((height,width, 3), np.uint8)
#camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
#camera.ExposureTime.SetValue(5000)
#x = input("enter spacebar to start")
#if x == " ":
#    grab_result = camera.GrabOne(1000)
#converter = pylon.ImageFormatConverter()
#converter.OutputPixelFormat = pylon.PixelType_RGB8packed
#converted = converter.Convert(grab_result)
#img = converted.GetArray()
#imageio.imsave('test.jpg', img)
def speclocfind(fuse):
    fusehsv = cv2.cvtColor(fuse,cv2.COLOR_RGB2HSV)        
    fuse00 = fusehsv[:,:,1]
    fuse22= fuse[:,:,2]
    y1=400
    x1=80
    x11,y11=0,0
    for x in range(30,80,1):
        for y in range(330,401,1):
            if fuse00[y,x]<100 and (y<y1 or x<x1):
                x11,y11 = x,y
                y1=y
                x1=x
                #print(x11,y11)
    print("coordinates are",x11,y11)
    if (x11,y11)==(0,0):
        print("applying rotation")
        fuse = imutils.rotate(fuse, 180)
        fusehsv = cv2.cvtColor(fuse,cv2.COLOR_RGB2HSV) 
        fuse00 = fusehsv[:,:,1]
        cv2.imwrite("fusehsv.jpg",fuse00)
        fuse22= fuse[:,:,2]
        y1=400
        x1=80
        x11,y11=0,0
        for x in range(30,80,1):
            for y in range(320,400,1):
                if ((fuse00[y,x]<100) and (y<y1 or x<x1)):
                    y1=y
                    x1=x
                    x11,y11 = x,y
                    #print(x11,y11)
    print("coordinates are",x11,y11)
    return fuse,fuse00,x11,y11

def classifier(num):
    knn = joblib.load('masterdata/knn_model.pkl')
    features_list = []
    features_label = []
    training_digit_image=scipy.misc.imresize(num, (40,40), interp='bilinear', mode=None)
    training_digit = color.rgb2gray(training_digit_image)
    df= hog(training_digit_image, orientations=8, pixels_per_cell=(8,8), cells_per_block=(2, 2))
    features_list.append(df)
    features  = np.array(features_list, 'float64')
    spec=knn.predict(features)
    return spec
def contrast(image):
    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1.5
    beta=5
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            new_image[y,x] = np.clip(alpha*image[y,x] + beta, 0, 255)
    return new_image

def findcoordinates(image,meth,template):
    img = image.copy()
    img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    method = eval(meth)
    res = cv2.matchTemplate(img,template,method)
    resmax=np.max(res)
    loc = np.where( res == resmax)
    for pt in zip(*loc[::-1]):
        min_val = pt[0]
        max_val = pt[1]
        print("left top",min_val,max_val)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print (min_val, max_val, min_loc, max_loc)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    return (top_left[0],top_left[1],res)

def imageshift(image,xshift,yshift):
    #xshift,yshift = -40,20
    rows,cols,a = image.shape
    M = np.float32([[1,0,-xshift],[0,1,-yshift]])
    dst = cv2.warpAffine(image,M,(cols,rows))
    return dst
def specfind(image):
    spec1 = []
    spec2 = []
    img = image.copy()
    fuse1 = img[1412:1412+500,1764:1764+240]
    fuse2 = img[1890:1890+240,2744:2744+500]
    fuse2 = imutils.rotate_bound(fuse2, -90)
    cv2.imwrite("fuse1.jpg",fuse1)
    cv2.imwrite("fuse2.jpg",fuse2)
    i=0
    fuse1=cv2.imread("fuse1.jpg")
    fuse2=cv2.imread("fuse2.jpg")
    fuses = [fuse1,fuse2]
    for fuse in fuses:       
        fuse,fuse00,x11,y11=speclocfind(fuse)
        cropped1 = fuse[(y11-5):(y11+55),x11-20:(x11+180)]
        cropped1 = cv2.fastNlMeansDenoisingColored(cropped1,None,10,10,7,21)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        cropped1 = cv2.filter2D(cropped1, -1, kernel)
        date=str(now.day)+'/'+str(now.month)+'/'+str(now.year)+"  "+str(now.hour)+':'+str(now.minute)+':'+str(now.second)
        name="specs/"+str(now.day)+str(now.month)+str(now.year)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"_"+str(i)+".jpg"
        cv2.imwrite(name,cropped1)
        roi1 = fuse00[y11+90:y11+100,x11:x11+10]
        c1 = np.mean(roi1)
        print("c1 is",c1)

    #cropped2=cv2.resize(cropped2,(xx2,yy2), interpolation =  cv2.INTER_LINEAR)
    #cropped1=cv2.resize(cropped1,(xx1,yy1), interpolation =  cv2.INTER_LINEAR)
    #cropped2 = cv2.cvtColor(cropped2, cv2.COLOR_BGR2GRAY) 
        #spec = pytesseract.image_to_string(cropped1,lang = 'eng', config="--psm 6 -c tessedit_char_whitelist=0123456789ABCDEGHIJKLMNOPQRSTUVWXY-Z -c tessedit_char_blacklist= F$!abcdefghijklmnopqrstuvwxyz§")
    
        #print(spec)
    #spec2 = pytesseract.image_to_string(cropped2,lang = 'eng', config="--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJLMNOPQRSTUVWXY-Z -c tessedit_char_blacklist= ()$!Kabcdef(ghijkl)mno<pqr>stuvwxyz§")
    #print("second",spec2)
    #fuse1 = img[1388:1388+530,1800:1800+250]
    #fuse2 = img[1880:1880+240,2778:2778+536]
        if i==0:
            num =cropped1[16:55,104:166]
            namenum = "num/"+str(now.day)+str(now.month)+str(now.year)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"_"+str(i)+".jpg"
            cv2.imwrite(namenum,num)
            num = cv2.cvtColor(num,cv2.COLOR_RGB2GRAY)
            ret,num = cv2.threshold(num, 150, 255, cv2.THRESH_BINARY)
            spec3 = classifier(num)
            print(spec2)
            spec = pytesseract.image_to_string(cropped1,lang = 'eng', config="--psm 6 -c tessedit_char_whitelist=012345678ABCDEGHIJKLMNOPQRSTUVWXY-Z -c tessedit_char_blacklist= 9F$!abcdefghijklm nopqrstuvwxyz§")
            spec2 = pytesseract.image_to_string(num,lang = 'eng', config="--psm 6 -c tessedit_char_whitelist=012345678ABCDEGHIJKLMNOPQRSTUVWXY-Z -c tessedit_char_blacklist= 9F$!abcdefghijklm nopqrstuvwxyz§")
            print(spec,spec2,spec3)
            if c1>=220 and '20' in (spec or spec2) and (x11,y11)!=(0,0):
                print("fuse detected and serial matched")
                stat1 = True
                cv2.rectangle(img,(1800,1388),(1800+250,1388+530),(0,255,0), 10)
            elif c1>=220 and '20' not in (spec or spec2) and (x11,y11)!=(0,0):
                print("fuse detected but serial not matched")
                stat1 = False
                cv2.rectangle(img,(1800,1388),(1800+250,1388+530),(0,0,255), 10)       
            else:
                print("fuse not detected")
                stat1 = False  
                cv2.rectangle(img,(1800,1388),(1800+250,1388+530),(0,0,255), 10) 
        else:
            num = cropped1[8:54,96:150]
            namenum = "num/"+str(now.day)+str(now.month)+str(now.year)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"_"+str(i)+".jpg"
            cv2.imwrite(namenum,num)
            num = cv2.cvtColor(num,cv2.COLOR_RGB2GRAY)
            ret,num = cv2.threshold(num, 90, 255, cv2.THRESH_BINARY)
            spec3 = classifier(num)
           # print(spec)
            spec = pytesseract.image_to_string(cropped1,lang = 'eng', config="--psm 6 -c tessedit_char_whitelist=012345678ABCDEGHIJKLMNOPQRSTUVWXY-Z -c tessedit_char_blacklist= 9F$!abcdefghijklmnopqrstu vwxyz§")
            spec2 = pytesseract.image_to_string(cropped1,lang = 'eng', config="--psm 6 -c tessedit_char_whitelist=012345678ABCDEGHIJKLMNOPQRSTUVWXY-Z -c tessedit_char_blacklist= 9F$!abcdefghijklmnopqrstu vwxyz§")
            spec = spec.replace(" ", "")
            print(spec,spec2,spec3)
            if c1>=220 and '17' in (spec or spec2) and (x11,y11)!=(0,0):
                print("fuse detected and serial matched")
                stat2 = True
                cv2.rectangle(img,(2778,1870),(2771+536,1870+241),(0,255,0), 10)
            elif c1>=220 and '17' not in (spec or spec2) and (x11,y11)!=(0,0):
                print("fuse detected but serial not matched")
                stat2 = False
                cv2.rectangle(img,(2778,1870),(2771+536,1870+241),(0,0,255), 10)       
            else:
                print("fuse not detected")
                stat2 = False  
                cv2.rectangle(img,(2778,1870),(2771+536,1870+241),(0,0,255), 10)  
        i=i+1
    if stat1 and stat2 == True:
        stat = True
    else:
        stat = False
    return stat,img
def heatshrinkcheck(img):
    #img2 =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2=img[:,:,0]
    line1 = [(2580,1356),(2580,1520)]
    line2=[(2586,1700),(2596,2050)]
    noline = 0
    for row in line2:
        x=row[1]
        y = row[0]
        roi = img2[y-7:y+7,x-7:x+7]
        z = cv2.mean(roi)[0]
        print(x,y,z)
        if z>100:
            print("heat shrink present")
            cv2.rectangle(img,(x-20,y-20),(x+20,y+20),(0,255,0), 10)
        else:
            cv2.rectangle(img,(x-20,y-20),(x+20,y+20),(0,0,255), 10)

            print("heat shrink not present")
            noline=noline+1
    for row in line1:
        x=row[1]
        y = row[0]
        roi = img2[y-7:y+7,x-7:x+7]
        z = cv2.mean(roi)[0]
        print(x,y,z)
        if z<100:
            print("heat shrink present")
            cv2.rectangle(img,(x-20,y-20),(x+20,y+20),(0,255,0), 10)
        else:
            cv2.rectangle(img,(x-20,y-20),(x+20,y+20),(0,0,255), 10)

            print("heat shrink not present")

    if noline>0:
        return False,img
    else:
        return True,img
def liningcheck(img):
    img2 = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img2 = img2[:,:,1]
    cv2.imwrite("hsv.jpg",img2)
    line = [(285,876),(285,3644),(2962,3615),(2962,866),(196,2208),(3081,2208),(208,1600),(1592,728),(1592,3761),(3064,1600),(202,2928),(3058,2928),(960,744),(960,3744),(2268,775),(2268,3735)]
    noline = 0
    for row in line:
        x=row[1]
        y = row[0]
        roi = img2[y-10:y+10,x-5:x+5]
        #roi = contrast(roi)
        z = np.mean(roi)
        print(x,y,z)
        if (z>100):
            print("Lining is present")
            cv2.rectangle(img,(x-80,y-80),(x+80,y+80),(0,255,0), 20)
            #cv2.circle(img,(x,y),100,(255,255,255), 20)
        else:
            cv2.rectangle(img,(x-80,y-80),(x+80,y+80),(0,0,255), 20)
            print("Lining not present")
            noline=noline+1
    if noline>0:
        return False,img
    else:
        return True,img
def colorcheck(img):
    img2 = img[:,:,2]
    pix = img2.copy()
    color = [(2268,1330),(2268,1497)]
    color2 = [(2268,1691),(2268,2065)]
    num = 0
    for clr in color:
        a=clr[1]
        b = clr[0]
        print(a,b)
        crop = img[b-156:b+220,a-40:a+40]
        crop2 = crop[:,:,0]
        crop3 = crop[:,:,2]
        cv2.imwrite("crop3.jpg",crop3)
        y1=250
        x1=80
        x11,y11=0,0
        for x in range(47,50,1):
            for y in range(170,250,1):
                cc = crop[y,x]
                if crop2[y,x]>130 and y<y1 and x<x1:
                    x11,y11 = x,y
                    y1=y
                    x1=x
        print("coordinates are",x11,y11)
        print(y1)
        if (x11,y11) == (0,0) or crop2[y11+27,x11]>50:
            a=clr[1]
            b = clr[0]
            print(a,b)
            crop = img[b-156:b+220,a-40:a+40]
            crop2 = crop[:,:,0]
            crop3 = crop[:,:,2]
            print("applying rotation and flipping")
            crop2 = imutils.rotate_bound(crop2, 180)
            crop2 = cv2.flip(crop2,1)
            crop3 = imutils.rotate_bound(crop3, 180)
            crop3 = cv2.flip(crop3,1)            
            y1=300
            x1 =60
            x11,y11=0,0
            for x in range(47,50,1):
                for y in range(196,260,1):
                    if crop2[y,x]>100 and y<y1 and x<x1:
                        x11,y11 = x,y
                        y1=y
                        x1=x
        print("coordinates are",x11,y11)
        roi2 = crop3[y11-33:y11-23,x11+11:x11+22]
        roi1 = crop3[y11-65:y11-60,x11-5:x11+5]
        cv2.imwrite("roi.jpg",roi1)
        now=datetime.datetime.now()
        date=str(now.day)+'/'+str(now.month)+'/'+str(now.year)+"  "+str(now.hour)+':'+str(now.minute)+':'+str(now.second)
        name="small_fuse/"+str(now.day)+str(now.month)+str(now.year)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"m"+str(a)+".jpg"
        cv2.imwrite(name,crop)
        print(name,crop.shape)
        c1 = np.mean(roi1)
        c2 = np.mean(roi2)
        print("color",c1,c2)
        if c1<=100 and c2>100:
            print("green")
            cv2.rectangle(img,(a-40,b-156),(a+40,b+220),(0,255,0), 10)
        else:
            print("no resistor present at location")
            cv2.rectangle(img,(a-40,b-156),(a+40,b+220),(0,0,255), 10)
            num = num+1
    for clr in color2:
        a=clr[1]
        b = clr[0]
        print(a,b)
        crop = img[b-156:b+220,a-40:a+40]
        crop2 = crop[:,:,0]
        crop3 = crop[:,:,2]
        y1=250
        x1=80
        x11,y11=0,0
        for x in range(47,50,1):
            for y in range(196,250,1):
                cc = crop[y,x]
                if crop2[y,x]>100 and y<y1 and x<x1:
                    x11,y11 = x,y
                    y1=y
                    x1=x
        print("coordinates are",x11,y11)
        print(y1)
        if (x11,y11) == (0,0) or crop2[y11+27,x11]>50:
            a=clr[1]
            b = clr[0]
            print(a,b)
            crop = img[b-156:b+220,a-40:a+40]
            crop2 = crop[:,:,0]
            crop3 = crop[:,:,2]
            print("applying rotation and flipping")
            crop2 = imutils.rotate_bound(crop2, 180)
            crop2 = cv2.flip(crop2,1)
            crop3 = imutils.rotate_bound(crop3, 180)
            crop3 = cv2.flip(crop3,1)            
            y1=300
            x1=80
            x11,y11=0,0
            for x in range(47,50,1):
                for y in range(196,260,1):
                    if crop2[y,x]>130 and y<y1 and x<x1:
                        x11,y11 = x,y
                        y1=y
                        x1=x
        print("coordinates are",x11,y11)
        roi2 = crop3[y11-33:y11-23,x11+11:x11+22]
        roi1 = crop3[y11-65:y11-60,x11-5:x11+5]
        now=datetime.datetime.now()
        date=str(now.day)+'/'+str(now.month)+'/'+str(now.year)+"  "+str(now.hour)+':'+str(now.minute)+':'+str(now.second)
        name="small_fuse/"+str(now.day)+str(now.month)+str(now.year)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"m"+str(a)+".jpg"
        cv2.imwrite(name,crop)
        print(name,crop.shape)
        c1 = np.mean(roi1)
        c2 = np.mean(roi2)
        print("color",c1,c2)
        if c1>=150 and c2>150:
            print("yellow")
            cv2.rectangle(img,(a-40,b-156),(a+40,b+220),(0,255,0), 10)
        else:
            print("no resistor present at location")
            cv2.rectangle(img,(a-40,b-156),(a+40,b+220),(0,0,255), 10)
            num = num+1
    if num>0:
        return False,img
    else:
        return True,img
mainwindow = "Main Window"
def page():
    head=cv2.imread("masterdata/head.png")
    head=imutils.resize(head, height=50)
    [hy,hx,aa]=head.shape
    #print hx,hy
    frame1[:] = (255, 255, 255)
    #print frame1.shape
    frame1[int(0.001*height):int(0.001*height)+hy, int(.2*width):int(.2*width)+hx]=head
    logo=cv2.imread("masterdata/logo.jpg")
    logo=imutils.resize(logo, width=int(width*.15))
    [ly,lx,aa]=logo.shape
    frame1[int(0.02*height):int(0.02*height)+ly, int(.75*width):int(.75*width)+lx]=logo
    #cvui.text(frame1, int(0.43*width), int(.92*height), 'Developed by Verifygn',(0.0005*height), 0x000000)
    #cvui.text(frame1, int(0.43*width), int(.92*height), 'Developed by Verifygn',(0.0005*height), 0x000000)
    cvui.text(frame1, int(0.85*width), int(.90*height), 'Developed by Verifygn',(0.0005*height), 0x000000)

def page1():
    cvui.text(frame1, int(0.2*width), int(.4*height), "Waiting for the Inspection...",(0.002*height), 0x000000)

def page2(date,pcount,mimg,simg,stat):
    ok = cv2.imread('masterdata/ok.png')
    notok = cv2.imread('masterdata/notok.png')
    tpic=cv2.imread(mimg)
    ok = imutils.resize(ok, height=int(height*0.03))
    notok = imutils.resize(notok, height=int(height*0.03))
    [oy,ox,oa] = ok.shape
    tpic=imutils.resize(tpic, height=int(height*.84))
    [ty,tx,aa]=tpic.shape
    frame1[int(0.075*height):int(0.075*height)+ty, int(.01*width):int(.01*width)+tx]=tpic
    ipic=cv2.imread(simg)
    ipic=imutils.resize(ipic, height=int(height*.3))
    [iy,ix,aa]=ipic.shape
    frame1[int(0.15*height):int(0.15*height)+iy, int(.73*width):int(.73*width)+ix]=ipic
    bpic=cv2.imread("masterdata/datebg.png")
    bpic=imutils.resize(bpic, width=int(width*.28))
    [by,bx,ba]=bpic.shape
    frame1[int(0.71*height):int(0.71*height)+by, int(.72*width):int(.72*width)+bx]=bpic
    print(oy,ox,oa)
    i=0.45
    for status in stat:
        i=i+0.05
        print(i)
        if status==True:
            frame1[int(i*height):int(i*height)+oy, int(.9*width):int(.9*width)+ox]=ok
        elif status==False:
            frame1[int(i*height):int(i*height)+oy, int(.9*width):int(.9*width)+ox]=notok            
    cvui.text(frame1, int(0.75*width), int(.5*height), "Large Fuse",(0.001*height), 0x000000)
    cvui.text(frame1, int(0.75*width), int(.55*height),"Outer sealing" ,(0.001*height), 0x000000)
    cvui.text(frame1, int(0.75*width), int(.6*height), "Small Fuse",(0.001*height), 0x000000)
    cvui.text(frame1, int(0.75*width), int(.65*height), "Heatshrink",(0.001*height), 0x000000)
    cvui.text(frame1, int(0.72*width), int(.75*height), date,(0.001*height), 0x000000)
    cvui.text(frame1, int(0.72*width), int(.81*height), pcount,(0.001*height), 0x000000)
def page3(date,pcount,mimg,simg):
    tpic=cv2.imread(mimg)
    tpic=imutils.resize(tpic, height=int(height*.84))
    [ty,tx,aa]=tpic.shape
    frame1[int(0.075*height):int(0.075*height)+ty, int(.01*width):int(.01*width)+tx]=tpic
    ipic=cv2.imread(simg)
    ipic=imutils.resize(ipic, height=int(height*.3))
    [iy,ix,aa]=ipic.shape
    frame1[int(0.15*height):int(0.15*height)+iy, int(.73*width):int(.73*width)+ix]=ipic
    bpic=cv2.imread("masterdata/datebg.png")
    bpic=imutils.resize(bpic, width=int(width*.28))
    [by,bx,ba]=bpic.shape
    frame1[int(0.71*height):int(0.71*height)+by, int(.72*width):int(.72*width)+bx]=bpic           
    cvui.text(frame1, int(0.75*width), int(.5*height), "Outer sealing",(0.001*height), 0x000000)
    cvui.text(frame1, int(0.75*width), int(.55*height), "Large Fuse",(0.001*height), 0x000000)
    cvui.text(frame1, int(0.75*width), int(.6*height), "Small Fuse",(0.001*height), 0x000000)
    cvui.text(frame1, int(0.75*width), int(.65*height), "Heatshrink",(0.001*height), 0x000000)
    cvui.text(frame1, int(0.72*width), int(.75*height), date,(0.001*height), 0x000000)
    cvui.text(frame1, int(0.72*width), int(.81*height), pcount,(0.001*height), 0x000000)
first=0
with open("masterdata/serial.yaml", 'r') as stream:
    countdata = yaml.load(stream)
partcount=countdata["count"]
#partcount=0
prost=0
cvui.init (mainwindow)
while True:

    k = cv2.waitKey(10) & 0xFF
    if first==0:
        page()
        page1()
        first=1

    if(prost==1):
        prost=0
        print("starting inspection")
        cv2.imwrite("shifted.jpg",shifted)
        img = shifted.copy()
        now=datetime.datetime.now()
        date=str(now.day)+'/'+str(now.month)+'/'+str(now.year)+"  "+str(now.hour)+':'+str(now.minute)+':'+str(now.second)
        stat1,img=specfind(img)
        stat2,img = liningcheck(img)
        stat3,img = colorcheck(img)
        stat4,resultimg = heatshrinkcheck(img)
        stat = [stat1,stat2,stat3,stat4]
        with open(r'masterdata/log.csv', 'a', newline='') as csvfile:
            fieldnames = ['Count', 'Specification','Lining','Fuse_colour','Heatshrinks']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Count':partcount, 'Specification':stat1,'Lining':stat2,'Fuse_colour':stat3,'Heatshrinks':stat4})
        cv2.imwrite('out.jpg',resultimg)
        print(stat1,stat2,stat3,stat4)
        if(stat1 and stat2 and stat3 and stat4 ==True):
            sy="masterdata/ok.png"
            im="out.jpg"
            page()
            page2(date,count,im,sy,stat)
            name="okdata/"+str(now.day)+str(now.month)+str(now.year)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"m"+str(now.microsecond)+".jpg"
            cv2.imwrite(name,resultimg)
        else:
            sy="masterdata/notok.png"
            im="out.jpg"
            page()
            page2(date,count,im,sy,stat)
            name="notokdata/"+str(now.day)+str(now.month)+str(now.year)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"m"+str(now.microsecond)+".jpg"
            cv2.imwrite(name,resultimg)
    if(k==32):
        #grab_result = camera.GrabOne(1000)
        #converter = pylon.ImageFormatConverter()
        #converter.OutputPixelFormat = pylon.PixelType_RGB8packed
        #converted = converter.Convert(grab_result)
        #img = converted.GetArray()
        #img = imutils.rotate_bound(img, 180)
        #imageio.imsave('test.jpg', img)
        img = cv2.imread('test.jpg')
        crop11=cv2.imread("masterdata/crop1.jpg",0)
        crop22 = cv2.imread("masterdata/crop2.jpg",0)
    #tmethod='cv2.TM_SQDIFF_NORMED'
    #tmethod='cv2.TM_CCORR_NORMED'
    #tmethod='cv2.TM_CCOEFF'
    #tmethod='cv2.TM_CCOEFF_NORMED'
        tmethod='cv2.TM_SQDIFF'
        (xx1,yy1,matchimg)=findcoordinates(img,tmethod,crop11)

        print (xx1,yy1)
        (xx2,yy2,matchimg)=findcoordinates(img,tmethod,crop22)

        print (xx1,yy1)
        print (xx2,yy2)
        sx1=xx1-1274
        sy1=yy1-1918
        sx2=xx2-2612
        sy2=yy2-1622
        print (sx1,sy1,sx2,sy2)
        sx=int((sx1+sx2)/2)
        sy=int((sy1+sy2)/2)
        print(sx,sy)
        if(-50<sx<40 and -20<sy<30):
            shifted=imageshift(img,sx,sy)
            print("image shifted")
        else:
            shifted=imageshift(img,-40,20)
            print("image not shifted")
        partcount=partcount+1
        serialdata={"count":partcount}
        with open("masterdata/serial.yaml", "w") as f:
            yaml.dump(serialdata, f)
        now=datetime.datetime.now()
        date=str(now.day)+'/'+str(now.month)+'/'+str(now.year)+"  "+str(now.hour)+':'+str(now.minute)+':'+str(now.second)
        count=str(partcount)
        sy="masterdata/ins.png"
        im="test.jpg"
        page()
        page3(date,count,im,sy)
        cvui.imshow(mainwindow, frame1)
        print ("serching")
        prost=1
    cvui.imshow(mainwindow, frame1)
    if k == 27:
        break
cv2.destroyAllWindows()
