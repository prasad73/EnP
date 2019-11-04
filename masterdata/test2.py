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
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
#camera.ExposureTime.SetValue(5000)
#x = input("enter spacebar to start")
#if x == " ":
#    grab_result = camera.GrabOne(1000)
#converter = pylon.ImageFormatConverter()
#converter.OutputPixelFormat = pylon.PixelType_RGB8packed
#converted = converter.Convert(grab_result)
#img = converted.GetArray()
#imageio.imsave('test.jpg', img)
def classifier(num):
    knn = joblib.load('masterdata/knn_model.pkl')
    features_list = []
    features_label = []
    training_digit_image=scipy.misc.imresize(num, (40,40), interp='bilinear', mode=None)
    training_digit = color.rgb2gray(training_digit_image)
    df= hog(training_digit_image, orientations=8, pixels_per_cell=(4,4), cells_per_block=(2, 2))
    features_list.append(df)
    features  = np.array(features_list, 'float64')
    spec=knn.predict(features)
    return spec
def contrast(img):
    image =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y,x]<=180:
                image[y,x] = 0
    return img

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
    rows,cols,a = image.shape
    xshift,yshift=-53,16
    M = np.float32([[1,0,-xshift],[0,1,-yshift]])
    dst = cv2.warpAffine(image,M,(cols,rows))
    return dst
def specfind(image):
    spec1 = []
    spec2 = []
    img = image.copy()
    fuse1 = img[1390:1390+500,1780:1780+240]
    fuse2 = img[1930:1930+200,2778:2778+500]
    fuse2 = imutils.rotate_bound(fuse2, -90)
    cv2.imwrite("fuse1.jpg",fuse1)
    cv2.imwrite("fuse2.jpg",fuse2)
    i=0
    fuses = [fuse1,fuse2]
    for fuse in fuses:
        fuse00 = fuse[:,:,0]
        fuse22= fuse[:,:,2]
        y1=400
        x1=80
        x11,y11=0,0
        for x in range(10,100,1):
            for y in range(341,401,1):
                if fuse00[y,x]>130 and y<y1 and x<x1:
                    x11,y11 = x,y
                    y1=y
                    x1=x
        print("coordinates are",x11,y11)
        if (x11,y11)==(0,0):
            print("applying rotation")
            fuse = imutils.rotate_bound(fuse, 180) 
            fuse00 = fuse[:,:,0]
            fuse22= fuse[:,:,2]
            y1=450
            x1=80
            x11,y11=0,0
            for x in range(10,100,1):
                for y in range(341,420,1):
                    if fuse00[y,x]>130 and y<y1 and x<x1:
                        x11,y11 = x,y
                        y1=y
                        x1=x
        print("coordinates are",x11,y11)
        cropped1 = fuse[y11:(y11+50),x11:(x11+170)]
        date=str(now.day)+'/'+str(now.month)+'/'+str(now.year)+"  "+str(now.hour)+':'+str(now.minute)+':'+str(now.second)
        name="specs/"+str(now.day)+str(now.month)+str(now.year)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"_"+str(i)+".jpg"
        cv2.imwrite(name,cropped1)
        roi1 = fuse00[320:330,50:60]
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
            num = cropped1[2:60,82:142]
            spec2 = classifier(num)
            print(spec2)
            spec = pytesseract.image_to_string(cropped1,lang = 'eng', config="--psm 6 -c tessedit_char_whitelist=012345678ABCDEGHIJKLMNOPQRSTUVWXY-Z -c tessedit_char_blacklist= 9F$!abcdefghijklmnopqrstuvwxyz§")
            print(spec)
            if c1<=80 and '200' in spec and (x11,y11)!=(0,0):
                print("fuse detected and serial matched")
                stat1 = True
                cv2.rectangle(img,(1800,1388),(1800+250,1388+530),(0,255,0), 10)
            elif c1<=80 and '200' not in spec and (x11,y11)!=(0,0):
                print("fuse detected but serial not matched")
                stat1 = False
                cv2.rectangle(img,(1800,1388),(1800+250,1388+530),(0,0,255), 10)       
            else:
                print("fuse not detected")
                stat1 = False  
                cv2.rectangle(img,(1800,1388),(1800+250,1388+530),(0,0,255), 10) 
        else:
            num = cropped1[2:40,110:160]
            #spec = classifier(num)
           # print(spec)
            spec = pytesseract.image_to_string(cropped1,lang = 'eng', config="--psm 6 -c tessedit_char_whitelist=012345678ABCDEGHIJKLMNOPQRSTUVWXY-Z -c tessedit_char_blacklist= 9F$!abcdefghijklmnopqrstuvwxyz§")
            print(spec)
            if c1<=80 and '175' in spec and (x11,y11)!=(0,0):
                print("fuse detected and serial matched")
                stat2 = True
                cv2.rectangle(img,(2778,1870),(2771+536,1870+241),(0,255,0), 10)
            elif c1<=80 and '175' not in spec and (x11,y11)!=(0,0):
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
    img2 =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    line = [(2645,1339),(2648,1520),(2657,1723),(2657,2098)]
    noline = 0
    for row in line:
        x=row[1]
        y = row[0]
        roi = img2[y-5:y+5,x-5:x+5]
        z = cv2.mean(roi)[0]
        print(x,y,z)
        if z<40:
            print("heat shrink present")
            cv2.rectangle(img,(x-20,y-20),(x+20,y+20),(0,255,0), 10)
        else:
            cv2.rectangle(img,(x-20,y-20),(x+20,y+20),(0,0,255), 10)

            print("heat shrink not present")
            noline=noline+1
    if noline>0:
        return False,img
    else:
        return True,img


def liningcheck(img):
    img2 = img[:,:,2]
    line = [(278,855),(287,3665),(2976,3665),(2990,865),(1080,744),(1080,3773),(1734,736),(1734,3780),(2448,762),(2448,3750),(192,1440),(3072,1440),(170,2219),(3100,2219),(195,3000),(3080,3000)]
    noline = 0
    for row in line:
        x=row[1]
        y = row[0]
        roi = img2[y-5:y+5,x-5:x+5]
        z = cv2.mean(roi)[0]
        print(x,y,z)
        if z<50:
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
    color = [(2268,1310),(2268,1497)]
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
            for y in range(200,250,1):
                cc = crop[y,x]
                if crop2[y,x]>130 and y<y1 and x<x1:
                    x11,y11 = x,y
                    y1=y
                    x1=x
        print("coordinates are",x11,y11)
        print(y1)
        if (x11,y11) == (0,0) or crop2[y11+27,x11]>30:
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
            for y in range(196,240,1):
                cc = crop[y,x]
                if crop2[y,x]>100 and y<y1 and x<x1:
                    x11,y11 = x,y
                    y1=y
                    x1=x
        print("coordinates are",x11,y11)
        print(y1)
        if (x11,y11) == (0,0) or crop2[y11+27,x11]>30:
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
    cvui.text(frame1, int(0.83*width), int(.90*height), 'Developed by Verifygn',(0.0006*height), 0x000000)

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
    cvui.text(frame1, int(0.8*width), int(.1*height), "spec",(0.0001*height), 0x000000)
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
    cvui.text(frame1, int(0.8*width), int(.1*height), "spec",(0.0001*height), 0x000000)
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
        name="savedata/"+str(now.day)+str(now.month)+str(now.year)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"m"+str(now.microsecond)+".jpg"
        cv2.imwrite(name,img)
        stat1,img=specfind(img)
        stat2,img = liningcheck(img)
        stat3,img = colorcheck(img)
        stat4,resultimg = heatshrinkcheck(img)
        stat = [stat1,stat2,stat3,stat4]
        with open(r'log.csv', 'a', newline='') as csvfile:
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
        grab_result = camera.GrabOne(1000)
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_RGB8packed
        converted = converter.Convert(grab_result)
        img = converted.GetArray()
        img = imutils.rotate_bound(img, 180)
        imageio.imsave('test.jpg', img)
        img = cv2.imread('test.jpg')
        crop11=cv2.imread("masterdata/temp1.jpg",0)
        crop22 = cv2.imread("masterdata/template2.jpg",0)
    #tmethod='cv2.TM_SQDIFF_NORMED'
    #tmethod='cv2.TM_CCORR_NORMED'
    #tmethod='cv2.TM_CCOEFF'
    #tmethod='cv2.TM_CCOEFF_NORMED'
        tmethod='cv2.TM_SQDIFF'
        (xx1,yy1,matchimg)=findcoordinates(img,tmethod,crop11)

        print (xx1,yy1)
        (xx2,yy2,matchimg)=findcoordinates(img,tmethod,crop22)

        print (xx2,yy2)
        print (xx2,yy2)
        sx1=xx1-1269
        sy1=yy1-920
        sx2=xx2-2100
        sy2=yy2-924
        print (sx1,sy1,sx2,sy2)
        sx=int((sx1+sx2)/2)
        sy=int((sy1+sy2)/2)
    #shifted=imageshift(image,sx,sy)
        if(sx<5 or sx>-5 and sy<5 or sy>-5):
            shifted=imageshift(img,sx,sy)
            print("image shifted")
        else:
            shifted=img.copy()
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
