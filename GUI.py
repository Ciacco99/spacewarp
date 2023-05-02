import cv2
import glob
import numpy as np

def nothing(x):
    pass

img_start = cv2.imread("Images/0.jpg")

img_1 = cv2.imread("Images/1.jpg")
img_2 = cv2.imread("Images/2.jpg")
img_3 = cv2.imread("Images/3.jpg")
#resize images
img_start.resize(512, 512)
img_1.resize(512, 512)
img_2.resize(512, 512)
img_3.resize(512, 512)


list_img2=[img_1, img_2, img_3]

cv2.namedWindow('image')

# create trackbars for image choice
cv2.createTrackbar('image_choice','image',0, 2,nothing)

# create a switch to save or not the image
switch = '0 : no \n 1 : Save'

cv2.createTrackbar(switch, 'image',0,1,nothing)
save = 0
while(1):
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27: #wait esc key
        break
    
    # get current chosen image
    img_idx = cv2.getTrackbarPos('image_choice','image')
    save = cv2.getTrackbarPos(switch,'image')
    
    img_choice = list_img2[img_idx]
    print(np.shape(img_choice))
    print(np.shape(img_start))

    conc = np.concatenate((img_start, img_choice), axis = 1)
    cv2.imshow('image', conc)

    if save:
        print("pairesaved")
        cv2.destroyAllWindows()
cv2.destroyAllWindows()
