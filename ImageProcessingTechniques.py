import cv2
import numpy as np

# Create a VideoCapture object and read from input file
video = cv2.VideoCapture('myVideo.mpeg')

#compute the total number of frames
total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
# Compute the number of frames per second, according to the OpenCV version       
if int(major_ver)  < 3 :
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
else :
    fps = video.get(cv2.CAP_PROP_FPS)


#Default resolutions of the frame are obtained and convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))

 
#Create VideoWriter object.The output is stored in 'filtered_video.mpeg' file.
filtered_video = cv2.VideoWriter('filtered_video.mpeg',cv2.VideoWriter_fourcc(*'MPEG'), fps, (frame_width,frame_height))


# 0-5sec
#Define a function that computes the GrayScale conversion
def cvtGrayScale(img):
    blurred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame_ = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR) #Converting it, because the video saver expects 3 channels and not 1
    return frame_


# 5-10sec
#Define a function that computes the smoothening effect, by using Gaussian Filter
def cvtGaussianFilter(img):
     blurred = cvtGrayScale(img)
     frame_ = cv2.GaussianBlur(blurred,(31,31),9) # σ=31
     
     return frame_
 
    
# 10-15sec
#Define a function that computes the smoothening effect, by using Bilateral Filter 
def cvtBilateralFilter(img):
    
    num_bilateral = 3  # number of bilateral filtering steps
    
    blurred = cvtGrayScale(img)
    
    #Repeatedly apply Bilateral filter
    #When applying the filter many times, in this video, one can see that the frames 
    #are being blurred to some extent, but at the same time the coloured objects are well distinguished and bright
    #in the foreground. The edges are also preserved well
    for _ in range(num_bilateral):
        blurred = cv2.bilateralFilter(blurred, d=21, sigmaColor=33, sigmaSpace=33)    
    
    return blurred


# 20-25sec
#Define a function that grabs a specific object in RGB color space
def RGBthresholding(img):

    #Define the approximate range of values of the orange cup. 
    #On the left and the right side of the cup, small gray regions exist, 
    #which, apparently, are excluded on the grabbed object.  
    bgr = [30, 70, 170]
    threshold_ = 30  #The optimal threshold has the value of the blue channel
 
    #Compute min, max values
    min_bgr = np.array([bgr[0] - threshold_, bgr[1] - threshold_, bgr[2] - threshold_])
    max_bgr = np.array([bgr[0] + threshold_, bgr[1] + threshold_, bgr[2] + threshold_])
 
    mask_ = cv2.inRange(img, min_bgr, max_bgr) #Find the mask of orange pixels
    frame_ = cv2.bitwise_and(img, img, mask = mask_) #Apply the mask to get the orange cup from the image 
    
    return frame_


# 25-30sec
#Define a function that grabs a specific object in HSV color space
def HSVthresholding(img):        
    
    #Convert initially the image from RGB to HSV color space
    HSV_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
	
    bgr = [30, 70, 170]
    threshold_ = 30
    
    #Convert now the previous bgr pixel (bgr=[30, 70, 170]) in HSV color space 
    #For converting one pixel to another color space, the 1D array is converted to a 3D array first
    #and then convert it to HSV and getting the first element     
    hsv = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2HSV)[0][0]
 
    #Compute min, max values
    min_hsv = np.array([hsv[0] - threshold_, hsv[1] - threshold_, hsv[2] - threshold_])
    max_hsv = np.array([hsv[0] + threshold_, hsv[1] + threshold_, hsv[2] + threshold_])
 
    mask_ = cv2.inRange(HSV_image, min_hsv, max_hsv) #Find the mask of orange pixels
    frame_ = cv2.bitwise_and(HSV_image, HSV_image, mask = mask_) #Apply the mask to HSV the image
    
    return frame_

  
    
# 30-35sec
#Define a function that improves the grabbing by using binary morphological operations      
def morphological_op(img):
    
    img= HSVthresholding(img) 
    
    kernel = np.ones((57,57),np.uint8)
    #Choose the closing operation, which actually is dilation followed by erosion
    frame_ = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return frame_ 


# 35-45sec
#Define a function that converts the initial video to cartoon
def cvtCartoon(img):
    
    #Apply initially a bilateral filter to reduce the color palette of the video frame.
    
    num_down = 2       # number of downsampling steps
    num_bilateral = 7  # number of bilateral filtering steps
    
    #Downsample the RGB frame using Gaussian pyramid
    #In this way, the bilateral filtering is speeded up 
    img_color = img
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
        
    #Repeatedly apply small Bilateral filter instead of
    # applying one large filter
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
        
    #Upsample frame to original size
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
        
    #Reduce noise in the initial video frames, using a median filter    
    # Convert the frame to grayscale and apply median filtering
    blurred = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(blurred, 7)

    #Detect and enhance edges
    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, blockSize=9, C=2)    

    #Combine the two frames
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
       
    
    return img_cartoon


# 45-60sec
#Define a function that changes the black color in the cartoon video part, with white color
def ChangeCartoonImage(img):
    img_ = cvtCartoon(img)
    img_[np.where((img_ == [0,0,0]).all(axis = 2))] = [255,255,255]
   
    return img_


#Initialiaze a counter to count the frames        
counter =1
        
# Read until video is completed
while(video.isOpened()):
 
  # Capture frame-by-frame
  ret, frame = video.read()
  
  if ret==True:  
    
    if counter < 150.0:                       # 0-5sec
        frame = cvtGrayScale(frame)
    elif counter < 300.0:                     # 5-10sec
        frame = cvtGaussianFilter(frame)
    elif counter < 450.0:                     # 10-15sec
        frame = cvtBilateralFilter(frame)
    elif counter < 600.0:                     # 15-20sec
        frame=frame                           #Present the video as it is
    elif counter < 750.0:                     # 20-25sec
        frame = RGBthresholding(frame)        
    elif counter < 900.0:                     # 25-30sec
        frame = HSVthresholding(frame)
    elif counter < 1050.0:                    # 30-35sec
        frame = morphological_op(frame)
    elif counter < 1350.0:                    # 35-45sec
        frame = cvtCartoon(frame)
    else:                                     # 45-60sec 
        frame = ChangeCartoonImage(frame)

    
    #Display the processed video
    cv2.imshow('Frame',frame)
    
    #Write the processed video
    filtered_video.write(frame) 
    
    
    # Press the letter q on keyboard to exit if you wish
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    counter += 1
  
    # Break the loop
  else: 
    break

#Release the processed video
filtered_video.release()

# When everything done, release the video capture object
video.release()
 
# Close all the frames
cv2.destroyAllWindows()