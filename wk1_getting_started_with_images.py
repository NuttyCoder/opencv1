# coding: utf-8

# # <font style="color:rgb(50,120,229)">Overview </font>
# 
# We have already discussed about how an image is formed and how it is stored. In this module, we will dive into the code and check out what are the functions available in OpenCV for manipulating images.
# 
# We will cover the following:
# 1. Image I/O - Read, Write & Display an image
# 2. Image Properties - color, channels, shape, image structure
# 3. Creating new images, accessing pixels and region of interest (ROI)

# # <font style="color:rgb(50,120,229)">Import Libraries</font>

# In[1]:


# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH
import os
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'


# # <font style="color:rgb(50,120,229)">Reading an Image</font>
# OpenCV allows reading different types of images (JPG, PNG, etc). You can load grayscale images colour images or you can also load images with Alpha channel (Alpha channel will be discussed in a later section). It uses the [**`imread`**](https://docs.opencv.org/4.1.0/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56) function which has the following syntax:
# 
# ### <font style="color:rgb(8,133,37)">Function Syntax </font>
# ``` python
# retval	=	cv2.imread(	filename[, flags]	)
# ```
# 
# It has **2 arguments**:
# 
# 1. `retval` is the image if it is successfully loaded. Otherwise it is `None`. This may happen if the filename is wrong or the file is corrupt.
# 2. `Path of the image file`: This can be an **absolute** or **relative** path. This is a **mandatory argument**.
# 3. `Flags`: These flags are used to read an image in a particular format (for example, grayscale/color/with alpha channel). This is an **optional argument** with a default value of `cv2.IMREAD_COLOR` or `1` which loads the image as a color image.
# 
# Before we proceed with some examples, let's also have a look at the `flags` available.
# 
# **Flags**
# 1. **`cv2.IMREAD_GRAYSCALE`** or **`0`**: Loads image in grayscale mode
# 2. **`cv2.IMREAD_COLOR`** or **`1`**: Loads a color image. Any transparency of image will be neglected. It is the default flag.
# 3. **`cv2.IMREAD_UNCHANGED`** or **`-1`**: Loads image as such including alpha channel.
# 

# Let us load this image and discuss further
# <img src="https://www.dropbox.com/s/ed0r779b052o0s2/number_zero.jpg?dl=1" width=100>

# In[3]:


imagePath = os.path.join(DATA_PATH,"images/number_zero.jpg")

# Read image in Grayscale format
testImage = cv2.imread(imagePath,0)
print(testImage)


# We print the 2-dimensional array to see what the image is. You can make out that the image signifies a `0`. 

# ## <font style="color:rgb(50,120,229)">Intensity </font>
# The values printed above are the intensity values of each pixel. 
# 
# **0 means black pixel and as the value increases, it moves towards white. A value of 255 is a white pixel.**

# ## <font style="color:rgb(50,120,229)">Image Properties</font>

# In[4]:


print("Data type = {}\n".format(testImage.dtype))
print("Object type = {}\n".format(type(testImage)))
print("Image Dimensions = {}\n".format(testImage.shape))


# The following observations can be made:
# 1. The datatype of the loaded image is **unsigned int and the depth is 8 bit**
# 1. The image is just a 2-dimesional numpy array with values ranging from **0 to 255**.
# 1. The size or resolution is **13x11** which means **height=13 and witdh=11**. In other words, it has **13 rows and 11 columns**.
# 

# #### <font style = "color:rgb(200,0,0)">NOTE</font>
# It should be kept in mind that in OpenCV, size is represented as a tuple of `widthxheight` or `#columnsX#rows`. But in numpy, the shape method returns size as a tuple of `heightxwidth`.

# # <font style="color:rgb(50,120,229)">Manipulating Pixels</font>
# So, we know that the grayscale image is simply a 2-D array. So, all operations supported on arrays should be available for working with images. Let us start by doing some pixel level manipulations. We will see how to access a particular pixel and modify it.

# ## <font style="color:rgb(50,120,229)">Accessing Pixels</font>
# In the above testImage, we saw that the first pixel has a value of 1. Let us check it.
# 
# Since this is a numpy array, we have zero-based indexing and we can access the first element using the index (0,0).

# In[5]:


print(testImage[0,0])


# #### <font style = "color:rgb(200,0,0)">NOTE on indexing</font>
# 
# As mentioned earlier, since matrices are numpy arrays, the first index will be the `row number` and second index is `column number`. This leads to a lot of confusion since we think of pixels in terms of `(x,y)` or `(column,row)` coordinates and not `(row,column)`
# 
# For example, to access the element at `4th row` and `5th column`, we should use `img[3,4]`. But as we will see in Image annotation section, we will deal with points which are represented as `(x,y)` and thus, the coordinates will be `(4,3)`.

# ## <font style="color:rgb(50,120,229)">Modifying pixel values</font>
# Similarly for modifying the value of a pixel, we can simply assign the value to the pixel. 
# 
# Let's change the value of the first element and check if the image is updated.

# In[6]:


testImage[0,0]=200
print(testImage)


# # <font style="color:rgb(50,120,229)">Manipulating Group of Pixels</font>
# So, now we know how to manipulate individual pixels. But what about a region or group of pixels? It can be done using range based indexing available in python. 
# 
# Let is try to access the values of a region and name it `test_roi`. ( *ROI is an abbreviation for Region of Interest* )

# ## <font style="color:rgb(50,120,229)">Access a region</font>

# In[7]:


test_roi = testImage[0:2,0:4]
print("Original Matrix\n{}\n".format(testImage))
print("Selected Region\n{}\n".format(test_roi))


# ## <font style="color:rgb(50,120,229)">Modifying a region</font>
# Modifying a region is also straightforward. 

# In[8]:


testImage[0:2,0:4] = 111
print("Modified Matrix\n{}\n".format(testImage))


# # <font style="color:rgb(50,120,229)">Displaying an Image</font>
# In the previous section, we printed out the Image matrix and were able to make out what the image was. However, this is not the correct way to visualize images as it wont be possible to print large arrays and make out anything.
# 
# Let's see how we should display the images so that it looks more familiar!
# 
# We can use two functions for displaying an image.

# ### <font style = "color:rgb(200,0,0)">NOTE </font>
# 
# One important thing to note while displaying images is the datatype of the image. The display functions expect the images to be in the following format.
# 1. If the image is in float data type, then the range of values should be between 0 and 1.
# 1. If the image is in int data type, then the range of values should be between 0 and 255.
# 
# Keep this in mind to avoid undesirable outputs while displaying the images.
# 

# ## <font style="color:rgb(50,120,229)">1. Matplotlib's imshow</font>
# This function will be used when we want to display the image in Jupyter Notebook.
# 
# ### <font style = "color:rgb(8,133,37)">Function Syntax</font>
# 
# ```Python:
# None	=	plt.imshow( mat )
# ```
# **Parameters**
# - **`mat`** - Image to be displayed.
# 
# 
# This function takes a many arguments but has only 1 mandatory argument. You can have a look at the [documentation](https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.imshow.html) to see what are the other arguments available.

# In[9]:


plt.imshow(testImage)
plt.colorbar()


# ## <font style="color:rgb(50,120,229)">2. OpenCV's imshow</font>
# This function will be used when we are running the Python script from command line.
# 
# ### <font style = "color:rgb(8,133,37)">Function Syntax</font>
# 
# ```Python:
# None	=	cv2.imshow(	winname, mat	)
# ```
# **Parameters**
# - **`winname`** - Name of the window.
# - **`mat`** - Image to be displayed.

# # <font style="color:rgb(50,120,229)">Additional Display Utility Functions </font>
# There are 4 more functions that are present in OpenCV which are commonly used with **`cv2.imshow`** function.

# ## <font style="color:rgb(50,120,229)">1. cv2.namedWindow</font>
# 
# This function is used to create a display window with a specific name. This name is provided as the first argument of this function. The second argument is a flag which decides whether the window can be **resized** (**`cv2.WINDOW_NORMAL`**) or it should be **fixed** to match the image size (**`cv2.WINDOW_AUTOSIZE`** - **Default flag**).
# 
# ### <font style = "color:rgb(8,133,37)">Function Syntax</font>
# [**`Docs`**](https://docs.opencv.org/4.1.0/d7/dfc/group__highgui.html#ga5afdf8410934fd099df85c75b2e0888b)
# ```Python:
# None	=	cv2.namedWindow(	winname[, flags]	)
# ```
# **Parameters**
# - **`winname`** - Name of the window in the window caption that may be used as a window identifier.
# - **`flags`** - Flags of the window. The supported flags are: (cv::WindowFlags)
# 
# **<font color=green>Can you think of any situation where you would prefer to have a resizable display window?</font>**
# 
# ## <font style="color:rgb(50,120,229)">2. cv2.waitKey</font>
# 
# This function is widely used in image as well as video processing. It is a **keyboard binding function**. Its only argument is time in **milliseconds**. The function waits for specified milliseconds for any keyboard event. If you press any key in that time, the program continues. If **0** is passed, it waits **indefinitely** for a key stroke. It can also be set to detect specific key strokes which can be quite useful in video processing applications, as we will see in later sections.
# 
# ### <font style = "color:rgb(8,133,37)">Function Syntax</font>
# [**`Docs`**](https://docs.opencv.org/4.1.0/d7/dfc/group__highgui.html#ga5628525ad33f52eab17feebcfba38bd7)
# ```Python:
# retval	=	cv2.waitKey(	[, delay]	)
# ```
# **Parameters**
# - **`delay`** - Delay in milliseconds. 0 is the special value that means "forever".
# 
# 
# ## <font style="color:rgb(50,120,229)">3. cv2.destroyWindow</font>
# 
# This function is used to destroy or close a particular display window. The name of the window is provided as an argument to this function.
# 
# ### <font style = "color:rgb(8,133,37)">Function Syntax</font>
# [**`Docs`**](https://docs.opencv.org/4.1.0/d7/dfc/group__highgui.html#ga851ccdd6961022d1d5b4c4f255dbab34)
# ```Python:
# None	=	cv2.destroyWindow(	winname	)
# ```
# **Parameters**
# - **`winname`** - Name of the window to be destroyed
# 
# ## <font style="color:rgb(50,120,229)">4. cv2.destroyAllWindows</font>
# 
# This function is used to destroy all display windows. This function does not take any arguments.
# 
# ### <font style = "color:rgb(8,133,37)">Function Syntax</font>
# [**`Docs`**](https://docs.opencv.org/4.1.0/d7/dfc/group__highgui.html#ga6b7fc1c1a8960438156912027b38f481)
# ```Python:
# None	=	cv2.destroyAllWindows(		)
# ```

# # <font style="color:rgb(50,120,229)">Write the Image to Disk</font>
# In most cases, you would want to save the output of your application. We do this using the [**`imwrite`**](https://docs.opencv.org/4.1.0/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce) function.
# 
# 
# ### <font style = "color:rgb(8,133,37)">Function Syntax</font>
# ```Python:
# retval	=	cv2.imwrite(	filename, img [, params]	)
# ```
# **Parameters**
# - **filename** - String providing the relative or absolute path where the image should be saved.
# - **img** - Image matrix to be saved.
# - **params** - Additional information, like specifying the JPEG compression quality etc. Check the full list [**here**](https://docs.opencv.org/4.1.0/d4/da8/group__imgcodecs.html#ga292d81be8d76901bff7988d18d2b42ac)

# In[10]:


cv2.imwrite("test.jpg",testImage)


# We can see that the above function call returned True which indicates that the file was successfuly saved. Let's check it using the `ls` magic command available in Jupyter Notebook.

# In[11]:


get_ipython().magic('ls -l test.jpg')


# # <font style="color:rgb(50,120,229)">Color Images</font>
# In the previous section, we saw how an image is represented as a 2D Matrix. In this section, we will see how to load color images and discuss a few important points related to color images. 
# 
# Let us load a different image this time.
# <img src="https://www.dropbox.com/s/odrry84c0w6p6rv/musk.jpg?dl=1" width=400>

# In[13]:


# Path of the image to be loaded
# Here we are supplying a relative path
imagePath = os.path.join(DATA_PATH,"images/musk.jpg")

# Read the image
img = cv2.imread(imagePath)
print("image Dimension ={}".format(img.shape))


# There are a a few things to note here : 
# 
# 1. The image in this case has 3 dimensions. 
# 1. The third dimension indicates the number of channels in an image. For most images, this number is 3 ( namely R,G,B ). In some cases, there may be an additional channel (called alpha channel) which contains transparency information of the pixels - More on this later!

# # <font style="color:rgb(50,120,229)">Image Channels</font>
# As mentioned above, the color image consists of multiple channels. Each channel itself is a grayscale image. **`The combination of intensity values of the three channels gives the color that is displayed on the screen`**. There are many color spaces used in practice and we will discuss some of them in the later sections. Let us have a brief look at the most popular color space - the RGB color space.
# 
# In OpenCV, the order of channels R, G and B is reverse. i.e. In the image matrix, the Blue channel is indexed first, followed by the Green Channel and finally the Red Channel. 

# ### <font style = "color:rgb(8,133,37)">Display the image</font>

# In[14]:


# Display image
plt.imshow(img)


# #### <font style = "color:rgb(200,0,0)">Important!</font>
# 
# Did you notice anything weird about the color here? This is because OpenCV uses **BGR** format by default whereas Matplotlib assumes the image to be in **RGB** format. This can be fixed by either 
# 
# * converting the image to RGB colorspace using **`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`** or 
# * reverse the order of channels - **`plt.imshow(img[:,:,::-1])`** swaps the 1st and 3rd channel.

# In[15]:


# Convert BGR to RGB colorspace
imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(imgRGB)


# In[16]:


# We can also use the following - It will reverse the order of the 3rd dimension i.e. channels
plt.imshow(img[:,:,::-1])


# Let us access the 3 channels and display the gray scale images in each channel

# In[17]:


# Show the channels
plt.figure(figsize=[20,5])

plt.subplot(131);plt.imshow(img[:,:,0]);plt.title("Blue Channel");
plt.subplot(132);plt.imshow(img[:,:,1]);plt.title("Green Channel");
plt.subplot(133);plt.imshow(img[:,:,2]);plt.title("Red Channel");


# #### <font style = "color:rgb(8,133,37)">Observation </font>
# We had already mentioned that a white pixel means a high intensity value. If you look at the channels closely and compare them with the original image, you should be able to make out the following observations:
# 1. We can see in the original image that the background is blue in color. Thus, the blue channel is also having higher intensity values for the bakground, whereas the red channel is almost black for the background.
# 1. The face is reddish in color and thus, the red channel has very high values in the face region, while the other channels are a bit lower.
# 1. There is a greenish tint in the original image which is also reflected in the Green channel
# 
# This kind of information extraction can be very useful if you want to build basic applications which take decisions based on color ( more so, using specific color channel )

# # <font style="color:rgb(50,120,229)">Splitting and Merging channels </font>
# An alternate way of working with the individual channels is using split and merge. We can access the channels using an opencv function `cv2.split()` and merge them into a color image using `cv2.merge()`. Let us have a look at how it is done.

# In[18]:


# Split the image into the B,G,R components
b,g,r = cv2.split(img)

# Show the channels
plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(b);plt.title("Blue Channel");
plt.subplot(142);plt.imshow(g);plt.title("Green Channel");
plt.subplot(143);plt.imshow(r);plt.title("Red Channel");

# Merge the individual channels into a BGR image
imgMerged = cv2.merge((b,g,r))
# Show the merged output
plt.subplot(144);plt.imshow(imgMerged[:,:,::-1]);plt.title("Merged Output");


# # <font style="color:rgb(50,120,229)">Manipulating Color Pixels</font>

# We saw how to access and modify individual pixels for a grayscale image. The same does not hold for color images. As we discussed, the color image has 3 channels, when we access a pixel, we get a tuple/array of values from the 3 channels. Similarly, we need to specify a tuple for changing the values in the color image.
# 
# Let us load the grayscale image in color. Note that it still looks the same( black and white ) since all the channels contain the same values.

# In[19]:


imagePath = os.path.join(DATA_PATH,"images/number_zero.jpg")
testImage = cv2.imread(imagePath,1)
plt.imshow(testImage)


# ## <font style="color:rgb(50,120,229)">Access Color pixel</font>

# In[20]:


print(testImage[0,0])


# You can see that the intensity value now has 3 elements - one for each channel

# ## <font style="color:rgb(50,120,229)">Modify Pixels</font>
# 
# Let us change the pixel at 
# - location [0,0] to Yellow ( It is a mixture of Red and Green )
# - location [1,1] to Cyan ( It is a mixture of Blue and Green )
# - location [2,2] to Magenta ( It is a mixture of Blue and Red )

# In[21]:


plt.figure(figsize=[20,20])

testImage[0,0] = (0,255,255)
plt.subplot(131);plt.imshow(testImage[:,:,::-1])

testImage[1,1] = (255,255,0)
plt.subplot(132);plt.imshow(testImage[:,:,::-1])

testImage[2,2] = (255,0,255)
plt.subplot(133);plt.imshow(testImage[:,:,::-1])


# ## <font style="color:rgb(50,120,229)">Modify Region of Interest</font>
# Similar to above, we will change the pixels at specific regions as given below. The code is self-explanatory.

# In[22]:


testImage[0:3,0:3] = (255,0,0)
testImage[3:6,0:3] = (0,255,0)
testImage[6:9,0:3] = (0,0,255)

plt.imshow(testImage[:,:,::-1])


# # <font style="color:rgb(50,120,229)">Images with Alpha Channel</font>

# In images with an alpha channel, each pixel not only has a color value, but also has a numerical transparency value ( between 0 to 255) that defines what will happen when the pixel is placed over another pixel.
# 
# There are 4 channels, i.e. 3 color channels and 1 alpha channel which indicates the transparency.

# In[23]:


# Path of the PNG image to be loaded
imagePath = os.path.join(DATA_PATH,"images/panther.png")

# Read the image
# Note that we are passing flag = -1 while reading the image ( it will read the image as is)
imgPNG = cv2.imread(imagePath,-1)
imgRGB = cv2.cvtColor(imgPNG,cv2.COLOR_BGR2RGB)
plt.imshow(imgRGB)
print("image Dimension ={}".format(imgPNG.shape))
#First 3 channels will be combined to form BGR image
#Mask is the alpha channel of the original image
imgBGR = imgPNG[:,:,0:3]
imgMask = imgPNG[:,:,3]
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(imgBGR[:,:,::-1]);plt.title('Color channels');
plt.subplot(122);plt.imshow(imgMask,cmap='gray');plt.title('Alpha channel');


# You can see the whiskers very clearly in the mask image. The alpha mask is basically a very accurate segmentation of the image. It is useful for creating overlays ( Augmented Reality type of applications ). If you don't have tha alpha mask, then you have to separate out the whiskers from the white background ( see original image above ) which can be very difficult.
# 
# You will create a fun application using alpha mask and PNG image in the next section.
