import cv2
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image

import numpy as np
import tensorflow as tf

# %matplotlib inline

from nst_utils import *

#!wget --output-document=imagenet-vgg-verydeep-19.mat 'https://storage.googleapis.com/marketing-files/colab-notebooks/style-transfer/imagenet-vgg-verydeep-19.mat'

model = load_vgg_model("/home/ravi/Desktop/imagenet-vgg-verydeep-19.mat")
print(model)



content_img = cv2.imread()

import os

CONTENT_IMAGE_FN = list(content_img)[0]
CONTENT_IMAGE_FN_temp = CONTENT_IMAGE_FN.strip().replace(" ", "_")

if CONTENT_IMAGE_FN != CONTENT_IMAGE_FN_temp:
  os.rename(CONTENT_IMAGE_FN, CONTENT_IMAGE_FN_temp)
  CONTENT_IMAGE_FN = CONTENT_IMAGE_FN_temp
  
print("Content image filename :", CONTENT_IMAGE_FN)

# %matplotlib inline
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
img = plt.imread(CONTENT_IMAGE_FN)
plt.axis('off')
plt.title('Content image')
plt.imshow(img)

content_image = scipy.misc.imread(CONTENT_IMAGE_FN)
imshow(content_image)

content_image= scipy.misc.imresize(content_image,(487,626))

content_image.shape

"""#### mask image"""

mask_img = cv2.imread()

import os

MASK_IMAGE_FN = list(mask_img)[0]
MASK_IMAGE_FN_temp = MASK_IMAGE_FN.strip().replace(" ", "_")

if MASK_IMAGE_FN != MASK_IMAGE_FN_temp:
  os.rename(MASK_IMAGE_FN, MASK_IMAGE_FN_temp)
  MASK_IMAGE_FN = MASK_IMAGE_FN_temp
  
print("Content image filename :", MASK_IMAGE_FN)

# %matplotlib inline
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
img = plt.imread(MASK_IMAGE_FN)
plt.axis('off')
plt.title('mask image')
plt.imshow(img)

mask_image = scipy.misc.imread(MASK_IMAGE_FN)
imshow(mask_image)
content_image= scipy.misc.imresize(content_image,(487,626))



def compute_content_cost(a_C, a_G):
    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.reshape(a_C,[-1])
    a_G_unrolled = tf.reshape(a_G,[-1])
    J_content = tf.reduce_sum((a_C_unrolled-a_G_unrolled)**2)/(4*n_H*n_W*n_C)
        
    return J_content

style_img = cv2.imread()

STYLE_IMAGE_FN = list(style_img)[0]
STYLE_IMAGE_FN_temp = STYLE_IMAGE_FN.strip().replace(" ", "_")

if STYLE_IMAGE_FN != STYLE_IMAGE_FN_temp:
  os.rename(STYLE_IMAGE_FN, STYLE_IMAGE_FN_temp)
  STYLE_IMAGE_FN = STYLE_IMAGE_FN_temp
  
print("Style image filename :", STYLE_IMAGE_FN)

style_image = scipy.misc.imread(STYLE_IMAGE_FN)
imshow(style_image)

style_image= scipy.misc.imresize(style_image,(487,626))

style_image.shape

def gram_matrix(A):
   
    GA =tf.matmul(A,tf.transpose(A))
    
    return GA

def compute_layer_style_cost(a_S, a_G):
  
    #print(a_G.shape)
   
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.reshape(a_S, [n_H*n_W, n_C])
    a_G = tf.reshape(a_G, [n_H*n_W, n_C])
    
    

    GS = gram_matrix(tf.transpose(a_S))
    GG = gram_matrix(tf.transpose(a_G))


    J_style_layer = tf.reduce_sum((GS - GG)**2) / (4 * n_C**2 * (n_W * n_H)**2)
    
    return J_style_layer

STYLE_LAYERS = [
    ('conv3_1', 0.2)]

STYLE_LAYERS_BACK = [
    ('conv1_1', 0.5),
    ('conv2_1', 0.5)]

def compute_style_cost_back(model,STYLE_LAYERS_BACK):
    
    

    J_back = 0

    for layer_name, coeff in STYLE_LAYERS_BACK:

        out = model[layer_name]

        
        a_S = sess.run(out)

        a_G = out #elementwise
        
        
        J_back_style_layer = compute_layer_style_cost(a_S, a_G)

       
        J_back += coeff * J_back_style_layer

    return J_back

def compute_style_cost(model, STYLE_LAYERS):
    
    

    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        out = model[layer_name]

        
        a_S = sess.run(out)

        a_G = out
        
        
        J_style_layer = compute_layer_style_cost(a_S, a_G)

       
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, J_back, alpha = 10, beta = 40, gamma = 25):
    
   
    J = alpha*J_content+beta*J_style+gamma*J_back
  
    
    return J

tf.reset_default_graph()

sess = tf.InteractiveSession()

content_image = reshape_and_normalize_image(content_image)

content_image.shape

mask_image = reshape_and_normalize_image(mask_image)

mask_image.shape

style_image = reshape_and_normalize_image(style_image)
print(style_image.shape)

generated_image = generate_noise_image(content_image)
imshow(generated_image[0])

generated_image = (content_image)

model = load_vgg_model("/home/ravi/Desktop/imagenet-vgg-verydeep-19.mat")

sess.run(model['input'].assign(content_image))


out = model['conv5_3']


a_C = sess.run(out)

a_G = out


J_content = compute_content_cost(a_C, a_G)

sess.run(model['input'].assign(style_image))


J_style = compute_style_cost(model, STYLE_LAYERS)

sess.run(model['input'].assign(style_image))
J_back = compute_style_cost_back(model, STYLE_LAYERS_BACK)

J = total_cost(J_content, J_style, J_back, alpha=5, beta=20, gamma=25)

optimizer = tf.train.AdamOptimizer(2.0)

train_step = optimizer.minimize(J)

def model_nn(sess, input_image, num_iterations = 250):
        
    sess.run(tf.global_variables_initializer())
    
    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):
    
        
        _=sess.run(train_step)
       
        generated_image = sess.run(model['input'])
        
        if i%20 == 0:
            Jt, Jc, Js, Jb = sess.run([J, J_content, J_style, J_back])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            print("background cost = " + str(Jb))
            
            
            save_image(str(i) + ".jpg", generated_image)
    
    
    save_image('generated_image.jpg', generated_image)
    
    return generated_image

model_nn(sess, generated_image)

fig = plt.figure(figsize=(10, 10))
img = plt.imread("generated_image.jpg")
plt.axis('off')
plt.title('Generated image')
plt.imshow(img)

"""# New Section"""

def load_mask(mask, shape):
    mask = mask # Grayscale mask load
    width, height, _ = shape
    mask = imresize(mask, (width, height), interp='bicubic').astype('float32')

    # Perform binarization of mask
    mask[mask <= 127] = 0
    mask[mask > 128] = 255

    max = np.amax(mask)
    mask /= max

    return mask

# util function to apply mask to generated image
def mask_content(content, generated, mask):
    width, height, channels = generated.shape

    for i in range(width):
        for j in range(height):
            if mask[i, j] == 0.:
                generated[i, j, :] = content[i, j, :]

    return generated

generated_image = imread("generated_image.jpg")
img_width, img_height, channels = generated_image.shape

content_image = imread(CONTENT_IMAGE_FN)
content_image = imresize(content_image, (img_width, img_height), interp='bicubic')


content_mask = imread(MASK_IMAGE_FN,mode="L")
mask = load_mask(content_mask, generated_image.shape)

img = mask_content(content_image, generated_image, mask)
cv2.imshow(img)

#imsave(image_path, img)

#print("Image saved at path : %s" % image_path)

img.shape

mask_updated = reshape_and_normalize_image(img)
print(mask_updated.shape)

def model_nn(sess, input_image, num_iterations = 100):
        
    sess.run(tf.global_variables_initializer())
    
    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):
    
        
        _=sess.run(train_step)
       
        generated_image = sess.run(model['input'])
        
        if i%20 == 0:
            Jt, Jc, Js, Jb = sess.run([J, J_content, J_style, J_back])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            print("background cost = " + str(Jb))
            
            
            save_image(str(i) + ".jpg", generated_image)
    
    
    save_image('generated_image.jpg', generated_image)
    
    return generated_image

model_nn(sess, mask_updated)

fig = plt.figure(figsize=(10, 10))
img = plt.imread("generated_image.jpg")
plt.axis('off')
plt.title('Generated image')
plt.imshow(img)



from google.colab.patches import cv2_imshow



import argparse
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.misc import imread, imresize, imsave, fromimage, toimage


# Util function to match histograms
def match_histograms(source, template):
        

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

  
def original_color_transform(content, generated, mask=None, hist_match=0, mode='YCbCr'):
    generated = fromimage(toimage(generated, mode='RGB'), mode=mode)  
    if mask is None:
        if hist_match == 1:
            for channel in range(3):
                generated[:, :, channel] = match_histograms(generated[:, :, channel], content[:, :, channel])
        else:
            generated[:, :, 1:] = content[:, :, 1:]
    else:
        width, height, channels = generated.shape

        for i in range(width):
            for j in range(height):
                if mask[i, j] == 1:
                    if hist_match == 1:
                        for channel in range(3):
                            generated[i, j, channel] = match_histograms(generated[i, j, channel], content[i, j, channel])
                    else:
                        generated[i, j, 1:] = content[i, j, 1:]

    generated = fromimage(toimage(generated, mode=mode), mode='RGB') 
    return generated



def load_mask(mask_path, shape):
    mask = imread(mask_path, mode="L") 
    width, height, _ = shape
    mask = imresize(mask, (width, height), interp='bicubic').astype('float32')

    
    mask[mask <= 127] = 0
    mask[mask > 128] = 255

    mask /= 255
    mask = mask.astype(np.int32)

    return mask


hist_match=1
if hist_match == 1:
    image_suffix = "_histogram_color.png"
    mode = "RGB"
else:
    image_suffix = "_original_color.png"
    mode = "YCbCr"


generated_image = imread("generated_image.jpg")
img_width, img_height, _ = generated_image.shape

content_image =  imread("n.jpeg")
content_image = imresize(content_image, (img_width, img_height), interp='bicubic')

mask_transfer = 0
if mask_transfer:
    mask_img = load_mask(args.mask, generated_image.shape)
else:
    mask_img = None

img = original_color_transform(content_image, generated_image, mask_img, hist_match, mode=mode)
cv2.imshow(img)

