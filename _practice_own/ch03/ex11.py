import sys, os
sys.path.append(os.pardir) #make parental directory file accessive
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image   #python image library module

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) #convert numpy-based image data into PIL object
    pil_img.show()

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize = False) #make data flattened

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)  #flattened data reshaping
print(img.shape)

img_show(img)