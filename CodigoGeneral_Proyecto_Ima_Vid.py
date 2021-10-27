import cv2
import numpy as np
import random

nivel = 10
# Cargamos la imagen
original = cv2.imread("C:/Users/User/Desktop/Proyecto_imagenes_videos/1.jpg")
 #%%
# Convertimos a escala de grises
gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

ret, th = cv2.threshold(gris, 200, 255, cv2.THRESH_BINARY_INV)
(contornos,_) = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

X = []
Y = []
W = []
H = []
Imagenes_insumo = []

Sub_imag = 0
for i in range(len(contornos)):  
    cnt = contornos[i]
    M = cv2.moments(cnt)
    CX = int(M["m10"]/(M["m00"]+0.001))
    CY = int(M["m01"]/(M["m00"]+0.001))
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)    
    x,y,w,h = cv2.boundingRect(cnt)  
    
    if perimetro > 450:
        X.append(x)
        Y.append(y)
        W.append(w)
        H.append(h)  
        
        imageOut = original[y:y+h,x:x+w]
        Sub_imag = Sub_imag + 1
        Imagenes_insumo.append(imageOut)

if Sub_imag == 1:
    X = []
    Y = []
    W = []
    H = []
    Imagenes_insumo = []
    # Aplicar suavizado Gaussiano
    gauss = cv2.GaussianBlur(gris, (5,5), 0)
    # Detectamos los bordes con Canny
    canny = cv2.Canny(gauss, 50, 150)
    #Buscamos los contornos
    (contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contornos)):  
        cnt = contornos[i]
        M = cv2.moments(cnt)
        CX = int(M["m10"]/(M["m00"]+0.001))
        CY = int(M["m01"]/(M["m00"]+0.001))
        
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)    
        x,y,w,h = cv2.boundingRect(cnt)
        
        if perimetro > 450:
            X.append(x)
            Y.append(y)
            W.append(w)
            H.append(h) 
            
            imageOut = original[y:y+h,x:x+w]
            Imagenes_insumo.append(imageOut)

#%%
##### Combinar primera forma
cv2.imshow("Figuras", original)
cv2.waitKey(0)

k = 0
while k < nivel:
    i = random.randrange(0, len(Imagenes_insumo))
    j = random.randrange(0, len(Imagenes_insumo))
    k = k + 1
    try:
        original[Y[i]:Y[i]+H[j],X[i]:X[i]+W[j]] = Imagenes_insumo[j]
    except:
        None
cv2.imshow("Figuras cambiadas", original)
cv2.waitKey(0)
#%%
##### Combinar segunda forma
cv2.imshow("Figuras", original)
cv2.waitKey(0)

k = 0
while k < nivel:
    i = random.randrange(0, len(Imagenes_insumo))
    j = random.randrange(0, len(Imagenes_insumo))
    k = k + 1
    s_img = Imagenes_insumo[j]
    l_img = original
    y1, y2 = Y[i], Y[i] + H[i]
    x1, x2 = X[i], X[i] + W[i]
    
    alpha_s = s_img[:, :, 2] / 255.0
    alpha_l = 1.0 - alpha_s
    try:
        for c in range(0, 3):
            l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                      alpha_l * l_img[y1:y2, x1:x2, c])
    except:
        None

cv2.imshow("Figuras cambiadas", l_img)
cv2.waitKey(0)

#%%
#### Prueba de colores
# Import the required libraries
import cv2 # opencv version 3.4.2
import numpy as np # numpy version 1.16.3
import matplotlib.pyplot as plt # matplotlib version 3.0.3
from enum import Enum

# Load the source image
src_img = original

# Reconocer figuras en una figura a partir del color
def figureColor(imagen):
    src_img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    # Change colorspace from BGR to HSV
    src_img_hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
    
    # Define limits of yellow HSV values
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([45, 255, 255])
    
    green_lower = np.array([40,100,100])
    green_upper = np.array([80,255,255])
    
    red_lower1 = np.array([0,150,50])
    red_upper1 = np.array([5,255,255])
    red_lower2 = np.array([175,150,50])
    red_upper2 = np.array([180,255,255])
    
    blue_lower = np.array([100, 80, 170])
    blue_upper = np.array([126, 255, 255])
    
    pink_lower = np.array([160, 80, 170])
    pink_upper = np.array([170, 255, 255])
    
    orange_lower = np.array([10,150,180])
    orange_upper = np.array([16,255,255])
    
    mask_red1 = cv2.inRange(src_img_hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(src_img_hsv, red_lower2, red_upper2)
    
    # Filter the image and get the mask
    mask_yellow = cv2.inRange(src_img_hsv, yellow_lower, yellow_upper)
    mask_red = mask_red1 + mask_red2
    mask_green = cv2.inRange(src_img_hsv, green_lower, green_upper)
    mask_blue = cv2.inRange(src_img_hsv, blue_lower, blue_upper)
    mask_pink = cv2.inRange(src_img_hsv, pink_lower, pink_upper)
    mask_orange = cv2.inRange(src_img_hsv, orange_lower, orange_upper)
    
    coloresUser = ['(1)mask_yellow','(2)mask_red','(3)mask_green','(4)mask_blue','(5)mask_pink','(6)mask_orange']
    colores = [mask_yellow,mask_red,mask_green,mask_blue,mask_pink,mask_orange]
    print(*coloresUser, sep="\n")
    print("Seleccione el número del color a probar:")
    color = int(input())
    color = color-1
    
    src_imag_mask = src_img_gray * colores[color]
    
    masked = cv2.bitwise_and(src_img, src_img, mask=src_imag_mask)
    return masked

cv2.imshow("Original Image", src_img)
cv2.imshow("Result", figureColor(src_img))
cv2.waitKey(0)

#%%
#### Cambiar caras
# Importar paquetes necesarios
import numpy as np
import PIL.Image
import tensorflow as tf
import tensorflow_hub as hub

# Cargar modelo pre entrenado desde tf hub
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Funciones auxiliares
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Cargar imágenes
imagen_contenido = load_img("C:/Users/User/Desktop/Proyecto_imagenes_videos/1.jpg")
imagen_estilo = load_img("C:/Users/User/Desktop/Proyecto_imagenes_videos/2.jpg")

# Obtener imagen resultante
stylized_image = hub_module(tf.constant(imagen_contenido), tf.constant(imagen_estilo))[0]
tensor_to_image(stylized_image)

#%%
#### Cuantas esquinas en la imagen
# INTEREST POINTS
# Cuantas esquinas tiene una imagen

image = cv2.imread("C:/Users/User/Desktop/Proyecto_imagenes_videos/1.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_draw = np.copy(image)

# Harris
dst = cv2.cornerHarris(image_gray.astype(np.float32), 2, 3, 0.04)
dst = cv2.dilate(dst, None)
image_draw[dst > 0.01 * dst.max()] = [0, 0, 255]

# Shi-Tomasi
corners = cv2.goodFeaturesToTrack(image_gray, 100, 0.0001, 10)
corners = corners.astype(np.int)
for i in corners:
    x, y = i.ravel()
    cv2.circle(image_draw, (x, y), 3, [255, 0, 0], -1)

# sift and orb
sift = cv2.SIFT_create(nfeatures=50)
orb = cv2.ORB_create(nfeatures=50)

keypoints_sift, descriptors = sift.detectAndCompute(image_gray, None)
keypoints_orb, descriptors = orb.detectAndCompute(image_gray, None)

image_draw = cv2.drawKeypoints(image_gray, keypoints_orb, None)

cv2.imshow("Image", image_draw)
cv2.waitKey(0)

# Buscar afectaciones de imagenes en internet
#%%
from enum import Enum
#### Adivinar el personaje
image = cv2.imread("C:/Users/User/Desktop/Proyecto_imagenes_videos/1.jpg")
def gradient_map(gray):
    # Image derivatives
    scale = 1
    delta = 0
    depth = cv2.CV_16S  # to avoid overflow
    grad_x = cv2.Sobel(gray, depth, 1, 0, ksize=3, scale=scale, delta=delta)
    grad_y = cv2.Sobel(gray, depth, 0, 1, ksize=3, scale=scale, delta=delta)
    grad_x = np.float32(grad_x)
    grad_x = grad_x * (1 / 512)
    grad_y = np.float32(grad_y)
    grad_y = grad_y * (1 / 512)
    # Gradient and smoothing
    grad_x2 = cv2.multiply(grad_x, grad_x)
    grad_y2 = cv2.multiply(grad_y, grad_y)
    # Magnitude of the gradient
    Mag = np.sqrt(grad_x2 + grad_y2)
    # Orientation of the gradient
    theta = np.arctan(cv2.divide(grad_y, grad_x + np.finfo(float).eps))
    return theta, Mag


def orientation_map(gray, n):
    # Image derivatives
    scale = 1
    delta = 0
    depth = cv2.CV_16S  # to avoid overflow
    grad_x = cv2.Sobel(gray, depth, 1, 0, ksize=3, scale=scale, delta=delta)
    grad_y = cv2.Sobel(gray, depth, 0, 1, ksize=3, scale=scale, delta=delta)
    grad_x = np.float32(grad_x)
    grad_x = grad_x * (1 / 512)
    grad_y = np.float32(grad_y)
    grad_y = grad_y * (1 / 512)
    # Gradient and smoothing
    grad_x2 = cv2.multiply(grad_x, grad_x)
    grad_y2 = cv2.multiply(grad_y, grad_y)
    grad_xy = cv2.multiply(grad_x, grad_y)
    g_x2 = cv2.blur(grad_x2, (n, n))
    g_y2 = cv2.blur(grad_y2, (n, n))
    g_xy = cv2.blur(grad_xy, (n, n))
    # Magnitude of the gradient
    Mag = np.sqrt(grad_x2 + grad_y2)
    M = cv2.blur(Mag, (n, n))
    # Gradient local aggregation
    vx = 2 * g_xy
    vy = g_x2 - g_y2
    fi = cv2.divide(vx, vy + np.finfo(float).eps)
    case1 = vy >= 0
    case2 = np.logical_and(vy < 0, vx >= 0)
    values1 = 0.5 * np.arctan(fi)
    values2 = 0.5 * (np.arctan(fi) + np.pi)
    values3 = 0.5 * (np.arctan(fi) - np.pi)
    theta = np.copy(values3)
    theta[case1] = values1[case1]
    theta[case2] = values2[case2]
    return theta, M

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1) high-pass kernels
kernel = cv2.getDerivKernels(1, 0, 3)
kernel = np.outer(kernel[1], kernel[0])
# 2) convolution
image_convolved = cv2.filter2D(image_gray, -1, kernel)

class Methods(Enum):
    Gradient = 1
    Map = 2

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
method = Methods.Gradient

if method == Methods.Gradient:
    [theta_data, M] = gradient_map(image_gray)
else:
    [theta_data, M] = orientation_map(image_gray, 7)

theta_data += np.pi / 2
theta_data /= np.pi
theta_uint8 = theta_data * 255
theta_uint8 = np.uint8(theta_uint8)
theta_uint8 = cv2.applyColorMap(theta_uint8, cv2.COLORMAP_JET)
theta_view = np.zeros(theta_uint8.shape)
theta_view = np.uint8(theta_view)
theta_view[M > 0.3] = theta_uint8[M > 0.3]

# theta view es interesante
cv2.imshow("Image", theta_view)
cv2.waitKey(0)