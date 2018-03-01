from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def sobel(ImageArray, Threshold):
    # Read a grayscale image file
    
    height, width = ImageArray.shape

    # Sobel operator gradients
    filterheight = 3
    filterwidth = 3
    filterhalfheight = (filterheight - 1) // 2
    filterhalfwidth = (filterwidth - 1) // 2

    h_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 4
    g_x = np.copy(ImageArray)
    for i in range(filterhalfheight, height - filterhalfheight):
        for j in range(filterhalfwidth, width - filterhalfwidth):
            pixelval = 0.0
            for k in range(i-filterhalfheight, i+filterhalfheight+1):
                for l in range(j-filterhalfwidth, j+filterhalfwidth+1):
                    pixelval += ImageArray[k, l] * h_x[i-k+filterhalfheight, j-l+filterhalfwidth]
            if pixelval < 0.0:
                pixelval = 0.0
            elif pixelval > 255.0:
                pixelval = 255.0
            g_x[i, j] = pixelval

    h_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4
    g_y = np.copy(ImageArray)
    for i in range(filterhalfheight, height - filterhalfheight):
        for j in range(filterhalfwidth, width - filterhalfwidth):
            pixelval = 0.0
            for k in range(i-filterhalfheight, i+filterhalfheight+1):
                for l in range(j-filterhalfwidth, j+filterhalfwidth+1):
                    pixelval += ImageArray[k, l] * h_y[i-k+filterhalfheight, j-l+filterhalfwidth]
            if pixelval < 0.0:
                pixelval = 0.0
            elif pixelval > 255.0:
                pixelval = 255.0
            g_y[i, j] = pixelval

    g_m = np.sqrt(np.add(np.square(g_x), np.square(g_y)))

    # Find Threshold% of pixels to be the threshold
    Sort_g_m = np.sort(g_m, axis=None)

    T_5 = Sort_g_m[int(Threshold*height*width)]
    T_5_g_m = np.copy(g_m)
    for i in range(height):
        for j in range(width):
            if T_5_g_m[i, j] < T_5:
                T_5_g_m[i, j] = 0
            else:
                T_5_g_m[i, j] = 255

    return T_5_g_m

if __name__ == "__main__":
    
    # Input: Image File
    # Input Threshold: 0.95 as 5%, 0.8 as 20%, etc
    ImageFile = "lena.bmp"
    Threshold = 0.5
    
    ImageArray = np.array(Image.open(ImageFile).convert('L'), 'f')
    Gradient = SobelOperator(ImageArray, Threshold)