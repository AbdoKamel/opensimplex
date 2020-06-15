import time
from PIL import Image  # Depends on the Pillow lib
import numpy as np

from opensimplex import OpenSimplex

WIDTH = 1024
HEIGHT = 1024
FEATURE_SIZE = 24.0


def main():
    simplex = OpenSimplex()

    print('Generating 2D image without arrays...')
    im = Image.new('L', (WIDTH, HEIGHT))
    t0 = time.time()
    for y in range(0, HEIGHT):
        for x in range(0, WIDTH):
            value = simplex.noise2d(x / FEATURE_SIZE, y / FEATURE_SIZE)
            color = int((value + 1) * 128)
            im.putpixel((x, y), color)
    t1 = time.time() - t0
    print("Time = {:.2f} sec.".format(t1))
    im.save('noise2d_no_arrays.png')

    print('Generating 2D image using arrays...')
    im_a = Image.new('L', (WIDTH, HEIGHT))
    t0 = time.time()
    y_2d, x_2d = np.mgrid[0:HEIGHT, 0:WIDTH]
    y_lin = y_2d.flatten()
    x_lin = x_2d.flatten()
    values = simplex.noise2d_array(x_lin / FEATURE_SIZE, y_lin / FEATURE_SIZE)
    values = ((values + 1) * 128).astype(np.uint8)
    values = np.reshape(values, (HEIGHT * WIDTH,))
    im_a.putdata(values)
    t2 = time.time() - t0
    print("Time = {:.2f} sec.".format(t2))
    im_a.save('noise2d_using_arrays.png')

    # check if both 2D images match
    err = np.mean(np.abs(np.array(im) - np.array(im_a)))
    print("Error between both 2D images = {}".format(err))


if __name__ == '__main__':
    main()
