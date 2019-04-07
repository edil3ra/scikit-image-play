from pathlib import Path
import numpy as np
from skimage import data
import skimage as si
import matplotlib.pyplot as plt
import os
import utils as ut

source_path = Path('images/original')
gray_path = Path('images/gray')



def gray_image_transform(image, **kwargs):
    grayscale_image = si.color.rgb2gray(image)
    return grayscale_image

def remove_color_image_transform(image, red=False, green=False, blue=False):
    copy_image = image.copy()
    if red:
        copy_image[:, :, 0] = 0
    if green:
        copy_image[:, :, 1] = 0
    if blue:
        copy_image[:, :, 2] = 0
    return copy_image


def original_to_transform(source_path, destination_path, transform_f, prefix=None, **kwargs):
    original_names, original_paths = list(zip(*[(image.name, image) for image in source_path.iterdir()]))
    print(f'reading {len(original_names)} images')
    images = [si.io.imread(path) for path in original_paths]

    transformed_paths = [destination_path.joinpath(name) for name in original_names]
    # transformed_images = transform_f(images, **args)

    print(f'start transforming')
    for path, image in zip(transformed_paths, images):
        print('transforming image')
        transformed_image = transform_f(image, **kwargs)
        name = path.name if not prefix else f'{prefix}_{path.name}'
        print(f'writing image: {name} to: {path.absolute()}')
        si.io.imsave(path, transformed_image)


def show_source_from_dirname(source_path, column=5, cmap='gray'):
    names, paths = list(zip(*[(image.name, image) for image in source_path.iterdir()]))
    print(f'reading {len(names)} images')
    images = [si.io.imread(path) for path in paths]

    figure, axes = plt.subplots((len(images) // column) + 1, column)
    axes = axes.ravel()

    for ax in axes:
        ax.axis('off')

    for image, ax in zip(images, axes):
        ax.axis('off')
        ax.imshow(image, cmap=cmap)


def show_all(*images, **kwargs):
    images = [si.util.img_as_float(img) for img in images]

    titles = kwargs.pop('titles', [])
    if len(titles) != len(images):
        titles = list(titles) + [''] * (len(images) - len(titles))

    limits = kwargs.pop('limits', 'image')
    if limits == 'image':
        kwargs.setdefault('vmin', min(img.min() for img in images))
        kwargs.setdefault('vmax', max(img.max() for img in images))
    elif limits == 'dtype':
        vmin, vmax = dtype_limits(images[0])
        kwargs.setdefault('vmin', vmin)
        kwargs.setdefault('vmax', vmax)

    nrows, ncols = kwargs.get('shape', (1, len(images)))

    size = nrows * kwargs.pop('size', 5)
    width = size * len(images)
    if nrows > 1:
        width /= nrows * 1.33
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, size))
    for ax, img, label in zip(axes.ravel(), images, titles):
        ax.imshow(img, **kwargs)
        ax.set_title(label)


    

def show_image_channels(image):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    f, axes = plt.subplots(2, 4, figsize=(16, 5))

    for ax in axes[0]:
        ax.axis('off')

    (ax_r, ax_g, ax_b, ax_color) = axes[0]
    (axh_r, axh_g, axh_b, axh_color) = axes[1]

    ax_r.imshow(r, cmap='gray')
    axh_r.hist(r.flatten(), bins=255, color='red')
    axh_r.set_title('red channel')

    ax_g.imshow(g, cmap='gray')
    axh_g.hist(g.flatten(), bins=255, density=True, color='green')
    axh_g.set_title('green channel')

    ax_b.imshow(b, cmap='gray')
    axh_b.hist(b.flatten(), bins=255, density=True, color='blue')
    axh_b.set_title('blue channel')

    ax_color.imshow(np.stack([r, g, b], axis=2))
    axh_color.hist(np.average(np.stack([r, g, b], axis=2), axis=2).flatten(), bins=255, density=True)
    axh_color.set_title('all channels')





def main():
    ioriginal = si.io.imread_collection('images/original/*')
    me = si.img_as_float64(ioriginal[6])

    # original_to_transform(source_path, gray_path, gray_image_transform, 'GRAY')
    
    show_all(
        remove_color_image_transform(me, red=False, green=True, blue=True ),
        remove_color_image_transform(me, red=True, green=False, blue=True ),
        remove_color_image_transform(me, red=True, green=True, blue=False)
    )

    ut.imshow_with_histogram(me)
