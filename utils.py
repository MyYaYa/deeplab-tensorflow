import numpy as np

label_colours = [(0, 0, 0)
                 # 0=background
    , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
    , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]


# 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor


def decode_mask(masks):
    n, h, w, c = masks.shape
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        for j_, j in enumerate(masks[i, :, :, 0]):
            for k_, k in enumerate(j):
                outputs[i, j_, k_, :] = label_colours[masks[i, j_, k_, 0]]
    return outputs


def inverse_channel(images, image_mean):
    n, h, w, c = images.shape
    outputs = np.zeros((n, h, w, c), dtype=np.uint8)
    for i in range(n):
        outputs[i] = (images[i] + image_mean)[:, :, ::-1].astype(np.uint8)

    return outputs