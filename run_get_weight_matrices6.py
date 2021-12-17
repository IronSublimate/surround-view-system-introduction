import os
import numpy as np
import cv2
from PIL import Image
from surround_view import FisheyeCameraModel6, display_image, BirdView6
import surround_view.param_settings6 as settings


def main():
    names = settings.camera_names
    images = [os.path.join(os.getcwd(), "images", name + ".png") for name in names]
    yamls = [os.path.join(os.getcwd(), "yaml", name + ".yaml") for name in names]
    camera_models = [FisheyeCameraModel6(camera_file, camera_name) for camera_file, camera_name in zip(yamls, names)]

    resize = settings.ratio
    projected = []
    for image_file, camera in zip(images, camera_models):
        img = cv2.imread(image_file)
        img[:img.shape[0] // 2, :, :] = 0
        img = camera.undistort(img)
        img = camera.project(img)
        # img = camera.flip(img)
        img = cv2.resize(img, (-1, -1), fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
        projected.append(img)

    birdview = BirdView6()
    Gmat, Mmat = birdview.get_weights_and_masks(projected)
    birdview\
        .update_frames(projected)\
        .make_luminance_balance()\
        .stitch_all_parts()\
        .make_white_balance()
    # birdview.copy_car_image()
    ret = display_image("BirdView Result", birdview.image)
    if ret > 0:
        fg = cv2.FileStorage(os.path.join(os.getcwd(), "yaml/projection.yml"), cv2.FileStorage_WRITE)
        fg.write("size", settings.total)
        fg.write("Gmat", Gmat)
        fg.write("Mmat", Mmat)
        fg.release()
        # np.save("Gmat.npy", Gmat)
        # np.save("Mmat.npy", Mmat)

        # Image.fromarray((Gmat * 255).astype(np.uint8)).save("weights.png")
        # Image.fromarray(Mmat.astype(np.uint8)).save("masks.png")


if __name__ == "__main__":
    main()
