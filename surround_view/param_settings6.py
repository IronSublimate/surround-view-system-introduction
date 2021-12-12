import os
import cv2

camera_names = ["1", "2", "3", "4", "5", "6"]

# Units cm
import numpy as np

total = np.array((20000, 20000), dtype=np.int)

# project_shapes = {
#     "front": (total_w, yt),
#     "back": (total_w, yt),
#     "left": (total_h, xl),
#     "right": (total_h, xl),
#     "1": (total_w, yt),
# }

# pixel locations of the four points to be chosen.
# you must click these pixels in the same order when running
# the get_projection_map.py script
project_keypoints = {
    "1": [np.array((-2870, +5200), dtype=np.int) + total // 2,
          np.array((+3070, +5200), dtype=np.int) + total // 2,
          np.array((-2870, +2700), dtype=np.int) + total // 2,
          np.array((+3070, +2700), dtype=np.int) + total // 2, ],

    "2": [np.array((+3070, +5200), dtype=np.int) + total // 2,
          np.array((+7250, +5200), dtype=np.int) + total // 2,
          np.array((+3070, +2700), dtype=np.int) + total // 2,
          np.array((+7250, +2700), dtype=np.int) + total // 2, ],

    "3": [np.array((+3070, -4800), dtype=np.int) + total // 2,
          np.array((+7250, -4800), dtype=np.int) + total // 2,
          np.array((+3070, -7300), dtype=np.int) + total // 2,
          np.array((+7250, -7300), dtype=np.int) + total // 2, ],

    "4": [np.array((-2870, -4800), dtype=np.int) + total // 2,
          np.array((+3070, -4800), dtype=np.int) + total // 2,
          np.array((-2870, -7300), dtype=np.int) + total // 2,
          np.array((+3070, -7300), dtype=np.int) + total // 2, ],

    "5": [np.array((-8250, -4800), dtype=np.int) + total // 2,
          np.array((-2870, -4800), dtype=np.int) + total // 2,
          np.array((-8250, -7300), dtype=np.int) + total // 2,
          np.array((-2870, -7300), dtype=np.int) + total // 2, ],

    "6": [np.array((-8250, +5200), dtype=np.int) + total // 2,
          np.array((-2870, +5200), dtype=np.int) + total // 2,
          np.array((-8250, +2700), dtype=np.int) + total // 2,
          np.array((-2870, +2700), dtype=np.int) + total // 2, ],
}
ratio = 0.05

car_image = cv2.imread(os.path.join(os.getcwd(), "images", "car.png"))
# car_image = cv2.resize(car_image, (xr - xl, yb - yt))

if __name__ == '__main__':
    # draw each car's projection use opencv
    img = np.zeros((total[1], total[0], 3), dtype=np.uint8)
    for car_name in camera_names:
        poly = project_keypoints[car_name]
        cv2.polylines(img,
                      [np.array(poly, dtype=np.int32)],
                      True,
                      (np.random.randint(256), np.random.randint(256), np.random.randint(256)),
                      thickness=200
                      )
    # cv2.polylines(img, list(project_keypoints.values()), True, (255, 255, 255), thickness=2)

    cv2.namedWindow("projection", cv2.WINDOW_NORMAL)
    cv2.imshow("projection", img)
    cv2.waitKey(0)
