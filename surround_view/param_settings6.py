import os
import cv2

camera_names = ["1", "2", "3", "4", "5", "6"]

# Units cm
import numpy as np

total = np.array((2000, 2000), dtype=np.int32)

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
theta = 3 * np.pi / 180 # 3
rotate_mat = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
project_keypoints = {
    "1": [(np.array((-287, +520 + 000)) + total / 2) @ rotate_mat,
          (np.array((+307, +520 + 000)) + total / 2) @ rotate_mat,
          (np.array((-287, +270 + 000)) + total / 2) @ rotate_mat,
          (np.array((+307, +270 + 000)) + total / 2) @ rotate_mat, ],
    "2": [(np.array((+307, +520 + 000)) + total / 2) @ rotate_mat,
          (np.array((+725, +520 + 000)) + total / 2) @ rotate_mat,
          (np.array((+307, +270 + 000)) + total / 2) @ rotate_mat,
          (np.array((+725, +270 + 000)) + total / 2) @ rotate_mat, ],
    "3": [(np.array((+307, -480 + 000)) + total / 2) @ rotate_mat,
          (np.array((+725, -480 + 000)) + total / 2) @ rotate_mat,
          (np.array((+307, -730 + 000)) + total / 2) @ rotate_mat,
          (np.array((+725, -730 + 000)) + total / 2) @ rotate_mat, ],
    "4": [(np.array((-287, -480 + 000)) + total / 2) @ rotate_mat,
          (np.array((+307, -480 + 000)) + total / 2) @ rotate_mat,
          (np.array((-287, -730 + 000)) + total / 2) @ rotate_mat,
          (np.array((+307, -730 + 000)) + total / 2) @ rotate_mat, ],
    "5": [(np.array((-825, -480 + 250)) + total / 2) @ rotate_mat,
          (np.array((-287, -480 + 250)) + total / 2) @ rotate_mat,
          (np.array((-825, -730 + 250)) + total / 2) @ rotate_mat,
          (np.array((-287, -730 + 250)) + total / 2) @ rotate_mat, ],
    "6": [(np.array((-825, +520 + 000)) + total / 2) @ rotate_mat,
          (np.array((-287, +520 + 000)) + total / 2) @ rotate_mat,
          (np.array((-825, +270 + 000)) + total / 2) @ rotate_mat,
          (np.array((-287, +270 + 000)) + total / 2) @ rotate_mat, ],
}
ratio = 0.5

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
