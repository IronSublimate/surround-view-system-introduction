import os
import numpy as np
import cv2
from PIL import Image
from PyQt5.QtCore import QMutex, QWaitCondition, QMutexLocker
from .base_thread import BaseThread
from .imagebuffer import Buffer
from . import param_settings6 as settings

from . import utils
from typing import Tuple, List


class ProjectedImageBuffer(object):
    """
    Class for synchronizing processing threads from different cameras.
    """

    def __init__(self, drop_if_full=True, buffer_size=8):
        self.drop_if_full = drop_if_full
        self.buffer = Buffer(buffer_size)
        self.sync_devices = set()
        self.wc = QWaitCondition()
        self.mutex = QMutex()
        self.arrived = 0
        self.current_frames = dict()

    def bind_thread(self, thread):
        with QMutexLocker(self.mutex):
            self.sync_devices.add(thread.device_id)

        name = thread.camera_model.camera_name
        shape = settings.project_shapes[name]
        self.current_frames[thread.device_id] = np.zeros(shape[::-1] + (3,), np.uint8)
        thread.proc_buffer_manager = self

    def get(self):
        return self.buffer.get()

    def set_frame_for_device(self, device_id, frame):
        if device_id not in self.sync_devices:
            raise ValueError("Device not held by the buffer: {}".format(device_id))
        self.current_frames[device_id] = frame

    def sync(self, device_id):
        # only perform sync if enabled for specified device/stream
        self.mutex.lock()
        if device_id in self.sync_devices:
            # increment arrived count
            self.arrived += 1
            # we are the last to arrive: wake all waiting threads
            if self.arrived == len(self.sync_devices):
                self.buffer.add(self.current_frames, self.drop_if_full)
                self.wc.wakeAll()
            # still waiting for other streams to arrive: wait
            else:
                self.wc.wait(self.mutex)
            # decrement arrived count
            self.arrived -= 1
        self.mutex.unlock()

    def wake_all(self):
        with QMutexLocker(self.mutex):
            self.wc.wakeAll()

    def __contains__(self, device_id):
        return device_id in self.sync_devices

    def __str__(self):
        return (self.__class__.__name__ + ":\n" +
                "devices: {}\n".format(self.sync_devices))


# def FI(front_image):
#     return front_image[:, :xl]
#
#
# def FII(front_image):
#     return front_image[:, xr:]
#
#
# def FM(front_image):
#     return front_image[:, xl:xr]
#
#
# def BIII(back_image):
#     return back_image[:, :xl]
#
#
# def BIV(back_image):
#     return back_image[:, xr:]
#
#
# def BM(back_image):
#     return back_image[:, xl:xr]
#
#
# def LI(left_image):
#     return left_image[:yt, :]
#
#
# def LIII(left_image):
#     return left_image[yb:, :]
#
#
# def LM(left_image):
#     return left_image[yt:yb, :]
#
#
# def RII(right_image):
#     return right_image[:yt, :]
#
#
# def RIV(right_image):
#     return right_image[yb:, :]
#
#
# def RM(right_image):
#     return right_image[yt:yb, :]


class BirdView6(BaseThread):

    def __init__(self,
                 proc_buffer_manager=None,
                 drop_if_full=True,
                 buffer_size=8,
                 parent=None):
        super(BirdView6, self).__init__(parent)
        self.proc_buffer_manager = proc_buffer_manager
        self.drop_if_full = drop_if_full
        self.buffer = Buffer(buffer_size)
        self.image = np.zeros((int(settings.total[0] * settings.ratio), int(settings.total[1] * settings.ratio), 3),
                              np.uint32)
        self.weights = None
        self.masks = None
        self.car_image = settings.car_image
        self.frames = None

    def get(self):
        return self.buffer.get()

    def update_frames(self, images):
        self.frames: List[np.ndarray] = images
        return self

    def load_weights_and_masks(self, weights_image, masks_image):
        GMat = np.asarray(Image.open(weights_image).convert("RGBA"), dtype=np.float) / 255.0
        self.weights = [np.stack((GMat[:, :, k],
                                  GMat[:, :, k],
                                  GMat[:, :, k]), axis=2)
                        for k in range(4)]

        Mmat = np.asarray(Image.open(masks_image).convert("RGBA"), dtype=np.float)
        Mmat = utils.convert_binary_to_bool(Mmat)
        self.masks = [Mmat[:, :, k] for k in range(6)]

    def merge(self, imA, imB, k):
        G = self.weights[k]
        return (imA * G + imB * (1 - G)).astype(np.uint8)

    # @property
    # def FL(self):
    #     return self.image[:yt, :xl]
    #
    # @property
    # def F(self):
    #     return self.image[:yt, xl:xr]
    #
    # @property
    # def FR(self):
    #     return self.image[:yt, xr:]
    #
    # @property
    # def BL(self):
    #     return self.image[yb:, :xl]
    #
    # @property
    # def B(self):
    #     return self.image[yb:, xl:xr]
    #
    # @property
    # def BR(self):
    #     return self.image[yb:, xr:]
    #
    # @property
    # def L(self):
    #     return self.image[yt:yb, :xl]
    #
    # @property
    # def R(self):
    #     return self.image[yt:yb, xr:]
    #
    # @property
    # def C(self):
    #     return self.image[yt:yb, xl:xr]

    def stitch_all_parts(self):
        # front, back, left, right = self.frames
        # np.copyto(self.F, FM(front))
        # np.copyto(self.B, BM(back))
        # np.copyto(self.L, LM(left))
        # np.copyto(self.R, RM(right))
        # np.copyto(self.FL, self.merge(FI(front), LI(left), 0))
        # np.copyto(self.FR, self.merge(FII(front), RII(right), 1))
        # np.copyto(self.BL, self.merge(BIII(back), LIII(left), 2))
        # np.copyto(self.BR, self.merge(BIV(back), RIV(right), 3))
        # sz = len(self.frames)
        # for i in range(sz):
        #     j = (i + 1) % sz
        self.image = self.image.astype(np.float32)
        for img, mask in zip(self.frames, self.weights):
            self.image += (img * mask).astype(np.float32)
        self.image = self.image.astype(np.uint8)
        return self

    def copy_car_image(self):
        np.copyto(self.C, self.car_image)

    def make_luminance_balance(self):

        def tune(x):
            if x >= 1:
                return x * np.exp((1 - x) * 0.5)
            else:
                return x * np.exp((1 - x) * 0.8)

        sz = len(self.frames)
        Gs = [np.ndarray((0,))] * sz
        Ms = [np.ndarray((0,))] * sz
        bgr_split = [(np.ndarray((0,)),)] * sz
        for i in range(sz):
            bgr_split[i] = cv2.split(self.frames[i])

        luminance_ratio = np.zeros((sz, 3))  # 3 channel bgr
        for i in range(sz):
            j = (i + 1) % sz
            k = (i + sz - 1) % sz
            for m in range(3):
                luminance_ratio[i, m] = \
                    utils.mean_luminance_ratio(bgr_split[i][m], bgr_split[j][m], self.masks[i])

        t = [1.0, 1.0, 1.0]
        for m in range(3):
            for i in range(sz):
                t[m] *= luminance_ratio[i, m]
            t[m] = t[m] ** (1 / sz)

        for i in range(sz):
            j = (i + 1) % sz
            k = (i + sz - 1) % sz
            x = [
                tune(t[0] / (luminance_ratio[k, 0] / luminance_ratio[j, 0]) ** 0.5),
                tune(t[1] / (luminance_ratio[k, 1] / luminance_ratio[j, 1]) ** 0.5),
                tune(t[2] / (luminance_ratio[k, 2] / luminance_ratio[j, 2]) ** 0.5)
            ]
            img_bgr = [
                utils.adjust_luminance(bgr_split[i][0], x[0]),
                utils.adjust_luminance(bgr_split[i][1], x[1]),
                utils.adjust_luminance(bgr_split[i][2], x[2]),
            ]
            self.frames[i] = cv2.merge(img_bgr)

        return self

    def get_weights_and_masks(self, images: List[np.ndarray]):
        # front, back, left, right = images
        sz = len(images)
        Gs = [np.ndarray((0,))] * sz
        Ms = [np.ndarray((0,))] * sz

        for i in range(sz):
            j = (i + 1) % sz
            k = (i + sz - 1) % sz
            G1, M1 = utils.get_weight_mask_matrix(images[i], images[j])
            G2, M2 = utils.get_weight_mask_matrix(images[i], images[k])
            Gs[i] = G1 * G2
            Ms[i] = M1
            # Ms[i] = cv2.bitwise_and(M1, M2)

        self.weights = [np.stack((G, G, G), axis=2) for G in Gs]
        self.masks = [(M / 255.0).astype(np.int) for M in Ms]
        return np.stack(Gs, axis=2), np.stack(Ms, axis=2)

    def make_white_balance(self):
        self.image = utils.make_white_balance(self.image)

    def run(self):
        if self.proc_buffer_manager is None:
            raise ValueError("This thread requires a buffer of projected images to run")

        while True:
            self.stop_mutex.lock()
            if self.stopped:
                self.stopped = False
                self.stop_mutex.unlock()
                break
            self.stop_mutex.unlock()
            self.processing_time = self.clock.elapsed()
            self.clock.start()

            self.processing_mutex.lock()

            self.update_frames(self.proc_buffer_manager.get().values())
            self.make_luminance_balance().stitch_all_parts()
            self.make_white_balance()
            self.copy_car_image()
            self.buffer.add(self.image.copy(), self.drop_if_full)
            self.processing_mutex.unlock()

            # update statistics
            self.update_fps(self.processing_time)
            self.stat_data.frames_processed_count += 1
            # inform GUI of updated statistics
            self.update_statistics_gui.emit(self.stat_data)
