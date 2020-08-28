import cv2
import numpy as np
from numpy import random

def xywh2xyxy(box_xywh):
    box_xyxy = box_xywh.copy()
    box_xyxy[..., 0] = box_xywh[..., 0] - box_xywh[..., 2] / 2.
    box_xyxy[..., 1] = box_xywh[..., 1] - box_xywh[..., 3] / 2.
    box_xyxy[..., 2] = box_xywh[..., 0] + box_xywh[..., 2] / 2.
    box_xyxy[..., 3] = box_xywh[..., 1] + box_xywh[..., 3] / 2.
    return box_xyxy


def xyxy2xywh(box_xyxy):
    box_xywh = box_xyxy.copy()
    box_xywh[..., 0] = (box_xyxy[..., 0] + box_xyxy[..., 2]) / 2.
    box_xywh[..., 1] = (box_xyxy[..., 1] + box_xyxy[..., 3]) / 2.
    box_xywh[..., 2] = box_xyxy[..., 2] - box_xyxy[..., 0]
    box_xywh[..., 3] = box_xyxy[..., 3] - box_xyxy[..., 1]
    return box_xywh

def PhotometricNoise(img_bgr,
                     h_delta=18.,
                     s_gain=0.5,
                     v_delta=32):
    if random.randint(2):
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # h[0, 359], s[0, 1.0], v[0, 255.]
        h_delta = np.random.uniform(-h_delta, h_delta)
        s_gain = np.random.uniform(1. - s_gain, 1. + s_gain)
        v_delta = np.random.uniform(-v_delta, v_delta)

        img_hsv[..., 0] = np.clip(img_hsv[..., 0] + h_delta, 0., 359.)
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] * s_gain, 0., 1.0)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + v_delta, 0., 255.0)

        img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        return img_bgr
    return img_bgr

def HorFlip(img, bboxes_xywh):
    if random.randint(2):
        img = cv2.flip(img, 1)
        bboxes_xywh[:, 0] = 1. - bboxes_xywh[:, 0]
        return img, bboxes_xywh
    return img, bboxes_xywh


def RandomTranslation(img, bboxes_xyxy, classes, delta = 100, max_iteration = 10):
    if random.randint(2):

        height, width = img.shape[0:2]

        img_org = img.copy()
        bboxes_xyxy_org = bboxes_xyxy.copy()

        for _ in range(max_iteration):
            img = img_org.copy()
            bboxes_xyxy = bboxes_xyxy_org.copy()

            tx = np.random.uniform(-delta, delta)
            ty = np.random.uniform(-delta, delta)

            #translation matrix
            tm = np.float32([[1, 0, tx],
                             [0, 1, ty]])  # [1, 0, tx], [1, 0, ty]

            img = cv2.warpAffine(img, tm, (width, height), borderValue=(127, 127, 127))

            tx /= width
            ty /= height

            bboxes_xyxy[:, [0, 2]] += tx
            bboxes_xyxy[:, [1, 3]] += ty

            clipped_bboxes_xyxy = np.clip(bboxes_xyxy, 0., 1.)

            clipped_bboxes_w = clipped_bboxes_xyxy[:, 2] - clipped_bboxes_xyxy[:, 0]
            clipped_bboxes_h = clipped_bboxes_xyxy[:, 3] - clipped_bboxes_xyxy[:, 1]

            valid_bboxes_inds = (clipped_bboxes_w > 0.01) & (clipped_bboxes_h > 0.01)
            if np.sum(valid_bboxes_inds) == 0:
                continue

            clipped_bboxes_xyxy = clipped_bboxes_xyxy[valid_bboxes_inds]
            clipped_bboxes_w = clipped_bboxes_w[valid_bboxes_inds]
            clipped_bboxes_h = clipped_bboxes_h[valid_bboxes_inds]

            bboxes_xyxy = bboxes_xyxy[valid_bboxes_inds]

            bboxes_w = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]
            bboxes_h = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]

            occlusion_proportion_w = 1. - (clipped_bboxes_w / bboxes_w)
            occlusion_proportion_h = 1. - (clipped_bboxes_h / bboxes_h)

            if np.sum(occlusion_proportion_w > 0.3) > 0 or np.sum(occlusion_proportion_h > 0.3) > 0:
                continue
            else:
                classes = classes[valid_bboxes_inds]
                return img, clipped_bboxes_xyxy, classes
        return img_org, bboxes_xyxy_org, classes

    return img, bboxes_xyxy, classes

def RandomScale(img, bboxes_xyxy, classes, scale=[-0.25, 0.25]):
    if random.randint(2):

        height, width = img.shape[0:2]
        max_iteration = 50

        img_org = img.copy()
        bboxes_xyxy_org = bboxes_xyxy.copy()

        n_bboxes = len(bboxes_xyxy_org)

        for _ in range(max_iteration):
            img = img_org.copy()
            bboxes_xyxy = bboxes_xyxy_org.copy()
            random_scale = np.random.uniform(1. + scale[0], 1. + scale[1])

            sm = cv2.getRotationMatrix2D(angle=0., center=(width / 2, height / 2), scale=random_scale)
            img = cv2.warpAffine(img, sm, (width, height), borderValue=(127, 127, 127))

            sm[0, 2] /= width
            sm[1, 2] /= height

            h_bboxes_xy_tl = np.concatenate([bboxes_xyxy[:, [0, 1]], np.ones((n_bboxes, 1))], axis=-1)
            h_bboxes_xy_br = np.concatenate([bboxes_xyxy[:, [2, 3]], np.ones((n_bboxes, 1))], axis=-1)

            h_bboxes_xy_tl = sm @ h_bboxes_xy_tl.T
            h_bboxes_xy_br = sm @ h_bboxes_xy_br.T

            bboxes_xyxy[:, [0, 1]] = h_bboxes_xy_tl.T
            bboxes_xyxy[:, [2, 3]] = h_bboxes_xy_br.T

            clipped_bboxes_xyxy = np.clip(bboxes_xyxy, 0., 1.)

            clipped_bboxes_w = clipped_bboxes_xyxy[:, 2] - clipped_bboxes_xyxy[:, 0]
            clipped_bboxes_h = clipped_bboxes_xyxy[:, 3] - clipped_bboxes_xyxy[:, 1]

            valid_bboxes_inds = (clipped_bboxes_w > 0.01) & (clipped_bboxes_h > 0.01)
            if np.sum(valid_bboxes_inds) == 0:
                continue

            clipped_bboxes_xyxy = clipped_bboxes_xyxy[valid_bboxes_inds]
            clipped_bboxes_w = clipped_bboxes_w[valid_bboxes_inds]
            clipped_bboxes_h = clipped_bboxes_h[valid_bboxes_inds]

            bboxes_xyxy = bboxes_xyxy[valid_bboxes_inds]

            bboxes_w = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]
            bboxes_h = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]

            occlusion_proportion_w = 1. - (clipped_bboxes_w / bboxes_w)
            occlusion_proportion_h = 1. - (clipped_bboxes_h / bboxes_h)

            if np.sum(occlusion_proportion_w > 0.3) > 0 or np.sum(occlusion_proportion_h > 0.3) > 0:
                continue
            else:
                classes = classes[valid_bboxes_inds]
                return img, clipped_bboxes_xyxy, classes
        return img_org, bboxes_xyxy_org, classes
    return img, bboxes_xyxy, classes



def RandomRotation(img, bboxes_xyxy, angle):
    height, width = img.shape[:2]

    angle = np.random.uniform(angle[0], angle[1], 15)
    angle = int(angle[np.random.random_integers(0, 7, 1)])

    if angle == 0.:
        img = img.copy()
        return img, bboxes_xyxy
    else:
        center = int(width / 2), int(height / 2)
        scale = 1.
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        img = cv2.warpAffine(img, matrix, (width, height) ,borderMode=cv2.cv2.BORDER_REFLECT_101)
        #img = cv2.warpAffine(img, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

        for coords in bboxes_xyxy:
            x_center_bbox = coords[0]
            y_center_bbox = coords[1]
            width_bbox = coords[2]
            height_bbox = coords[3]
            x_left = int((x_center_bbox - width_bbox / 2.) * width)
            x_right = int((x_center_bbox + width_bbox / 2.) * width)
            y_top = int((y_center_bbox - height_bbox / 2.) * height)
            y_bottom = int((y_center_bbox + height_bbox / 2.) * height)

            points = np.array([[x_left, y_top, 1.],
                                [x_left, y_bottom, 1.],
                                [x_right, y_top, 1.],
                                [x_right, y_bottom, 1.]])

            points = np.dot(matrix, points.T).T
            x_left = int(min(p[0] for p in points))
            x_right = int(max(p[0] for p in points))
            y_top = int(min(p[1] for p in points))
            y_bottom = int(max(p[1] for p in points))
            x_left, x_right = np.clip([x_left, x_right], 0, width)
            y_top, y_bottom = np.clip([y_top, y_bottom], 0, height)

            return cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), (255, 255, 255), 2)


def drawBBox(img, bboxes_xyxy):
    h, w = img.shape[:2]

    print(bboxes_xyxy)

    bboxes_xyxy[:, [0, 2]] *= w
    bboxes_xyxy[:, [1, 3]] *= h

    for bbox_xyxy in bboxes_xyxy:
        #print((int(bbox_xyxy[0]), int(bbox_xyxy[1])), (int(bbox_xyxy[2]), int(bbox_xyxy[3])))
        img = cv2.rectangle(img, (int(bbox_xyxy[0]), int(bbox_xyxy[1])), (int(bbox_xyxy[2]), int(bbox_xyxy[3])), (0, 255, 0), 10)

    return img

def get_annotations(line, height_image, width_image):
    annotation = line.split(" ")
    category = float(annotation[0])
    x_center_bbox = float(annotation[1])
    y_center_bbox = float(annotation[2])
    width_bbox = float(annotation[3])
    height_bbox = float(annotation[4])
    #x_left = int((x_center_bbox - width_bbox / 2.) * width_image)
    #x_right = int((x_center_bbox + width_bbox / 2.) * width_image)
    #y_top = int((y_center_bbox - height_bbox / 2.) * height_image)
    #y_bottom = int((y_center_bbox + height_bbox / 2.) * height_image)
    return [category, x_center_bbox, y_center_bbox, width_bbox, height_bbox]




if __name__ == '__main__':

    while(True):
        img = cv2.imread("282_120.jpg", cv2.IMREAD_COLOR).astype(np.float32)
        #img = cv2.resize(img, (416, 416)).astype(np.float32)

        height, width = img.shape[:2]
        #import dataset
        label = list()

        with open("282_120.txt", "r") as f:
            for line in f.readlines():
                line = line.strip()

                label.append(get_annotations(line, height, width))

        label = np.asarray(label)
        #print(label)
        #label = dataset.read_annotation_file("000017.txt")
        classes, bboxes_xywh = label[:, 0:1], label[:, 1:]

        #img = PhotometricNoise(img)
        #img, bboxes_xywh = HorFlip(img, bboxes_xywh)

        #print(label)
        #print(bboxes_xywh)
        bboxes_xyxy = xywh2xyxy(bboxes_xywh)
        #print(bboxes_xyxy)
        #img, bboxes_xyxy, classes = RandomTranslation(img, bboxes_xyxy, classes, delta= 100, max_iteration=10)
        #img, bboxes_xyxy, classes = RandomScale(img, bboxes_xyxy, classes)

        img = RandomRotation(img, bboxes_xywh, [-45, 45])

        #if len(bboxes_xyxy) != len(classes):
            #print("bbox랑 class 수랑 일치하지 않아~")

        #img = drawBBox(img, bboxes_xyxy)
        img /= 255.
        cv2.imshow("img", img)
        ch = cv2.waitKey(0)

        if ch == 27:
            break
