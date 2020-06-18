import tensorflow as tf
import cv2
import os
import time
import math
import numpy as np
from common import CustomResize
from scipy import interpolate
from tensorflow.python.framework import graph_util
from after_proc import img_proc

class_names = ['table']

DETECTRON_PALETTE = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.167, 0.000, 0.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3) * 255

PALETTE_RGB = DETECTRON_PALETTE


def _scale_box(box, scale):
    w_half = (box[2] - box[0]) * 0.5
    h_half = (box[3] - box[1]) * 0.5
    x_c = (box[2] + box[0]) * 0.5
    y_c = (box[3] + box[1]) * 0.5

    w_half *= scale
    h_half *= scale

    scaled_box = np.zeros_like(box)
    scaled_box[0] = x_c - w_half
    scaled_box[2] = x_c + w_half
    scaled_box[1] = y_c - h_half
    scaled_box[3] = y_c + h_half
    return scaled_box


def _paste_mask(box, mask, shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    Returns:
        A uint8 binary image of hxw.
    """
    assert mask.shape[0] == mask.shape[1], mask.shape

    ACCURATE_PASTE = True

    if ACCURATE_PASTE:
        # This method is accurate but much slower.
        mask = np.pad(mask, [(1, 1), (1, 1)], mode='constant')
        box = _scale_box(box, float(mask.shape[0]) / (mask.shape[0] - 2))

        mask_pixels = np.arange(0.0, mask.shape[0]) + 0.5
        mask_continuous = interpolate.interp2d(mask_pixels, mask_pixels, mask, fill_value=0.0)
        h, w = shape
        ys = np.arange(0.0, h) + 0.5
        xs = np.arange(0.0, w) + 0.5
        ys = (ys - box[1]) / (box[3] - box[1]) * mask.shape[0]
        xs = (xs - box[0]) / (box[2] - box[0]) * mask.shape[1]
        # Waste a lot of compute since most indices are out-of-border
        res = mask_continuous(xs, ys)
        return (res >= 0.5).astype('uint8')
    else:
        # This method (inspired by Detectron) is less accurate but fast.

        # int() is floor
        # box fpcoor=0.0 -> intcoor=0.0
        x0, y0 = list(map(int, box[:2] + 0.5))
        # box fpcoor=h -> intcoor=h-1, inclusive
        x1, y1 = list(map(int, box[2:] - 0.5))    # inclusive
        x1 = max(x0, x1)    # require at least 1x1
        y1 = max(y0, y1)

        w = x1 + 1 - x0
        h = y1 + 1 - y0

        # rounding errors could happen here, because masks were not originally computed for this shape.
        # but it's hard to do better, because the network does not know the "original" scale
        mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
        ret = np.zeros(shape, dtype='uint8')
        ret[y0:y1 + 1, x0:x1 + 1] = mask
        return ret


def clip_boxes(boxes, shape):
    """
    Args:
        boxes: (...)x4, float
        shape: h, w
    """
    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)


def area(boxes):
  """Computes area of boxes.

  Args:
    boxes: Numpy array with shape [N, 4] holding N boxes

  Returns:
    a numpy array with shape [N*1] representing box areas
  """
  return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def draw_text(img, pos, text, color, font_scale=0.4):
    """
    Draw text on an image.

    Args:
        pos (tuple): x, y; the position of the text
        text (str):
        font_scale (float):
        color (tuple): a 3-tuple BGR color in [0, 255]
    """
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, 1)
    # Place text background.
    if x0 + text_w > img.shape[1]:
        x0 = img.shape[1] - text_w
    if y0 - int(1.15 * text_h) < 0:
        y0 = int(1.15 * text_h)
    back_topleft = x0, y0 - int(1.3 * text_h)
    back_bottomright = x0 + text_w, y0
    cv2.rectangle(img, back_topleft, back_bottomright, color, -1)
    # Show text.
    text_bottomleft = x0, y0 - int(0.25 * text_h)
    cv2.putText(img, text, text_bottomleft, font, font_scale, (222, 222, 222), lineType=cv2.LINE_AA)
    return img


def draw_boxes(im, boxes, labels, results, color=None):
    """
    Args:
        im (np.ndarray): a BGR image in range [0,255]. It will not be modified.
        boxes (np.ndarray): a numpy array of shape Nx4 where each row is [x1, y1, x2, y2].
        labels: (list[str] or None)
        color: a 3-tuple BGR color (in range [0, 255])

    Returns:
        np.ndarray: a new image.
    """
    boxes = np.asarray(boxes, dtype='int32')
    if labels is not None:
        assert len(labels) == len(boxes), "{} != {}".format(len(labels), len(boxes))
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    sorted_inds = np.argsort(-areas)    # draw large ones first
    assert areas.min() > 0, areas.min()
    # allow equal, because we are not very strict about rounding error here
    assert boxes[:, 0].min() >= 0 and boxes[:, 1].min() >= 0 \
        and boxes[:, 2].max() <= im.shape[1] and boxes[:, 3].max() <= im.shape[0], \
        "Image shape: {}\n Boxes:\n{}".format(str(im.shape), str(boxes))

    im = im.copy()
    if color is None:
        color = (15, 128, 15)
    if im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 1):
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for i in sorted_inds:
        if results[1][i] >= 0.95:
            box = boxes[i, :]
            if labels is not None:
                im = draw_text(im, (box[0], box[1]), labels[i], color=color)
            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]),
                          color=color, thickness=6)
    return im


def draw_mask(im, mask, alpha=0.5, color=None):
    """
    Overlay a mask on top of the image.

    Args:
        im: a 3-channel uint8 image in BGR
        mask: a binary 1-channel image of the same size
        color: if None, will choose automatically
    """
    if color is None:
        color = PALETTE_RGB[np.random.choice(len(PALETTE_RGB))][::-1]
    color = np.asarray(color, dtype=np.float32)
    im = np.where(np.repeat((mask > 0)[:, :, None], 3, axis=2),
                  im * (1 - alpha) + color * alpha, im)
    im = im.astype('uint8')
    return im


def draw_final_outputs(img, results):
    """
    Args:
        results: [DetectionResult]
    """
    if len(results) == 0:
        return img

    # Display in largest to smallest order to reduce occlusion
    boxes = np.asarray([r for r in results[0]])
    ret = img
    tags = []

    # for t in range(len(results[3])):
    #    if results[1][t] > 0.95:
    #        ret = draw_mask(ret, results[3][t])

    for i, r in enumerate(results[2]):
        tags.append("{},{:.2f}".format(class_names[r-1], results[1][i]))
    ret = draw_boxes(ret, boxes, tags, results)
    return ret


def save_crop(cropimg, box, file):
    img = cropimg.copy()
    drawimg = cv2.polylines(img, np.int32([box]), True, (0, 0, 255), 7)
    save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_perfect',
                             file + "_0" + '.jpg')
    cv2.imwrite(save_path, drawimg)

    orignal_W1 = math.ceil(np.sqrt((box[3][1] - box[2][1]) ** 2 + (box[3][0] - box[2][0]) ** 2))
    orignal_W2 = math.ceil(np.sqrt((box[1][1] - box[0][1]) ** 2 + (box[1][0] - box[0][0]) ** 2))
    orignal_W = orignal_W1 if orignal_W1 > orignal_W2 else orignal_W2

    orignal_H1 = math.ceil(np.sqrt((box[3][1] - box[0][1]) ** 2 + (box[3][0] - box[0][0]) ** 2))
    orignal_H2 = math.ceil(np.sqrt((box[2][1] - box[1][1]) ** 2 + (box[2][0] - box[1][0]) ** 2))
    orignal_H = orignal_H1 if orignal_H1 > orignal_H2 else orignal_H2

    # 原图中的四个顶点,与变换矩阵
    pts1 = np.float32([box[2], box[3], box[0], box[1]])
    pts2 = np.float32(
        [[int(orignal_W + 1), int(orignal_H + 1)], [0, int(orignal_H + 1)], [0, 0], [int(orignal_W + 1), 0]])

    # 生成透视变换矩阵；进行透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)

    result_img = cv2.warpPerspective(cropimg, M, (int(orignal_W), int(orignal_H)))
    save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_perfect',
                             file + "_1" + '.jpg')
    cv2.imwrite(save_path, result_img)



def cv_display(image, origimg, result, file):
    img = image.copy()
    masks = result[3]
    boxes = result[0]
    boxes[:, 0] = boxes[:, 0] - 100
    boxes[:, 1] = boxes[:, 1] - 100
    boxes[:, 2] = boxes[:, 2] + 100
    boxes[:, 3] = boxes[:, 3] + 100
    boxes = clip_boxes(boxes, image.shape[:2])
    for i, mask in enumerate(masks):
        if result[1][i] > 0.97:
            crop_img = origimg[int(boxes[i][1]):int(boxes[i][3]), int(boxes[i][0]):int(boxes[i][2]), :]
            thresh = np.where(mask > 0, 255, 0).astype('uint8')
            # ans_points = img_proc(img, thresh, file, i)
            ans_points = img_proc(thresh)
            crop_points = ans_points.copy()
            crop_points[:, 0] = ans_points[:, 0] - boxes[i][0]
            crop_points[:, 1] = ans_points[:, 1] - boxes[i][1]
            save_crop(crop_img, crop_points, file)
            img = cv2.polylines(img, np.int32([ans_points]), True, (0, 0, 255), 7)
    return img


def proc_opencv(origimg, result, file):
    boxes = result[0]
    boxes[:, 0] = boxes[:, 0] - 100
    boxes[:, 1] = boxes[:, 1] - 100
    boxes[:, 2] = boxes[:, 2] + 100
    boxes[:, 3] = boxes[:, 3] + 100
    boxes = clip_boxes(boxes, origimg.shape[:2])
    for i, box in enumerate(boxes):
        if result[1][i] > 0.975:
            crop_img = origimg[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            # gray = cv2.blur(gray, (3, 3))
            # kernel = np.ones((3, 3), np.uint8)
            # erosion = cv2.erode(gray, kernel, iterations=3)
            # threshold = 35
            # thresh = cv2.Canny(erosion, threshold, threshold * 3, apertureSize=3)
            # thresh = cv2.dilate(edge, kernel, iterations=3)
            thresh = cv2.adaptiveThreshold(~gray, 255,  # ~取反，很重要，使二值化后的图片是黑底白字
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
            save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_photo', 'thresh_'+str(i)+'.jpg')
            cv2.imwrite(save_path, thresh)

            # ans_points = img_proc(thresh)
            # crop_img = cv2.polylines(crop_img, np.int32([ans_points]), True, (0, 0, 255), 7)
            # lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 100, minLineLength=200, maxLineGap=100)
            # for line in lines:
            '''
                rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
                a = np.cos(theta)  # theta是弧度
                b = np.sin(theta)
                x0 = a * rho  # 代表x = r * cos（theta）
                y0 = b * rho  # 代表y = r * sin（theta）
                x1 = int(x0 + 1000 * (-b))  # 计算直线起点横坐标
                y1 = int(y0 + 1000 * a)  # 计算起始起点纵坐标
                x2 = int(x0 - 1000 * (-b))  # 计算直线终点横坐标
                y2 = int(y0 - 1000 * a)  # 计算直线终点纵坐标
            '''
                #x1, y1, x2, y2 = line[0]
                # 注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
                #crop_img = cv2.line(crop_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 点的坐标必须是元组，不能是列表

            # 识别横线
            scale = 40
            erode_iters = 1
            dilate_iters = 2
            rows, cols = thresh.shape
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
            # (cols // scale, 1) 为了获取横向的表格线，设置腐蚀和膨胀的操作区域为一个比较大的横向直条
            eroded = cv2.erode(thresh, kernel, iterations=erode_iters)
            dilatedcol = cv2.dilate(eroded, kernel, iterations=dilate_iters)

            lines = cv2.HoughLinesP(dilatedcol, 1, np.pi / 180, 100, minLineLength=cols//2, maxLineGap=200)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                crop_img = cv2.line(crop_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 点的坐标必须是元组，不能是列表

            save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_perfect',
                                     file + "_" + str(i) + '.jpg')
            cv2.imwrite(save_path, crop_img)


def crop_box(origimg, result, file):
    boxes = result[0]
    h, w = origimg.shape[:2]
    # scale = h // 20 if h < w else w // 20
    for i, box in enumerate(boxes):
        if result[1][i] > 0.975:
            # ly = (int(box[3]) - int(box[1])) // 20
            # lx = (int(box[2]) - int(box[0])) // 20
            ly = 100
            lx = 100
            upy = int(box[1]) - ly if int(box[1]) - ly > 200 else 200
            dwy = int(box[3]) + ly if int(box[3]) + ly < h - 200 else h - 200
            lfx = int(box[0]) - lx if int(box[0]) - lx > 200 else 200
            rtx = int(box[2]) + lx if int(box[2]) + lx < w - 200 else w - 200
            # crop_img = origimg[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
            crop_img = origimg[upy:dwy, lfx:rtx, :]
            save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_photo',
                                     file[:-4] + "_" + str(i) + '.jpg')
            cv2.imwrite(save_path, crop_img)


def infer_with_pb_model(pb_path, image_rootPath):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        # 读取pb模型
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

            # 运行tensorflow 进行预测
            with tf.Session() as sess:
                # sess.run(tf.global_variables_initializer())
                # output_tensors = get_tensors_by_names(['output/boxes', 'output/scores', 'output/labels'])
                # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
                names = ['output/boxes:0', 'output/scores:0', 'output/labels:0']# , 'output/masks:0']
                output_tensors = []
                for n in names:
                    output_tensors.append(tf.get_default_graph().get_tensor_by_name(n))

                created_img = np.zeros((400, 400, 3), np.uint8)
                sess.run(output_tensors, feed_dict={"image:0": created_img})

                sum = 0
                count = 0

                file = 'img_300403.jpg'
                # for file in os.listdir(image_rootPath):
                #     if file[-4:] == 'json':
                #         continue
                if file == 'img_300403.jpg':
                    print(file)

                    image_path = os.path.join(image_rootPath, file)

                    # 读取预测图片
                    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

                    WHITE = [255, 255, 255]
                    img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=WHITE)

                    print('origin shape: ', img.shape)

                    # resizer = CustomResize(800, 1333)
                    # resized_img = resizer.augment(img)

                    h, w = img.shape[:2]
                    size = 800.0
                    scale = size / min(h, w)
                    if h < w:
                        newh, neww = size, scale * w
                    else:
                        newh, neww = scale * h, size
                    if max(newh, neww) > 1333:
                        scale = 1333 * 1.0 / max(newh, neww)
                        newh = newh * scale
                        neww = neww * scale
                    neww = int(neww + 0.5)
                    newh = int(newh + 0.5)

                    resized_img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_LINEAR)

                    if img.ndim == 3 and resized_img.ndim == 2:
                        resized_img = resized_img[:, :, np.newaxis]

                    print('resized shape: ', resized_img.shape)

                    start = time.time()

                    result = sess.run(output_tensors, feed_dict={"image:0": resized_img})

                    print('box: ', np.array(result[0]).shape)
                    print('score: ', np.array(result[1]).shape)
                    print('lable: ', np.array(result[2]).shape)
                    # print('masks: ', np.array(result[3]).shape)

                    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
                    boxes = result[0] / scale
                    boxes = clip_boxes(boxes, img.shape[:2])
                    result[0] = boxes
                    '''
                    masks = result[3]
                    full_masks = [_paste_mask(box, mask, img.shape[:2])
                                  for box, mask in zip(boxes, masks)]
                    masks = full_masks
                    result[3] = masks
                    '''
                    result_img = draw_final_outputs(img, result)
                    # result_img = cv_display(result_img, img, result, file)
                    # proc_opencv(img.copy(), result, file)
                    crop_box(img.copy(), result, file)

                    end = time.time()
                    print('time consume: ', end - start)
                    sum += (end - start)
                    count += 1

                    save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_perfect', file)
                    cv2.imwrite(save_path, result_img)

                    print('\n')

                print('count: ', count)
                print('pre time {} /s: '.format(sum / count))


if __name__ == '__main__':
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    pb_path = '/home/ubuntu/cs/tensorpack/examples/FasterRCNN/train_log/fastrcnn_resume/frozen_model62.pb'
    image_rootPath = '/home/ubuntu/cs/table_test_data'
    infer_with_pb_model(pb_path, image_rootPath)

