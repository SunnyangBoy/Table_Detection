import cv2
import math
import numpy as np
import os

def GeneralEquation(first_x, first_y, second_x, second_y):
    # 一般式 Ax+By+C=0
    A = second_y-first_y
    B = first_x-second_x
    C = second_x * first_y - first_x * second_y
    return A, B, C


def GetIntersectPointofLines(x1, y1, x2, y2, x3, y3, x4, y4):
    A1, B1, C1 = GeneralEquation(x1, y1, x2, y2)

    A2, B2, C2 = GeneralEquation(x3, y3, x4, y4)

    m = A1 * B2 - A2 * B1

    if m == 0:
        print("直线平行，无交点")
        return None, None
    else:
        x = C2 / m * B1 - C1 / m * B2
        y = C1 / m * A2 - C2 / m * A1
        return x, y


def dist(key_point):
    left = key_point[0]
    right = key_point[1]
    return math.sqrt(math.pow((left[0] - right[0]), 2) + math.pow((left[1] - right[1]), 2))


def cpt_line(points):
    lines = []
    pt_num = len(points)
    for p in range(pt_num):
        q = (p + 1) % pt_num
        left = points[p]
        right = points[q]
        lines.append([left, right, p])
    return lines


def new_order(points):
    '''
    new_points = []
    avg_x = np.mean(points[:, 0])
    avg_y = np.mean(points[:, 1])
    for point in points:
        if point[0] < avg_x and point[1] < avg_y:
            new_points.append(point)
            break
    for point in points:
        if point[0] > avg_x and point[1] < avg_y:
            new_points.append(point)
            break
    for point in points:
        if point[0] > avg_x and point[1] > avg_y:
            new_points.append(point)
            break
    for point in points:
        if point[0] < avg_x and point[1] > avg_y:
            new_points.append(point)
            break
    return np.array(new_points)
    '''
    new_points = []
    avg_x = np.mean(points[:, 0])
    avg_y = np.mean(points[:, 1])
    lines = cpt_line(points)
    lines = sorted(lines, key=lambda k: dist(k), reverse=False)

    # 倾斜角度小于45度下成立，最短边在左右两边，使用竖直分割线
    if is_vertical(lines[0]):
        left_points = []
        right_points = []
        for point in points:
            if point[0] < avg_x:
                left_points.append(point)
            else:
                right_points.append(point)

        assert len(left_points) == 2, "顶点排序出错，请检查new_order函数"

        left_points = sorted(left_points, key=lambda k: k[1], reverse=False)
        right_points = sorted(right_points, key=lambda k: k[1], reverse=False)
        new_points.append(left_points[0])
        new_points.append(right_points[0])
        new_points.append(right_points[1])
        new_points.append(left_points[1])
    # 倾斜角度小于45度下成立，最短边在上下两边，使用水平分割线
    else:
        up_points = []
        down_points = []
        for point in points:
            if point[1] < avg_y:
                up_points.append(point)
            else:
                down_points.append(point)

        assert len(up_points) == 2, "顶点排序出错，请检查new_order函数"

        up_points = sorted(up_points, key=lambda k: k[0], reverse=False)
        down_points = sorted(down_points, key=lambda k: k[0], reverse=False)
        new_points.append(up_points[0])
        new_points.append(up_points[1])
        new_points.append(down_points[1])
        new_points.append(down_points[0])
    return np.array(new_points)


def expand_points(points):
    expand_pixel = 20
    for i in range(4):
        if i == 0:
            points[i][0] -= expand_pixel
            points[i][1] -= expand_pixel
        elif i == 1:
            points[i][0] += expand_pixel
            points[i][1] -= expand_pixel
        elif i == 2:
            points[i][0] += expand_pixel
            points[i][1] += expand_pixel
        else:
            points[i][0] -= expand_pixel
            points[i][1] += expand_pixel
    return points


def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        index1  : offset of point1
        index2  : offset of point2
        r       : [r1, r2, r3, r4] in paper
        coef    : shrink ratio in paper
    Output:
        vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (length_x)
        vertices[y1_index] += ratio * (length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * (-length_x)
        vertices[y2_index] += ratio * (-length_y)
    return vertices


def shrink_poly(vt, coef=0.03):
    '''shrink the text region
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        coef    : shrink ratio in paper
    Output:
        v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vt[0][0], vt[0][1], vt[1][0], vt[1][1], vt[2][0], vt[2][1], vt[3][0], vt[3][1]
    r1 = min(cal_distance(x1, y1, x2, y2), cal_distance(x1, y1, x4, y4))
    r2 = min(cal_distance(x2, y2, x1, y1), cal_distance(x2, y2, x3, y3))
    r3 = min(cal_distance(x3, y3, x2, y2), cal_distance(x3, y3, x4, y4))
    r4 = min(cal_distance(x4, y4, x1, y1), cal_distance(x4, y4, x3, y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1, y1, x2, y2) + cal_distance(x3, y3, x4, y4) > \
            cal_distance(x2, y2, x3, y3) + cal_distance(x1, y1, x4, y4):
        offset = 0  # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1  # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = np.array([x1, y1, x2, y2, x3, y3, x4, y4])
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v.reshape((4, 2))


def is_vertical(line):
    left = line[0]
    right = line[1]
    deta_x = abs(right[0] - left[0])
    deta_y = abs(right[1] - left[1])
    if deta_y > deta_x:
        return True
    else:
        return False


# def img_proc(image, thresh, file, index):
def img_proc(thresh):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.erode(thresh, kernel)
    thresh = cv2.dilate(thresh, kernel)

    # image, contours, hierarchy = cv2.findContours(thresh, 3, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)

    contours = sorted(contours, key=lambda k: len(k), reverse=True)

    # image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    num = 1
    for i in range(num):
        cnt = contours[i]
        print('cnt: ', len(cnt))

        acc = 30
        for j in range(5):
            approx = cv2.approxPolyDP(cnt, acc, True)
            acc += 30
            print('approx: ', len(approx))
            if len(approx) <= 8:
                break
        approx = np.squeeze(approx, axis=1)
        print('approx: ', approx)

        # image = cv2.polylines(image, np.int32([approx]), True, (0, 0, 255), 6)

        assert len(approx) >= 4, "轮廓提取出错，少于4个顶点"

        if len(approx) != 4:
            lines = cpt_line(approx)

            vet_lines = []
            hor_lines = []
            for line in lines:
                if is_vertical(line):
                    vet_lines.append(line)
                else:
                    hor_lines.append(line)

            vet_lines = sorted(vet_lines, key=lambda k: dist(k), reverse=True)
            hor_lines = sorted(hor_lines, key=lambda k: dist(k), reverse=True)

            assert len(vet_lines) >= 2, '线条分类错误'
            assert len(hor_lines) >= 2, '线条分类错误'



            lines = []
            lines.append(vet_lines[0])
            lines.append(hor_lines[0])
            lines.append(vet_lines[1])
            lines.append(hor_lines[1])
            # lines = sorted(lines[:4], key=lambda k: k[2], reverse=False)
            print('lines: ', lines)

            ans_points = []
            for p in range(4):
                q = (p + 1) % 4
                line1 = lines[p]
                line2 = lines[q]
                x1, y1, x2, y2 = line1[0][0], line1[0][1], line1[1][0], line1[1][1]
                x3, y3, x4, y4 = line2[0][0], line2[0][1], line2[1][0], line2[1][1]
                x, y = GetIntersectPointofLines(x1, y1, x2, y2, x3, y3, x4, y4)
                ans_points.append([x, y])
            ans_points = np.array(ans_points)
        else:
            ans_points = approx

        assert len(ans_points) == 4, "提取顶点出错，必须是4个顶点，请检查直线交点计算过程"

        # print('ans_points: ', ans_points)
        # image = cv2.polylines(image, np.int32([ans_points]), True, (0, 255, 0), 6)

        ans_points = new_order(ans_points)
        print('new_ans_points: ', ans_points)

        # ans_points = expand_points(ans_points)
        # ans_points = shrink_poly(ans_points)
        # image = cv2.polylines(image, np.int32([ans_points]), True, (0, 0, 255), 3)

        # rootPath = '/home/ubuntu/cs/tensorpack/examples/FasterRCNN/results2'
        # save_path = os.path.join(rootPath, file[:-3] + "_" + str(index) + '.jpg')
        # cv2.imwrite(save_path, image)

        return ans_points
    # cv2.imwrite('saveimg.jpg', image)


if __name__ == '__main__':

    img = cv2.imread('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/orig.jpg', 0)
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('thresh.jpg', thresh)
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_proc(image, thresh)
