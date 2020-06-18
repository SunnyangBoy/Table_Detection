import cv2
import os
import math
import numpy as np

'''
    使用横线提取的方法后， 进行轮廓提取，筛选去除轮廓点数量少的白色区域，剩下的所有轮廓点组成一个集合，用凸包函数找外围的轮廓点，再用多边形拟合成大于等于4，小于等于6的多边形，取多边形
    最长的两条横线，最长两条竖线，计算4条线的交点就是最终顶点
'''

# 计算两点形成的直线
def GeneralEquation(first_x, first_y, second_x, second_y):
    # 一般式 Ax+By+C=0
    A = second_y-first_y
    B = first_x-second_x
    C = second_x * first_y - first_x * second_y
    return A, B, C

# 计算两条直线的交点
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

# 计算两点之间的距离
def dist(key_point):
    left = key_point[0]
    right = key_point[1]
    return math.sqrt(math.pow((left[0] - right[0]), 2) + math.pow((left[1] - right[1]), 2))

# 输入点集，返回相互连接的线段
# line[0]是起始端点，line[1]是终止端点，line[2]是形成线段时的顺序
def cpt_line(points):
    lines = []
    pt_num = len(points)
    for p in range(pt_num):
        q = (p + 1) % pt_num
        left = points[p]
        right = points[q]
        lines.append([left, right, p])
    return lines

# 对4个顶点进行排序，按照左上，右上，右下，左下顺序
def new_order(points):
    new_points = []
    left_points = []
    right_points = []
    avg_x = np.mean(points[:, 0])
    for point in points:
        if point[0] < avg_x:
            left_points.append(point)
        else:
            right_points.append(point)

    # assert len(left_points) == 2, "顶点排序出错，请检查new_order函数"
    if len(left_points) != 2:
        return None

    left_points = sorted(left_points, key=lambda k: k[1], reverse=False)
    right_points = sorted(right_points, key=lambda k: k[1], reverse=False)
    new_points.append(left_points[0])
    new_points.append(right_points[0])
    new_points.append(right_points[1])
    new_points.append(left_points[1])
    return np.array(new_points)

# 输入角度，旋转图片
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    theta = angle * math.pi / 180
    a = math.sin(theta)
    b = math.cos(theta)
    wdst = int(h * abs(a) + w * abs(b))
    hdst = int(w * abs(a) + h * abs(b))
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    M[0, 2] += (wdst - w) / 2  # 重点在这步，目前不懂为什么加这步
    M[1, 2] += (hdst - h) / 2  # 重点在这步
    
    rotated = cv2.warpAffine(image, M, (wdst, hdst))
    return rotated

# 计算两点形成的线段的方位角
def azimuthAngle(x1, y1, x2, y2):
    assert x1 <= x2
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if x2 == x1:
        angle = math.pi / 2.0
        if y2 == y1:
            angle = 0.0
        elif y2 < y1:
            angle = -(math.pi / 2.0)
    elif y2 > y1:
        angle = math.atan(dy / dx)
    elif y2 < y1:
        angle = - math.atan(-dy / dx)
    return angle * 180 / math.pi

# 判断生成的顶点是否正常，排除斜率和面积异常的情况
def not_avaliable(points, w, h):
    up_angle = abs_angle(points[0][0], points[0][1], points[1][0], points[1][1])
    dw_angle = abs_angle(points[3][0], points[3][1], points[2][0], points[2][1])
    rt_angle = abs_angle(points[1][0], points[1][1], points[2][0], points[2][1])
    lf_angle = abs_angle(points[0][0], points[0][1], points[3][0], points[3][1])

    if abs(up_angle - dw_angle) >= 6 or abs(rt_angle - lf_angle) >= 6 \
            or (points[2][0] - points[0][0]) * (points[2][1] - points[0][1]) <= w*h/4:
        return True
    return False

# 判断是横线还是竖线，竖线返回True
def is_vertical(line):
    left = line[0]
    right = line[1]
    deta_x = abs(right[0] - left[0])
    deta_y = abs(right[1] - left[1])
    if deta_y > deta_x:
        return True
    else:
        return False

# 判断计算的交点是否超出图片范围
def outof_img(x, y, height, width):
    print('out img ', x, y, width, height)
    if x < -0.5 or x > width+0.5 or y < -0.5 or y > height+0.5:
        return True
    return False

# 计算两点形成的线段的绝对角度0-90度
def abs_angle(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dx == 0:
        angle = math.pi / 2.0
    else:
        angle = math.atan(dy / dx)
    return angle * 180 / math.pi

# 判断异常的直线
def bad_line(lines, p, q):
    pline1 = abs_angle(lines[p][0][0], lines[p][0][1], lines[p][1][0], lines[p][1][1])
    p2 = (p + 2) % 4
    pline2 = abs_angle(lines[p2][0][0], lines[p2][0][1], lines[p2][1][0], lines[p2][1][1])

    qline1 = abs_angle(lines[q][0][0], lines[q][0][1], lines[q][1][0], lines[q][1][1])
    q2 = (q + 2) % 4
    qline2 = abs_angle(lines[q2][0][0], lines[q2][0][1], lines[q2][1][0], lines[q2][1][1])

    if abs(pline1 - pline2) < abs(qline1 - qline2):
        return q
    return p


def proc_opencv(imgpath):
    imges = os.listdir(imgpath)
    #file = "00170_0.jpg"
    #if file == '00170_0.jpg':
    for t, file in enumerate(sorted(imges)):
        print(file)
        origimg = cv2.imread(os.path.join(imgpath, file))
        rows, cols = origimg.shape[:2]
        gray = cv2.cvtColor(origimg, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 121, 22)

        # 闭运算，避免断线
        kernel = np.ones((3, 3), np.uint8)
        thresh0 = cv2.dilate(thresh, kernel, iterations=1)
        thresh1 = cv2.erode(thresh0, kernel, iterations=1)

        # 竖向膨胀，有利于斜线的提取
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8), (-1, -1))
        thresh2 = cv2.dilate(thresh1, kernel, iterations=1)

#**************** 新增部分 用该部分代替 去除部分 - 1 ******************************
        # 注意!!! 第181行 和 193行 的函数第一个参数接的是上面的结果 thresh2
        hflines = cv2.HoughLinesP(thresh2, 1, np.pi / 180, 200, minLineLength=cols // 2, maxLineGap=100)
        hfangel = 0.0
        if hflines is not None:
            hfangels = []
            for hfline in hflines:
                x1, y1, x2, y2 = hfline[0]
                # 计算方位角，统计所有横线角度
                tmp_angle = azimuthAngle(x1, y1, x2, y2)
                if abs(tmp_angle) < 45: # 避免引入竖线角度
                    hfangels.append(tmp_angle)
            if len(hfangels) != 0:
                hfangel = np.mean(hfangels) # 所有横线角度的平均值
        # 角度校正
        rothresh = rotate(thresh2, hfangel)
#**********************************************

#****************** 原部分有改动!!! 199行的函数第一个参数是上面的结果 rothresh ****************************
        # 开运算，提取直线
        hline = cv2.getStructuringElement(cv2.MORPH_RECT, ((cols // 16), 1), (-1, -1))
        openlines = cv2.morphologyEx(rothresh, cv2.MORPH_OPEN, hline)


        # 去除部分 - 1
        '''
        # 霍夫直线检测，仅检测横线
        hflines = cv2.HoughLinesP(openlines, 1, np.pi / 180, 200, minLineLength=cols//2, maxLineGap=100)
        hfangel = 0.0
        if hflines is not None:
            hfangels = []
            for hfline in hflines:
                x1, y1, x2, y2 = hfline[0]
                # 计算方位角，统计所有横线角度
                tmp_angle = azimuthAngle(x1, y1, x2, y2)
                if abs(tmp_angle) < 50: # 避免引入竖线角度
                    hfangels.append(tmp_angle)
            hfangel = np.mean(hfangels) # 所有横线角度的平均值
        # 角度校正
        rothresh = rotate(openlines, hfangel)

        # 横线膨胀，避免断线
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1), (-1, -1))
        rothresh = cv2.dilate(rothresh, kernel, iterations=1)
        '''

        # 去除部分 - 2
        '''
        # 计算所有的轮廓点
        contours, hierarchy = cv2.findContours(rothresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda k: len(k), reverse=True) # 按照轮廓点数量从大到小排序

        flag = 0 # 放弃做处理的标志

        if contours is not None and len(contours) > 0:
            hull = []
            maxlen = len(contours[0])
            # 将所有轮廓点合成一个集合，计算凸包
            for k in range(len(contours)):
                if len(contours[k]) > maxlen // 3: # 过滤掉小于最大轮廓数量3分之一的轮廓
                    for cont in contours[k]:
                        hull.append(cont)
            hull = np.array([hull])
            hull = cv2.convexHull(hull[0])
        '''
#**************** 新增部分 用该部分代替 去除部分 - 2 ******************************
        # 注意!!! 第245行 的函数第一个参数接的是上面的结果 openlines
        contours, hierarchy = cv2.findContours(openlines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contours = sorted(contours, key=lambda k: len(k), reverse=True) # 按照轮廓点数量从大到小排序, 被去除
        flag = 0

        if contours is not None and len(contours) > 0:
            hull = []
            # 找每个轮廓集中的宽度并记录在k_length中，并取其中最大宽度记录在maxlen中
            k_length = []
            maxlen = -1
            for k in range(len(contours)):
                x_min = 100000
                y_min = 100000
                x_max = -1
                y_max = -1
                for cont in contours[k]:
                    if cont[0][0] < x_min:
                        x_min = cont[0][0]
                        y_min = cont[0][1]
                    if cont[0][0] > x_max:
                        x_max = cont[0][0]
                        y_max = cont[0][1]
                # 过滤掉贯穿整张图片的横线 和 触碰到上下边缘的横线
                if x_min > 3 and x_max < rothresh.shape[1] - 3 and y_min > 3 and y_min < rothresh.shape[0] - 3 and y_max > 3 and y_max < rothresh.shape[0] - 3:
                # if x_max - x_min < rothresh.shape[1] - 3 and y_min != 0 and y_min != rothresh.shape[0] - 1 and y_max != 0 and y_max != rothresh.shape[0] - 1:
                    k_length.append([x_max - x_min, k])
                    if x_max - x_min > maxlen:
                        maxlen = x_max - x_min
            # 大于最长横线的1/2的横线可以被送入属于表格横线的集合中
            for k in range(len(k_length)):
                if k_length[k][0] > maxlen / 2:
                    for cont in contours[k_length[k][1]]:
                        hull.append(cont)
            hull = np.array([hull])
            hull = cv2.convexHull(hull[0])
#**********************************************

            # 多边形逼近
            acc = 3
            for j in range(5):
                approx = cv2.approxPolyDP(hull, acc, True)
                acc += 6
                if len(approx) <= 6:
                    break
            approx = np.squeeze(approx, axis=1)

            # assert len(approx) >= 4, "轮廓提取出错，少于4个顶点"
            if len(approx) > 4:
                lines = cpt_line(approx) # 连接成线段

                # 分类横线和竖线
                vet_lines = []
                hor_lines = []
                for line in lines:
                    if is_vertical(line):
                        vet_lines.append(line)
                    else:
                        hor_lines.append(line)
                # 横线和竖线分别排序，按照长度从大到小
                vet_lines = sorted(vet_lines, key=lambda k: dist(k), reverse=True)
                hor_lines = sorted(hor_lines, key=lambda k: dist(k), reverse=True)

                # assert len(vet_lines) >= 2, '线条分类错误'
                # assert len(hor_lines) >= 2, '线条分类错误'
                if len(vet_lines) >= 2 and len(hor_lines) >= 2:
                    lines = []
                    lines.append(vet_lines[0])
                    lines.append(hor_lines[0])
                    lines.append(vet_lines[1])
                    lines.append(hor_lines[1])
                    lines = sorted(lines[:4], key=lambda k: k[2], reverse=False)
                    # 计算横竖线交点
                    ans_points = []
                    out_lines = []
                    p = 0
                    while p < 4:
                        q = (p + 1) % 4
                        line1 = lines[p]
                        line2 = lines[q]
                        x1, y1, x2, y2 = line1[0][0], line1[0][1], line1[1][0], line1[1][1]
                        x3, y3, x4, y4 = line2[0][0], line2[0][1], line2[1][0], line2[1][1]
                        x, y = GetIntersectPointofLines(x1, y1, x2, y2, x3, y3, x4, y4)
                        height, width = rothresh.shape[:2]
                        if outof_img(x, y, height, width): # 统计交点超出图片范围的两条直线的序号
                            out_lines.append([p, q])
                        ans_points.append([x, y])
                        p += 1
                    if len(out_lines) > 0:
                        for out_line in out_lines: # 放弃产生干扰的那条直线，使用正常线段的端点当做最终顶点
                            bad = bad_line(lines, out_line[0], out_line[1])
                            ans_points[(bad-1) % 4] = [lines[(bad-1) % 4][1][0], lines[(bad-1) % 4][1][1]]
                            ans_points[bad] = [lines[(bad+1) % 4][0][0], lines[(bad+1) % 4][0][1]]
                    ans_points = np.array(ans_points)
                    
                    rotimg = rotate(origimg, hfangel)
                    h, w = rotimg.shape[:2]
                    ans_points = new_order(ans_points)
                    if ans_points is None or not_avaliable(ans_points, w, h):
                        ans_points = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
                    drawimage = cv2.polylines(rotimg, np.int32([ans_points]), True, (0, 255, 0), 10)
                    save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_pdf',
                                            file[:-4] + '_img' + '.jpg')
                    cv2.imwrite(save_path, drawimage)
                else:
                    flag = 1
            elif len(approx) == 4:
                ans_points = approx

                rotimg = rotate(origimg, hfangel)
                h, w = rotimg.shape[:2]
                ans_points = new_order(ans_points)
                if not_avaliable(ans_points, w, h):
                    ans_points = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

                drawimage = cv2.polylines(rotimg, np.int32([ans_points]), True, (0, 255, 0), 10)
                save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_pdf',
                                         file[:-4] + '_img' + '.jpg')
                cv2.imwrite(save_path, drawimage)
            else:
                flag = 1
        else:
            flag = 1

        if flag:
            # 不做处理
            h, w = origimg.shape[:2]
            ans_points = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

            drawimage = cv2.polylines(origimg.copy(), np.int32([ans_points]), True, (0, 255, 0), 10)
            save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_pdf',
                                     file[:-4] + '_img' + '.jpg')
            cv2.imwrite(save_path, drawimage)

if __name__ == "__main__":
    imgpath = '/home/ubuntu/cs/publaynet/publaynet_data/opencv_test_T3'
    proc_opencv(imgpath)
