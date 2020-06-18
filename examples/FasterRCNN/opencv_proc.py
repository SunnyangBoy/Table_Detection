import cv2
import os
import math
import numpy as np

'''
    使用横线提取的方法后， 进行轮廓提取，筛选去除轮廓点数量少的区域，剩下的所有轮廓点组合在一起，用找凸包的函数处理，再用多边形拟合方式
'''

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


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    theta = angle * math.pi / 180.0
    a = math.sin(theta)
    b = math.cos(theta)
    wdst = int(h * abs(a) + w * abs(b))
    hdst = int(w * abs(a) + h * abs(b))
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    M[0, 2] += (wdst - w) / 2  # 重点在这步，目前不懂为什么加这步
    M[1, 2] += (hdst - h) / 2  # 重点在这步

    rotated = cv2.warpAffine(image, M, (wdst, hdst), borderValue=(255, 255, 255))
    return rotated


def azimuthAngle(x1, y1, x2, y2):
    assert x1 <= x2
    angle = 0.0
    dx = (x2 - x1) * 1.0
    dy = (y2 - y1) * 1.0
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


def not_avaliable(points, w, h):
    up_angle = abs_angle(points[0][0], points[0][1], points[1][0], points[1][1])
    dw_angle = abs_angle(points[3][0], points[3][1], points[2][0], points[2][1])
    rt_angle = abs_angle(points[1][0], points[1][1], points[2][0], points[2][1])
    lf_angle = abs_angle(points[0][0], points[0][1], points[3][0], points[3][1])
    print('h_angle ', abs(up_angle - dw_angle))
    print('v_angle ', abs(rt_angle - lf_angle))
    print(rt_angle)
    print(lf_angle)
    print(up_angle)
    print(dw_angle)
    print('area ', (points[2][0] - points[0][0]) * (points[2][1] - points[0][1]), w * h / 4)
    if abs(up_angle - dw_angle) >= 6 or abs(rt_angle - lf_angle) >= 6 \
            or (points[2][0] - points[0][0]) * (points[2][1] - points[0][1]) <= w*h/4:
        return True
    return False


def is_vertical(line):
    left = line[0]
    right = line[1]
    deta_x = abs(right[0] - left[0])
    deta_y = abs(right[1] - left[1])
    if deta_y > deta_x:
        return True
    else:
        return False


def outof_img(x, y, height, width):
    print('out img ', x, y, width, height)
    if x < -0.5 or x > width+0.5 or y < -0.5 or y > height+0.5:
        return True
    return False


def abs_angle(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dx == 0:
        angle = math.pi / 2.0
    else:
        angle = math.atan(dy / dx)
    return angle * 180 / math.pi


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
    # file = "961587641794_.pic_hd.jpg"
    # if file == '961587641794_.pic_hd.jpg':
    for t, file in enumerate(sorted(imges)):
        print(file)
        origimg = cv2.imread(os.path.join(imgpath, file))
        rows, cols = origimg.shape[:2]
        gray = cv2.cvtColor(origimg, cv2.COLOR_BGR2GRAY)
        # gray = cv2.blur(gray, (3, 3))
        # erosion = cv2.erode(gray, kernel, iterations=3)
        # threshold = 35
        # thresh = cv2.Canny(erosion, threshold, threshold * 3, apertureSize=3)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 121, 22)

        kernel = np.ones((3, 3), np.uint8)
        thresh0 = cv2.dilate(thresh, kernel, iterations=1)
        thresh1 = cv2.erode(thresh0, kernel, iterations=1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8), (-1, -1))
        thresh2 = cv2.dilate(thresh1, kernel, iterations=1)

        # save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_pdf', file[:-4]
        #                          + "_thresh" + '.jpg')
        # cv2.imwrite(save_path, thresh2)
        '''
        newImage = thresh.copy()
        results = []
        for i in range(rows):
            for j in range(cols):
                if newImage[i][j] == 255:
                    newImage[i, j] = 0
                    m = 0
                    minc = j
                    maxc = j
                    mylist = []
                    mylist.append([i, j])
                    while m < len(mylist):
                        rc = mylist[m]
                        rcs = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
                        for k in range(len(rcs)):
                            nr = rc[0] + rcs[k][0]
                            nc = rc[1] + rcs[k][1]
                            if nr > 0 and nr < rows and nc > 0 and nc < cols and newImage[nr, nc] == 255:
                                mylist.append([nr, nc])
                                newImage[nr, nc] = 0
                                if nc < minc:
                                    minc = nc
                                if nc > maxc:
                                    maxc = nc
                        m += 1
                    if (maxc - minc) > (cols * 0.3):
                        results.append(mylist)

        for i in range(len(results)):
            mylist2 = results[i]
            for j in range(len(mylist2)):
                rc = mylist2[j]
                newImage[rc[0], rc[1]] = 255
        print('finish')
        '''
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
        '''
        scale = 200
        erode_iters = 1
        dilate_iters = 1
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
        dilatedcolh = cv2.dilate(thresh, kernel1, iterations=dilate_iters)
        erodedh = cv2.erode(dilatedcolh, kernel2, iterations=erode_iters)
        '''
        hflines = cv2.HoughLinesP(thresh2, 1, np.pi / 180, 200, minLineLength=cols // 2, maxLineGap=100)
        draw = cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR)
        hfangel = 0.0
        if hflines is not None:
            hfangels = []
            for hfline in hflines:
                x1, y1, x2, y2 = hfline[0]
                tmp_angle = azimuthAngle(x1, y1, x2, y2)
                if abs(tmp_angle) < 45:
                    print(abs(tmp_angle))
                    hfangels.append(tmp_angle)
                    draw = cv2.line(draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
            save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_pdf', file[:-4]
                                      + "_huofline" + '.jpg')
            cv2.imwrite(save_path, draw)
            if len(hfangels) != 0:
                hfangel = np.mean(hfangels)
        print('angle ', hfangel)
        rothresh = rotate(thresh2, hfangel)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1), (-1, -1))
        # rothresh = cv2.dilate(rothresh, kernel, iterations=1)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6), (-1, -1))
        # rothresh = cv2.erode(rothresh, kernel, iterations=1)
        # hline = cv2.getStructuringElement(cv2.MORPH_RECT, ((cols // 16), 1), (-1, -1))
        # openlines = cv2.morphologyEx(rothresh, cv2.MORPH_OPEN, hline)
        # save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_pdf', file[:-4]
        #                          + "_rotthresh" + '.jpg')
        # cv2.imwrite(save_path, rothresh)

        hline = cv2.getStructuringElement(cv2.MORPH_RECT, ((cols // 16), 1), (-1, -1))
        openlines = cv2.morphologyEx(rothresh, cv2.MORPH_OPEN, hline)

        '''
        # 识别竖线
        scale = 50
        erode_iters = 1
        dilate_iters = 1
        rows, cols = thresh.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, cols // scale))
        #  (cols // scale, 1) 为了获取横向的表格线，设置腐蚀和膨胀的操作区域为一个比较大的横向直条
        erodedv = cv2.erode(thresh, kernel, iterations=erode_iters)
        dilatedcolv = cv2.dilate(erodedv, kernel, iterations=dilate_iters)
        save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_pdf',
                                 'threshlinev_' + str(i) + '.jpg')
        cv2.imwrite(save_path, dilatedcolv)

        save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_pdf',
                                 'threshlinemerge_' + str(i) + '.jpg')
        cv2.imwrite(save_path, cv2.add(dilatedcolh, dilatedcolv))
        
        lines = cv2.HoughLinesP(dilatedcol, 1, np.pi / 180, 100, minLineLength=cols//2, maxLineGap=200)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            origimg = cv2.line(origimg, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 点的坐标必须是元组，不能是列表

        save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_pdf',
                                         'huofuline_' + str(i) + '.jpg')
        cv2.imwrite(save_path, dilatedcol)
        '''

        contours, hierarchy = cv2.findContours(openlines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contours = sorted(contours, key=lambda k : len(k), reverse=True) 去除这一步
        draw = cv2.cvtColor(openlines, cv2.COLOR_GRAY2BGR)

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

            drawimage = cv2.polylines(draw.copy(), [hull], True, (0, 255, 0), 10)
            save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_pdf',
                                       file[:-4] + '_hull' + '.jpg')
            cv2.imwrite(save_path, drawimage)

            acc = 3
            for j in range(5):
                approx = cv2.approxPolyDP(hull, acc, True)
                acc += 6
                print('approx: ', len(approx))
                if len(approx) <= 6:
                    break
            approx = np.squeeze(approx, axis=1)
            print('approx: ', approx)

            drawimage = cv2.polylines(draw.copy(), [approx], True, (255, 0, 0), 6)
            for pot in approx:
                cv2.circle(drawimage, (pot[0], pot[1]), 8, color=(0, 0, 255), thickness=12)
            save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_pdf',
                                     file[:-4] + '_approx' + '.jpg')
            cv2.imwrite(save_path, drawimage)

            # assert len(approx) >= 4, "轮廓提取出错，少于4个顶点"
            if len(approx) > 4:
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

                # assert len(vet_lines) >= 2, '线条分类错误'
                # assert len(hor_lines) >= 2, '线条分类错误'
                if len(vet_lines) >= 2 and len(hor_lines) >= 2:
                    lines = []
                    lines.append(vet_lines[0])
                    lines.append(hor_lines[0])
                    lines.append(vet_lines[1])
                    lines.append(hor_lines[1])
                    lines = sorted(lines[:4], key=lambda k: k[2], reverse=False)
                    print('lines: ', lines)

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
                        if outof_img(x, y, height, width):
                            out_lines.append([p, q])
                        ans_points.append([x, y])
                        p += 1
                    if len(out_lines) > 0:
                        for out_line in out_lines:
                            print('bad line: ', out_line[0], out_line[1])
                            bad = bad_line(lines, out_line[0], out_line[1])
                            print('bad line: ', bad)
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
            h, w = origimg.shape[:2]
            ans_points = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

            drawimage = cv2.polylines(origimg.copy(), np.int32([ans_points]), True, (0, 255, 0), 10)
            save_path = os.path.join('/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_pdf',
                                     file[:-4] + '_img' + '.jpg')
            cv2.imwrite(save_path, drawimage)


if __name__ == "__main__":
    imgpath = '/home/ubuntu/cs/tensorpack/examples/FasterRCNN/result_test/result_photo'
    proc_opencv(imgpath)
