from glob import glob
from os.path import join
import cv2
import numpy as np
import pytesseract
import collections
import pandas as pd

PATH_TO_DATA = '/home/aynes/Desktop/WorkSpace/Expload_pipline/Data/Игра 1 _ 30.07.19/Эмулятор/3 игра'
PATH_TO_EXEL = '/home/aynes/Desktop/WorkSpace/Expload_pipline/Data/Игра 1 _ 30.07.19/Universal Summer Cup _ Group-A _ 3 игра.xlsx'

def pipline(PATH_TO_DATA, PATH_TO_EXEL):
    leaderboard = []
    for path_to_screenshot in glob(join(PATH_TO_DATA, '*')):
        screenshot = cv2.imread(path_to_screenshot)
        print(path_to_screenshot)
        h, w = screenshot.shape[:2]

        left_half = screenshot[:, : w // 2]
        right_half = screenshot[:, w // 2:]

        image = right_half
        place_to_names_and_kills = {}
        bounding_boxes = line_detection(image)

        num_lines = len(bounding_boxes)

        kills_on_image = []
        names_on_image = []
        places_on_image = []
        for box in bounding_boxes:
            x, y, w, h = box
            line_image = image[y:y + h, x:x + w]
            line_image = cv2.bitwise_not(line_image)

            kills_box, name_box, place_box = column_detection_by_hends(box, image)

            place = place_recognition(place_box)
            name = name_recognition(name_box)
            kills = kills_recognition(kills_box)[0]

            places_on_image.append(place)
            names_on_image.append(name)
            kills_on_image.append(kills)
        places_on_image = find_place_list(places_on_image, num_lines)
        screenshot_leaderboard = {}
        for i, id in enumerate(places_on_image):
            screenshot_leaderboard[id] = [names_on_image[i], kills_on_image[i]]
        leaderboard.append(screenshot_leaderboard)

    image = left_half
    bounding_boxes = line_detection(image)

    kills_on_image = []
    names_on_image = []
    places_on_image = [1,2,3,4,5,6]
    for box in bounding_boxes:
        x, y, w, h = box
        line_image = image[y:y + h, x:x + w]
        line_image = cv2.bitwise_not(line_image)

        kills_box, name_box, place_box = column_detection_by_hends(box, image)

        name = name_recognition(name_box)
        kills = kills_recognition(kills_box)[0]
        names_on_image.append(name)
        kills_on_image.append(kills)

    print(names_on_image)
    print(kills_on_image)

    top_leaderboard = {}
    for i, id in enumerate(places_on_image):
        top_leaderboard[id] = [names_on_image[i], kills_on_image[i]]
    leaderboard.append(top_leaderboard)

    data = {}
    keys = set()
    for d in leaderboard:
        for k, v in d.items():  # d.items() in Python 3+
            if k not in keys:
                keys.add(k)
                data[k]=v[:2]

    sorted_data = collections.OrderedDict(sorted(data.items()))

    df = pd.DataFrame(sorted_data).T
    df.columns = ['Распознанные Никнеймы', 'Распознанные Киллы']


    target_df = pd.read_excel(PATH_TO_EXEL, header=1, index_col='Место')
    target_df['Распознанные Никнеймы'] = df['Распознанные Никнеймы']
    target_df['Распознанные Киллы'] = df['Распознанные Киллы']

    print(target_df)


    name_accuracy = (target_df['Никнейм'] == target_df['Распознанные Никнеймы']).sum() / target_df.shape[0]
    target_kills = target_df['Киллы'].values
    pred_kills = target_df['Распознанные Киллы'].values

    j=0
    for i, k in enumerate(target_kills):

        if  pred_kills[i].isdigit() and int(k) == int(pred_kills[i]):
            j += 1

    print( j/44)


    print('name accuracy: {}'.format(name_accuracy))

    target_df.to_excel(join('/home/aynes/Desktop/PREDICT','predict.xlsx'))

    return




def line_detection(image):
    """
    Функция для поиска строк текста на изображение.
    :param image: исходное изображение.
    :return: список координат  прямоугольников описывающих искомые строки
             например: [(x1, y1, w1, h1), (x2, y2, w2, h2), ...].
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (750, 10))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    bounding_boxes = []
    padding = 20
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        y -= padding // 2
        h += padding
        if x == 0 and w > image.shape[1] // 2 and y > 0 : #y!=image.shape[2]-h
            title = "x:{}, y:{}, w:{}, h:{}".format(x, y, w , h)
            #cv2.imshow(title, image[y:y + h, x:x + w])
            #cv2.waitKey(0)
            bounding_boxes.append((x, y, w, h))

    return bounding_boxes[::-1]

def column_detection_by_hends(box, image):
    x, y, w, h = box
    image = image[y:y + h, x:x + w]
    h,  w = image.shape[:2]
    place_box = cv2.bitwise_not(image[:h, : w//4])
    name_box =  cv2.bitwise_not(image[:h, w//5: 3*w//5])
    kills_box =  cv2.bitwise_not(image[:h, 3*w//4 :])

    #cv2.imshow('', image[:h, : w//4])
    #cv2.waitKey(0)
    return  kills_box, name_box, place_box

def place_recognition(image):


    #config = '--psm 2 -c tessedit_char_whitelist=0123456789'
    #place2 = pytesseract.image_to_string(image, config=config)

    config = '--psm 3 -c tessedit_char_whitelist=0123456789'
    place3 = pytesseract.image_to_string(image, config=config)

    config = '--psm 4 -c tessedit_char_whitelist=0123456789'
    place4 = pytesseract.image_to_string(image, config=config)

    config = '--psm 5 -c tessedit_char_whitelist=0123456789'
    place5 = pytesseract.image_to_string(image, config=config)

    config = '--psm 6 -c tessedit_char_whitelist=0123456789'
    place6 = pytesseract.image_to_string(image, config=config)

    config = '--psm 7 -c tessedit_char_whitelist=0123456789'
    place7 = pytesseract.image_to_string(image, config=config)

    config = '--psm 8 -c tessedit_char_whitelist=0123456789'
    place8 = pytesseract.image_to_string(image, config=config)

    config = '--psm 9 -c tessedit_char_whitelist=0123456789'
    place9 = pytesseract.image_to_string(image, config=config)

    config = '--psm 10 -c tessedit_char_whitelist=0123456789'
    place10 = pytesseract.image_to_string(image, config=config)[:2]

    config = '--psm 11 -c tessedit_char_whitelist=0123456789'
    place11 = pytesseract.image_to_string(image, config=config)

    config = '--psm 12 -c tessedit_char_whitelist=0123456789'
    place12 = pytesseract.image_to_string(image, config=config)

    config = '--psm 13 -c tessedit_char_whitelist=0123456789'
    place13 = pytesseract.image_to_string(image, config=config)

    p = [place3, place4, place5, place6, place7, place8, place9, place10, place11, place12, place13]
    #print('-' * 10)
    #print('3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}, 10: {}, 11: {}, 12: {}, 13: {}'.format(place3, place4, place5, place6, place7, place8, place9, place10, place11, place12, place13))
    place = -1
    for place_i in p:
        if place_i.isdigit():
            if int(place_i) > place:
                place = int(place_i)




    #cv2.imshow('place: {}'.format(place), image)
    #

    #cv2.waitKey(0)
    return str(place)

def name_recognition(image):

    config = '--psm 7'
    name = pytesseract.image_to_string(image, lang='eng', config=config)
    #print('name :',name)
    #cv2.imshow('name: {}'.format(name), image)
    #cv2.waitKey(0)
    return name

def kills_recognition(image):
    config = '--psm 10'
    kills = pytesseract.image_to_string(image, lang='eng', config=config)

    if kills == 'kill' or kills == '{kill':
        kills = '1 kill'

    if kills == 'Z kills':
        kills = '2 kills'

    if kills == 'Okills' or kills == 'okills':
        kills = '0 kills'

    if kills == '4kills':
        kills = '4 kills'

    #print('kills :',kills)
    #cv2.imshow('kills: {}'.format(kills), image)
    #cv2.waitKey(0)
    return kills






def find_place_list(place_list, n):
    alpha = 0
    for k in range(n):
        if place_list[k].isdigit() and place_list[-1-k].isdigit():
            dif = int(place_list[-1-k]) - int(place_list[k])
            #print(place_list[k], place_list[-1-k], dif)
            if dif == n-2*alpha - 1:
                result_place = range(int(place_list[k]) - alpha, int(place_list[-1-k]) + alpha + 1)
                return result_place
            else:
                alpha += 1
    for k in range(n):
        if place_list[0].isdigit():
           result_place = range(int(place_list[0]), int(place_list[0]) + n + 1)
           return result_place
        elif place_list[-1].isdigit():
             result_place = range(int(place_list[-1]) - n, int(place_list[-1]) + 1)
             return result_place

    return place_list

pipline(PATH_TO_DATA, PATH_TO_EXEL)
