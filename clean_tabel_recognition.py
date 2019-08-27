from glob import glob
from os.path import join
import cv2 as cv
import numpy as np
import pytesseract
import collections
import pandas as pd

PATH_TO_DATA = '../Data/Игра 1 _ 30.07.19/Эмулятор/3 игра'
PATH_TO_EXEL = '/home/aynes/Desktop/WorkSpace/Expload_pipline/Data/Игра 1 _ 30.07.19/Universal Summer Cup _ Group-A _ 3 игра.xlsx'

def pipline(PATH_TO_DATA, PATH_TO_EXEL):
    '''
    Формирует Exel таблицу с результатами матча на основе набора скриншотов.

    :param PATH_TO_DATA: путь к папке со скриншотами результатов матча.
    :param PATH_TO_EXEL: птуь к месту сохранения Exel файла с результатоми матча.
    :return: Exel файл с результатом матча (places, nicknames, kills)
    '''

    whole_leaderboard = [] # список словарей с результатами матча по каждому скриншоту
    for PATH_TO_SCREENSHOT in glob(join(PATH_TO_DATA,"*")):
        print(PATH_TO_SCREENSHOT)
        kills_on_screenshot = []
        names_on_screenshot = []
        places_on_screenshot = []

        screenshot = cv.imread(PATH_TO_SCREENSHOT)
        height, width = screenshot.shape[:2]
        left_half, right_half = screenshot[:, :width // 2], screenshot[:, width // 2:]

        # так как левая половина изображения повторяется на всех скриншотах
        # в цикле будет рассмартиваться толко правая половина изображения
        image = right_half

        row_boxes = row_detection(image)

        for box in row_boxes:
            x, y, w, h = box
            row = image[y:y + h, x:x + w]
            kills_box, place_box, name_box = column_detection(row)

            kills = kills_recornition(kills_box, row)
            place = place_recognition(place_box, row)
            name =  name_recognition(name_box, row)

            kills_on_screenshot.append(kills)
            places_on_screenshot.append(place)
            names_on_screenshot.append(name)

        leaderboard_on_screenshot = {}
        for i, id in enumerate(places_on_screenshot):
            leaderboard_on_screenshot[id] = [names_on_screenshot[i], kills_on_screenshot[i]]

        whole_leaderboard.append(leaderboard_on_screenshot)

    united_leaderboard = {}
    places = set()
    for leaderboard_on_screenshot in whole_leaderboard:
        for place, name_and_kills in leaderboard_on_screenshot.items():
            if place not in places:
                places.add(place)
                united_leaderboard[place] = name_and_kills[:2]

    united_leaderboard = collections.OrderedDict(sorted(united_leaderboard.items()))

    for place, name_and_kills in united_leaderboard.items():
        print(place,':', name_and_kills)


    return


def find_bounding_boxes(image, kernel):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    gradX = cv.Sobel(blackhat, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    gradX = cv.morphologyEx(gradX, cv.MORPH_CLOSE, kernel)
    thresh = cv.threshold(gradX, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    return contours

def row_detection(image):
    """
    Функция для поиска строк текста на изображение.
    :param image: исходное изображение.
    :return: список координат  прямоугольников описывающих искомые строки
             например: [(x1, y1, w1, h1), (x2, y2, w2, h2), ...].
    """
    bounding_boxes = []
    padding = 20

    width = image.shape[1]
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (width, 10)) #750
    contours = find_bounding_boxes(image, kernel)

    for contour in contours :
        x, y, w, h = cv.boundingRect(contour)
        y -= padding // 2
        h += padding
        if x == 0 and w > image.shape[1] // 2 and y > 0:  # y!=image.shape[2]-h
            #print("x:{}, y:{}, w:{}, h:{}".format(x, y, w, h))
            #title = "x:{}, y:{}, w:{}, h:{}".format(x, y, w, h)
            #cv.imshow(title, image[y:y + h, x:x + w])

            cv.waitKey(0)
            bounding_boxes.append((x, y, w, h))
    print()

    return bounding_boxes[::-1]

def column_detection(image):
    origin = image.copy()

    bounding_boxes = []
    padding = 20
    alpha = 2
    beta = 20

    #for h in range(image.shape[0]):
     #   for w in range(image.shape[1]):
      #      for c in range(image.shape[2]):
       #         image[h, w, c] = np.clip(alpha * image[h, w, c] + beta, 0, 255)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 10))
    contours = find_bounding_boxes(image, kernel)

    height, width = image.shape[:2]
    #print('_'*10, height, width)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w > 5:
            x -= padding // 2
            w += padding
            y = 0
            h = height
           # print("x:{}, y:{}, w:{}, h:{}".format(x, y, w, h))
            title = "x:{}, y:{}, w:{}, h:{}".format(x, y, w, h)
            #cv.imshow(title, origin[:, x:x + w])
            #cv.waitKey(0)
            bounding_boxes.append((x, y, w, h))
    return bounding_boxes # kills place names

def kills_recornition(kills_box, row):
    x, y, w, h = kills_box
    kills_image = row[y:y + h, x:x + w]
    kills_image = cv.bitwise_not(kills_image)
    config = '--psm 7'
    kills = pytesseract.image_to_string(kills_image, lang='eng', config=config)
    return kills

def place_recognition(place_box, row):
    x, y, w, h = place_box
    place_image = row[y:y + h, x:x + w]
    place_image = cv.bitwise_not(place_image)
    config = '--psm 7'
    place = pytesseract.image_to_string(place_image, lang='eng', config=config)

    return place

def name_recognition(name_box, row):
    x, y, w, h = name_box
    name_image = row[y:y + h, x:x + w]
    name_image = cv.bitwise_not(name_image)
    config = '--psm 7'
    name = pytesseract.image_to_string(name_image, lang='eng', config=config)
    return name


pipline(PATH_TO_DATA, PATH_TO_EXEL)

