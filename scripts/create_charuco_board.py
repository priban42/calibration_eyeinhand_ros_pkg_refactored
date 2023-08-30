import cv2
import numpy as np
import copy
from PIL import Image

def rectangle_not_rounded(image, x, y, width, height, thickness, color):
    for i in range(thickness):
        x = x+1
        y = y+1
        width = width -2
        height = height -2
        cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness=1)
    return image

def generate_aruco_board(board_size, square_size, marker_length, output_file, img_size, aruco_dictionary):
    # Define the dictionary and board parameters
    aruco_dictionary
    charuco_board = cv2.aruco.CharucoBoard(
        board_size, square_size, marker_length, aruco_dictionary
    )

    board_image = charuco_board.generateImage(img_size, marginSize=0, borderBits=1)

    cv2.imwrite(output_file, board_image)
    print("ArUco board image saved as", output_file)
    return board_image

def mini_tags(img, scale):
    xs = [0, 1550, 800, 800, 450, 1100, 450, 1100, 800]
    ys = [550, 550, 0, 1100, 350, 350, 750, 750, 550]
    width, height = 100, 100
    x_offset_m = 10*scale
    y_offset_m = 20*scale

    # correction around the mini tags - (0 or 10)
    correction = 10*scale

    for i in range(9):
        xs[i], ys[i] = xs[i]*scale, ys[i]*scale
        rectangle_not_rounded(img, xs[i]-1, ys[i]+correction-1, scale*width+1, scale*height+1-2*correction, 50*scale, (255, 255, 255))
        img[ys[i]+y_offset_m:ys[i]+y_offset_m+mini_image_size[1],xs[i]+x_offset_m:xs[i]+x_offset_m+mini_image_size[0]] = mini_board
    return img

# IMAGE SCALE
scale = 1

# MAIN BOARD
board_size = (33, 24)  # Number of squares (interior points)
square_size = 50  # Size of each square (in pixels)
marker_length = 30  # Length of each marker (in pixels)
output_file = "aruco_board_base.bmp"
image_size = (scale*1650, scale*1200)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
board_image = generate_aruco_board(board_size, square_size, marker_length, output_file, image_size, dictionary)

# MINI BOARD
mini_board_size = (8, 6)
mini_square_size = 10
mini_marker_length = 6
mini_output_file = "aruco_board_mini.bmp"
mini_image_size = (scale*80, scale*60)
dictionary_mini = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
mini_board = generate_aruco_board(mini_board_size, mini_square_size, mini_marker_length, mini_output_file, mini_image_size, dictionary_mini)

# ---MINI TAGS---
# White rectangles under the mini tags
board_image = mini_tags(board_image, scale)
board_copy = copy.deepcopy(board_image)

# ---WHITE BASE---
# white base rectangle
x, y = 230*scale, 200*scale
width, height = 1190*scale, 800*scale
rectangle_not_rounded(board_image, x-1, y-1, width+1, height+1, 20*scale, (255, 255, 255))

# ---CUTTING LINES---
# cutting line - inner
x, y = 245*scale, 215*scale
width, height = 1160*scale, 770*scale
rectangle_not_rounded(board_image, x-1, y-1, width+1, height+1, 1*scale, (0, 0, 0))

# cutting line - outer
x, y = 235*scale, 205*scale
width, height = 1180*scale, 790*scale
rectangle_not_rounded(board_image, x-1, y-1, width+1, height+1, 1*scale, (0, 0, 0))


#cv2.imshow('1',board_image)
cv2.imwrite("aruco_board_result.png", board_image)
#cv2.waitKey(0)

split = True
if split:
    x, y = 245*scale, 215*scale
    width, height = 1160*scale, 770*scale
    #cropped_img = board_copy[x:x+width, y:y+height]
    cropped_img = board_copy[y:y+height, x:x+width]
    cv2.imshow('2',cropped_img)
    cv2.imwrite("aruco_board_c.bmp", cropped_img)
    cv2.waitKey(0)

    x, y = 245*scale, 215*scale
    width, height = 1160*scale, 770*scale
    #rectangle_not_rounded(board_image_copy, x-1, y-1, width+1, height+1, 1*scale, (0, 0, 0))
    cv2.rectangle(board_copy, (x, y), (x + width, y + height), (255, 255, 255), thickness=-1)

    crop_ratio = 19.5/33

    width= int(board_copy.shape[1]*1/33/1.5) #*scale
    x = int(board_copy.shape[1]*crop_ratio)-int(width/2)
    y0, y1 = 215*scale, board_copy.shape[0]-215*scale
    cv2.rectangle(board_copy, (x, y0), (x + width, y1), (255, 255, 255), thickness=-1)

    cv2.line(board_copy, (x + int(width/2), y0), (x + int(width/2), y1), (0,0,0), thickness=2*scale)

    num_cross = 100
    step = board_copy.shape[1]/(num_cross+1)
    cross_position = 0
    for i in range(num_cross):
        cross_position += step
        if(cross_position < y0 or cross_position > y1):
            continue
        cv2.line(board_copy, (x, int(cross_position)), (x+width, int(cross_position)), (0,0,0), thickness=1*scale) 
    
    #cv2.imshow('3',board_copy)
    cv2.imwrite("aruco_board_test.bmp", board_copy)
    #cv2.waitKey(0)

    cropped_img = board_copy[0:board_copy.shape[0], 0:int(board_copy.shape[1]*crop_ratio)]
    cv2.imshow('4',cropped_img)
    cv2.imwrite("aruco_board_a.bmp", cropped_img)
    cv2.waitKey(0)

    cropped_img = board_copy[0:board_copy.shape[0], int(board_copy.shape[1]*crop_ratio)+1:board_copy.shape[1]]
    cv2.imshow('5',cropped_img)
    cv2.imwrite("aruco_board_b.bmp", cropped_img)
    cv2.waitKey(0)

    image_1 = Image.open('aruco_board_a.bmp')
    im_1 = image_1.convert('RGB')
    im_1.save('a.pdf')

    image_1 = Image.open('aruco_board_b.bmp')
    im_1 = image_1.convert('RGB')
    im_1.save('b.pdf')

    image_1 = Image.open('aruco_board_c.bmp')
    im_1 = image_1.convert('RGB')
    im_1.save('c.pdf')

else:
    x, y = 245*scale, 215*scale
    width, height = 1160*scale, 770*scale
    #cropped_img = board_copy[x:x+width, y:y+height]
    cropped_img = board_copy[y:y+height, x:x+width]
    cv2.imshow('inner part',cropped_img)
    cv2.imwrite("inner part.bmp", cropped_img)
    cv2.waitKey(0)
    inner_part = copy.deepcopy(cropped_img)

    x, y = 245*scale, 215*scale
    width, height = 1160*scale, 770*scale
    #rectangle_not_rounded(board_image_copy, x-1, y-1, width+1, height+1, 1*scale, (0, 0, 0))
    cv2.rectangle(board_copy, (x, y), (x + width, y + height), (255, 255, 255), thickness=-1)

    crop_ratio = 19.5/33

    width= int(board_copy.shape[1]*1/33/1.5) #*scale
    x = int(board_copy.shape[1]*crop_ratio)-int(width/2)
    cv2.rectangle(board_copy, (x, 0), (x + width, board_copy.shape[0]), (255, 255, 255), thickness=-1)

    cv2.line(board_copy, (x + int(width/2), 0), (x + int(width/2), board_copy.shape[0]), (0,0,0), thickness=2*scale)

    num_cross = 100
    step = board_copy.shape[1]/(num_cross+1)
    cross_position = 0
    for i in range(num_cross):
        cross_position += step
        cv2.line(board_copy, (x, int(cross_position)), (x+width, int(cross_position)), (0,0,0), thickness=1*scale) 
    
    cv2.imshow('full',board_copy)
    cv2.waitKey(0)    



    x, y = 245*scale, 215*scale

    sh, sw = cropped_img.shape[:2]
    board_copy[y:y+sh, x:x+sw] = inner_part


    cv2.imwrite("aruco_board_full.bmp", board_copy)
    cv2.imshow("aruco_board_full", board_copy)
    cv2.waitKey(0)

cv2.destroyAllWindows()