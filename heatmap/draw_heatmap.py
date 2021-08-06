import cv2
import numpy as np

movie_path = r"C:\myapps\sample-videos-master\movie_data\mall_origin01.MOV"

def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 25)
    img_canny = cv2.Canny(img_blur, 5, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=4)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode

def get_contours(img, img_original):
    img_contours = img_original.copy()
    contours, hierarchies = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), -1) 
    # If you want to omit smaller contours, loop through the detected contours, and only draw them on the image if they are at least a specific area. Don't forget to remove the line above if you choose the below block of code.
    # for cnt in contours: 
    #     if cv2.contourArea(cnt) > 500:
    #         cv2.drawContours(img_contours, [cnt], -1, (0, 255, 0), -1) 

    return img_contours

cap = cv2.VideoCapture(movie_path)

success, img1 = cap.read()
success, img2 = cap.read()
heat_map = np.zeros(img1.shape[:-1])

while success:
    diff = cv2.absdiff(img1, img2)
    img_contours = get_contours(process(diff), img1)

    heat_map[np.all(img_contours == [0, 255, 0], 2)] += 3 # The 3 can be tweaked depending on how fast you want the colors to respond
    heat_map[np.any(img_contours != [0, 255, 0], 2)] -= 3
    heat_map[heat_map < 0] = 0
    heat_map[heat_map > 255] = 255

    img_mapped = cv2.applyColorMap(heat_map.astype('uint8'), cv2.COLORMAP_JET)

#    img1[heat_map > 160] = img_mapped[heat_map > 160] Use this line to draw the heat map on the original video at a specific temperature range. For this it's where ever the temperature is above 160 (min is 0 and max is 255)

    # cv2.imshow("Original", img1)
    # cv2.imshow("Heat Map", img_mapped)

    assert img1.shape == img_mapped.shape
    
    alpha = 0.7
    blended = cv2.addWeighted(img1, alpha, img_mapped, 1 - alpha, 0)

    cv2.imshow("Blended", blended)

    img1 = img2
    success, img2 = cap.read()
    
    if cv2.waitKey(1) == ord('q'):
        break