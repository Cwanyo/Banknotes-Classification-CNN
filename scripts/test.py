import cv2

img = cv2.imread('../20.jpg')
print('')

w = 128
h = 128
y = int(img.shape[0] / 2)
x = int(img.shape[1] / 2)

resize = cv2.resize(img, (w, h))
crop = img[y:y + h, x:x + w]

cv2.imshow("img", img)
cv2.imshow("resize", resize)
cv2.imshow("crop", crop)
cv2.waitKey(0)
