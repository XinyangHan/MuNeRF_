import cv2
import os

root = "/data/heyue/104/dataset/1110/girl10_5_new"
root1= "/data/heyue/104/dataset/1110/girl10_5_raw"
save_root = "/data/heyue/104/dataset/1110/cropped/munerf"
save_root1= "/data/heyue/104/dataset/1110/cropped/raw"

things=  os.listdir(root)
things.sort()

path = things[5]
img = cv2.imread(os.path.join(root, path))
img1 = cv2.imread(os.path.join(root1, path))

# print(img.shape)
# print(img1.shape)

img = cv2.resize(img, (512,512))


# print(img.shape)
# print(img1.shape)

x0 = 55
y0 = 70
d = 330

img = img[y0:y0+d, x0:x0+d]
img1 = img1[y0:y0+d, x0:x0+d]
cv2.imwrite(os.path.join(save_root, path), img)
cv2.imwrite(os.path.join(save_root1, path), img1)