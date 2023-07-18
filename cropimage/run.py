from cropimage import Cropper
import cv2
cropper = Cropper()

# Get a Numpy array of the cropped image
# Set completeness to be True if you expect the 'face' to be focused rather than 'person'
# Set target_size to be a tuple (size, size), only square output is supported now
result = cropper.crop('/data/heyue/makeup_related/cropimage/images/input.jpg')

# Save the cropped image
cv2.imwrite('cropped.jpg', result)