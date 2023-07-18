from PIL import Image
from autocrop import Cropper
import pdb
cropper = Cropper()

# Get a Numpy array of the cropped image
cropped_array = cropper.crop('/data/heyue/makeup_related/autocrop/tests/data/duncan.jpg')

# Save the cropped image with PIL if a face was detected:
# pdb.set_trace()
if cropped_array:
    print("ok")
    cropped_image = Image.fromarray(cropped_array)
    cropped_image.save('cropped.png')