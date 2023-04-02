from PIL import Image
import cv2 as cv
import os

def PNG_JPG(PngPath):
    img = cv.imread(PngPath, 0)
    w, h = img.shape[::-1]
    infile = PngPath
    outfile = os.path.splitext(infile)[0] + ".jpg"
    img = Image.open(infile)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile)
            os.remove(PngPath)
        else:
            img.convert('RGB').save(outfile)
            os.remove(PngPath)
        return outfile
    except Exception as e:
        print("PNG转换JPG 错误", e)


if __name__ == '__main__':
    path1 = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-publicdata/crop/before"
    #path2 = "/data/hanxinyang/MuNeRF_latest/process_data_hy/makeup-transfer-publicdata/crop/before"
    filelist = os.listdir(path1)
    for file in filelist:
        blend = os.path.join(path1, file)
        PNG_JPG(blend)
