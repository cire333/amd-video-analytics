import numpy as np, cv2, sys
def pattern(w, h, name):
    img = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    img[..., 0] = (xx * 255 // max(w - 1, 1)).astype(np.uint8)          # B ramps in x
    img[..., 1] = (yy * 255 // max(h - 1, 1)).astype(np.uint8)          # G ramps in y
    img[..., 2] = (((xx // 8 + yy // 8) % 2) * 255).astype(np.uint8)    # R 8px checker
    # solid corner squares to detect placement
    img[0:40, 0:40] = (255, 255, 255)
    img[0:40, w-40:w] = (0, 255, 255)
    img[h-40:h, 0:40] = (255, 0, 255)
    img[h-40:h, w-40:w] = (0, 0, 255)
    cv2.imwrite(name, img)
pattern(1280, 720, sys.argv[1] + "/pat_1280x720.png")
pattern(1920, 1080, sys.argv[1] + "/pat_1920x1080.png")
pattern(640, 480, sys.argv[1] + "/pat_640x480.png")
pattern(1000, 700, sys.argv[1] + "/pat_1000x700.png")
