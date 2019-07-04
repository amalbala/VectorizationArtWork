import numpy as np

def bincount_app(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)


def findinnercolor(image, contour):
    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)
    pts = np.where(mask == 255)
    colors = image[pts[0], pts[1]]
    return bincount_app(colors)


def savetosvg (image, contours, background):
    f = open('path.svg', 'w+')
    f.write('<svg width="' + str(image.shape[1])
            + '" height="' + str(image.shape[0])
            + '" xmlns="http://www.w3.org/2000/svg">')

    f.write('<rect width = "100%" height = "100%" fill="rgb('
            + str(background[2]) + ',' + str(background[1]) + ',' + str(background[0]) + ')" /> ')
    for contour in contours:
        f.write('<path d="M ')
        color = findinnercolor(image, contour)
        for i in range(len(contour)):
            x, y = contour[i][0]
            f.write(str(x) + ' ' + str(y) + ' ')
        f.write('Z" fill ="rgb(' + str(color[2]) + ',' + str(color[1]) + ',' + str(color[0]) + ')"')
        f.write('/>')
    f.write('</svg>')
    f.close()