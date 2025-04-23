import numpy as np
import cv2
from random import random, randint

def id_to_bits(id):
    txt_file = '/home/suyeonchoi/Infrared_apriltag_detection/deeparuco/codebook/apriltagCustom52h12_codebook_orig.txt'
    result = 0
    with open(txt_file, 'r') as infile:
        for line in infile:
            row = line.strip().split(',')
            if int(row[0]) == id:
                binary_string = ''.join(map(str, row[1:53]))
                integer = int(binary_string, 2)
                result = [float(bit) for bit in format(integer, '052b')]  
    return result

def get_marker(id, size = 512, border_width = 1.0):

    marker = np.zeros((8, 8, 4), dtype=np.uint8)
    marker[0:8,0:8] = (0, 0, 0, 255.0)
    marker[1:7,1:7] = cv2.cvtColor(np.reshape(np.asarray([(i * 255.0) for i in id_to_bits(id)])
                                                .astype(np.uint8), (6, 6)), cv2.COLOR_GRAY2BGRA)
    
    canvas = np.ones((size, size, 4), dtype = np.uint8) * 255.0
    canvas[:,:,3] = 0

    center = size//2
    bg_size = int(size - 2 * (size / 10) * (1 - border_width))
    canvas[center - bg_size//2:center + bg_size//2, 
           center - bg_size//2:center + bg_size//2, 3] = 255.0
    
    wo_border = int(size - 2 * (size / 10))
    canvas[(size - wo_border) // 2:(size + wo_border) // 2,
        (size - wo_border) // 2:(size + wo_border) // 2] = \
            cv2.resize(marker, (wo_border, wo_border), interpolation=cv2.INTER_NEAREST)

    corners = [[(size - wo_border) // 2, (size - wo_border) // 2],
               [(size - wo_border) // 2, (size + wo_border) // 2 - 1],
               [(size + wo_border) // 2 - 1, (size + wo_border) // 2 - 1],
               [(size + wo_border) // 2 - 1, (size - wo_border) // 2]]

    return canvas, corners

def rotate_ccw(border_points):
    """
    Rotate a 1D array of 52 border points (14x14 grid, excluding corners)
    """
    assert len(border_points) == 52, "Expected 52 border points (13 per side, excluding corners)."

    # Extract the sides
    top    = border_points[0:13]           # left to right (excluding corners)
    right  = border_points[13:26]          # top to bottom (excluding corners)
    bottom = border_points[26:39]          # right to left (excluding corners, reversed)
    left   = border_points[39:52]          # bottom to top (excluding corners, reversed)

    # Rotate:
    new_top    = right                     # right becomes new top (same order)
    new_right  = bottom                    # bottom becomes new right (already reversed)
    new_bottom = left                      # left becomes new bottom (already reversed)
    new_left   = top                       # top becomes new left (must reverse)

    # Concatenate to form new border
    rotated = np.concatenate([new_top, new_right, new_bottom, new_left])
    return rotated


ids_as_bits = [id_to_bits(i) for i in range(250)]
def find_id(bits):

    rot0   = bits.flatten()
    rot90  = rotate_ccw(rot0)
    rot180 = rotate_ccw(rot90)
    rot270 = rotate_ccw(rot180)

    distances = [int(np.min([np.sum(np.abs(rot0 - check_bits)),
                np.sum(np.abs(rot90 - check_bits)),
                np.sum(np.abs(rot180 - check_bits)),
                np.sum(np.abs(rot270 - check_bits))])) 
                for check_bits in ids_as_bits]
    
    id = int(np.argmin(distances))

    return (id, distances[id])


if __name__ == '__main__':
    marker, corners = get_marker(randint(0, 250), border_width = random())
    for c in corners:
        cv2.circle(marker, (c[0], c[1]), 5, (0, 255, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.imwrite('test_customTag.png', marker)


