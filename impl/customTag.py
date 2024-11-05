import numpy as np
import cv2
from random import random, randint

def id_to_bits(id):
    txt_file = '../codebook/apriltagCustom52h12_codebook_orig.txt'
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
    Rotate a 1D array of 52 border points counterclockwise by 90 degrees.
    """
    # Split the 1D array into the respective parts of the border (for 14x14 grid):
    top_row = border_points[:14]               # First 14 elements (top row)
    right_column = border_points[14:26]        # Next 12 elements (right column)
    bottom_row = border_points[26:40]          # Next 14 elements (bottom row, reversed)
    left_column = border_points[40:52]         # Last 12 elements (left column, reversed)

    # Rotate the points: top -> left, left -> bottom, bottom -> right, right -> top
    new_top_row = right_column                 # Right column becomes the new top row
    new_right_column = bottom_row[::-1]        # Bottom row becomes the new right column
    new_bottom_row = left_column               # Left column becomes the new bottom row
    new_left_column = top_row[::-1]            # Top row becomes the new left column

    # Concatenate the rotated border points back into a 1D array
    rotated_border_points = np.concatenate([new_top_row, new_right_column, new_bottom_row, new_left_column])

    return rotated_border_points


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


