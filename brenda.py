'''
------------------------
| Bash Render (BRenda) |
------------------------

Can binarise sets of images to be displayed on BASH terminal.
'''
import numpy as np
import os
import pathlib
from PIL import Image, ImageTransform

os.system("")

_STOP = False

_w, _h = 30, 30
_HW_STRETCH = 2

ANSI_NEWLINE = "\012"
ANSI_ESC = "\033"
ANSI_CURSORUP = lambda n: f"{ANSI_ESC}[{n}A"
ANSI_HIDECURSOR = f"{ANSI_ESC}[?25l"
ANSI_SHOWCURSOR = f"{ANSI_ESC}[?25h"

print(ANSI_HIDECURSOR)

def rgb_to_bash_fg(rgb: tuple[int, int, int] | list[int, int, int], text: str) -> str:
    '''Returns `text` formatted for bash so that it prints with fg colour defined by `rgb`'''
    r, g, b = rgb
    return '\033[38;2;%d;%d;%dm%s\033[0m' % (*rgb, text)

def rgb_to_bash_bg(rgb: tuple[int, int, int] | list[int, int, int], text: str) -> str:
    '''Returns `text` formatted for bash so that it prints with bg colour defined by `rgb`'''
    r, g, b = rgb
    return '\033[48;2;%d;%d;%dm%s\033[0m' % (*rgb, text)

class FrameData(bytes):

    def __init__(self, string: str, **kwargs) -> None:
        super().__init__(string, encoding="utf-8", **kwargs)

    @classmethod
    def generate_from_image(cls, imgpath: pathlib.Path | str) -> bytes:
        framedata = bytearray("", "utf-8")

        with Image.open(imgpath) as image:
            # bash chars are not square so image must be streched to keep visual proportion
            image = image.resize((_w * _HW_STRETCH, _h)).convert("L")
            image_arr = np.array(image, dtype="uint8")
            
        for row in image_arr:
            row_str = "".join(rgb_to_bash_bg((g, g, g), " ") for g in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
            framedata += bytearray(row_str, 'utf-8')

        return framedata + bytes(ANSI_CURSORUP(_h+1), "utf-8")

    @classmethod
    def generate_from_image_c(cls, imgpath: pathlib.Path | str) -> bytes:
        framedata = bytearray("", "utf-8")

        with Image.open(imgpath) as image:
            # bash chars are not square so image must be streched to keep visual proportion
            image = image.resize((_w * _HW_STRETCH, _h))
            image_arr = np.array(image, dtype="uint8")
            
        for row in image_arr:
            row_str = "".join(rgb_to_bash_bg(rgb, " ") for rgb in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
            framedata += bytearray(row_str, 'utf-8')

        return framedata + bytes(ANSI_CURSORUP(_h+1), "utf-8")
    
    @classmethod
    def generate_from_array(cls, a: np.array) -> bytes:
        framedata = bytearray("", "utf-8")
        image = Image.fromarray(a).resize((_w, _h)).convert("L")
        image_arr = np.array(image, dtype="uint8")

        for row in image_arr:
            row_str = "".join(rgb_to_bash_bg((g, g, g), " ") for g in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
            framedata += bytearray(row_str, 'utf-8')

        return framedata + bytes(ANSI_CURSORUP(_h+1), "utf-8")

def generate_rect(x: int, y: int, size: tuple[int, int]) -> bytearray:
    '''Generates framedata for a rect of given size at top-left coordinates (x, y)'''
    global _w, _h
    framedata = bytearray("", "utf-8")

    canvas = np.zeros((_w, _h), dtype="uint8")
    rect = np.ones(size, dtype="uint8") * 255
    canvas[x:x+rect.shape[0], y:y+rect.shape[1]] = rect
    canvas = canvas.transpose()
            
    for row in canvas:
        row_str = "".join(rgb_to_bash_bg((g, g, g), " ") for g in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
        framedata += bytearray(row_str, 'utf-8')

    return framedata + bytes(ANSI_CURSORUP(_h+1), "utf-8")



frame = FrameData.generate_from_image("C:\\Users\\Me\\Pictures\\pixelbricks.jpg")
frames = [frame]

while not _STOP:
    for frame in frames:
        print(frame.decode())

        try:
            os.system("bash -c 'read -rsn1'")
        
        except KeyboardInterrupt:
            _STOP = True
            break
    

print(ANSI_NEWLINE * _h)
