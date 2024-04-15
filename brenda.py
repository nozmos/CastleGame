'''
------------------------
| Bash Render (BRenda) |
------------------------

Can binarise sets of images to be displayed on BASH terminal.
'''
import numpy as np
import os
import pathlib
from glob import glob
from PIL import Image

os.system("")

_STOP = False

_w, _h = 50, 30

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
            image = image.resize((_w, _h)).convert("L")
            image_arr = np.array(image, dtype="uint8")
            
        for row in image_arr:
            row_str = "".join(rgb_to_bash_bg((g, g, g), " ") for g in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
            framedata += bytearray(row_str, 'utf-8')

        return framedata + bytes(ANSI_CURSORUP(_h+1), "utf-8")

    @classmethod
    def generate_from_image_c(cls, imgpath: pathlib.Path | str) -> bytes:
        framedata = bytearray("", "utf-8")

        with Image.open(imgpath) as image:
            image = image.resize((_w, _h))
            image_arr = np.array(image, dtype="uint8")
            
        for row in image_arr:
            row_str = "".join(rgb_to_bash_bg(rgb, " ") for rgb in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
            framedata += bytearray(row_str, 'utf-8')

        return framedata + bytes(ANSI_CURSORUP(_h+1), "utf-8")

def generate_framedata(
        imgpath: pathlib.Path | str
    ) -> bytearray:

    framedata = bytearray("", "utf-8")

    with Image.open(imgpath) as image:
        image = image.resize((_w, _h)).convert("L")
        image_arr = np.array(image, dtype="uint8")
        
    for row in image_arr:
        row_str = "".join(rgb_to_bash_bg((g, g, g), " ") for g in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
        framedata += bytearray(row_str, 'utf-8')

    return framedata + bytes(ANSI_CURSORUP(_h+1), "utf-8")

def generate_framedata_c(
        imgpath: pathlib.Path | str
    ) -> bytearray:

    framedata = bytearray("", "utf-8")

    with Image.open(imgpath) as image:
        image = image.resize((_w, _h))
        image_arr = np.array(image, dtype="uint8")
        
    for row in image_arr:
        row_str = "".join(rgb_to_bash_bg(rgb, " ") for rgb in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
        framedata += bytearray(row_str, 'utf-8')

    return framedata + bytes(ANSI_CURSORUP(_h+1), "utf-8")

frame = generate_framedata("C:\\Users\\Me\\Downloads\\testf\\frame_01_delay-0.1s.jpg")
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
