'''
-------------------------------------
| Bash Rendering Algorithm (BRendA) |
-------------------------------------

Renders images to bash terminal using UTF-8 characters with ANSI sequences.
'''
import math
import numpy as np
import os
import pathlib
from PIL import Image, ImageDraw

os.system("")

STOP = False

SCREEN_W, SCREEN_H = 64, 64
HW_STRETCH = 2

ANSI_NEWLINE = "\012"
ANSI_ESC = "\033"
ANSI_CURSORUP = lambda n: f"{ANSI_ESC}[{n}A"
ANSI_HIDECURSOR = f"{ANSI_ESC}[?25l"
ANSI_SHOWCURSOR = f"{ANSI_ESC}[?25h"

CELL_SIZE = 64
RENDER_DISTANCE_GRID = 3
RENDER_DISTANCE_WORLD = float(RENDER_DISTANCE_GRID * CELL_SIZE)

CAMERA_DISTANCE_FACTOR: float = 16.0

FOV: float = np.pi
CAM_FOV = FOV / CAMERA_DISTANCE_FACTOR

with Image.open("C:\\Users\\Me\\Pictures\\pixelbricks.jpg") as img:
    WALL_TEXTURE = img.resize((CELL_SIZE, CELL_SIZE)).convert("RGB").load()

def rgb_to_bash_fg(rgb: tuple[int, int, int] | list[int, int, int], text: str) -> str:
    '''Returns `text` formatted for bash so that it prints with fg colour defined by `rgb`'''
    return '\033[38;2;%d;%d;%dm%s\033[0m' % (*rgb, text)

def rgb_to_bash_bg(rgb: tuple[int, int, int] | list[int, int, int], text: str) -> str:
    '''Returns `text` formatted for bash so that it prints with bg colour defined by `rgb`'''
    return '\033[48;2;%d;%d;%dm%s\033[0m' % (*rgb, text)

def cellmap_to_image(cellmap: list[str]) -> Image.Image:
    img_w = max(len(row) for row in cellmap)
    img_h = len(cellmap)
    img = Image.new("RGB", (img_w * CELL_SIZE, img_h * CELL_SIZE))
    pixels = img.load()
    for i in range(img.width):
        for j in range(img.height):
            if cellmap[i//CELL_SIZE][j//CELL_SIZE] == "1":
                pixels[i, j] = (255, 255, 255)
            else:
                pixels[i, j] = (0, 0, 0)
    return img

def grid_to_world_coords(grid_x: int, grid_y: int) -> tuple:
    return CELL_SIZE * (grid_x + 0.5), CELL_SIZE * (grid_y + 0.5)

def world_to_grid_coords(world_x: int, world_y: int) -> tuple:
    return int(world_x / CELL_SIZE), int(world_y / CELL_SIZE)

def get_grid_size(cellmap: list[str]) -> tuple:
    return CELL_SIZE * max(len(row) for row in cellmap), CELL_SIZE * len(cellmap)

class FrameData(bytes):

    def __init__(self, string: str, **kwargs) -> None:
        super().__init__(string, encoding="utf-8", **kwargs)

    @classmethod
    def generate_from_image_path(cls, imgpath: pathlib.Path | str) -> bytes:
        global SCREEN_W, SCREEN_H
        framedata = bytearray("", "utf-8")

        with Image.open(imgpath) as image:
            # bash chars are not square so image must be streched to keep visual proportion
            image = image.resize((SCREEN_W * HW_STRETCH, SCREEN_H)).convert("L")

            image_arr = np.array(image, dtype="uint8")
            
        for row in image_arr:
            row_str = "".join(rgb_to_bash_bg((g, g, g), " ") for g in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
            framedata += bytearray(row_str, 'utf-8')

        return framedata + bytes(ANSI_CURSORUP(SCREEN_H + 1), "utf-8")

    @classmethod
    def generate_from_image_path_c(cls, imgpath: pathlib.Path | str) -> bytes:
        global SCREEN_W, SCREEN_H
        framedata = bytearray("", "utf-8")

        with Image.open(imgpath) as image:
            # bash chars are not square so image must be streched to keep visual proportion
            image = image.resize((SCREEN_W * HW_STRETCH, SCREEN_H))
            image_arr = np.array(image, dtype="uint8")
            
        for row in image_arr:
            row_str = "".join(rgb_to_bash_bg(rgb, " ") for rgb in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
            framedata += bytearray(row_str, 'utf-8')

        return framedata + bytes(ANSI_CURSORUP(SCREEN_H + 1), "utf-8")
    
    @classmethod
    def generate_from_image(cls, img: Image.Image) -> bytearray:
        global SCREEN_W, SCREEN_H
        framedata = bytearray("", "utf-8")
        image_arr = np.array( img.copy().resize((SCREEN_W * HW_STRETCH, SCREEN_H)).convert("L"), dtype="uint8" )
            
        for row in image_arr:
            row_str = "".join(rgb_to_bash_bg((v, v, v), " ") for v in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
            framedata += bytearray(row_str, 'utf-8')

        return framedata + bytes(ANSI_CURSORUP(SCREEN_H + 1), "utf-8")
    
    @classmethod
    def generate_from_image_c(cls, img: Image.Image) -> bytearray:
        global SCREEN_W, SCREEN_H
        framedata = bytearray("", "utf-8")
        image_arr = np.array( img.copy().resize((SCREEN_W * HW_STRETCH, SCREEN_H)), dtype="uint8" )
            
        for row in image_arr:
            row_str = "".join(rgb_to_bash_bg(rgb, " ") for rgb in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
            framedata += bytearray(row_str, 'utf-8')

        return framedata + bytes(ANSI_CURSORUP(SCREEN_H + 1), "utf-8")

def generate_rect(x: int, y: int, size: tuple[int, int]) -> bytearray:
    '''Generates framedata for a rect of given size at top-left coordinates (x, y)'''
    global SCREEN_W, SCREEN_H
    framedata = bytearray("", "utf-8")

    canvas = np.zeros((SCREEN_W, SCREEN_H), dtype="uint8")
    rect = np.ones(size, dtype="uint8") * 255
    canvas[x:x+rect.shape[0], y:y+rect.shape[1]] = rect
    canvas = canvas.transpose()
            
    for row in canvas:
        row_str = "".join(rgb_to_bash_bg((g, g, g), " ") for g in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
        framedata += bytearray(row_str, 'utf-8')

    return framedata + bytes(ANSI_CURSORUP(SCREEN_H + 1), "utf-8")

####################################

example_worldmap = [
    "11111",
    "11011",
    "10001",
    "11011",
    "11111"
]

_grid_w, _grid_h = get_grid_size(example_worldmap)
_world_w, _world_h = CELL_SIZE * _grid_w, CELL_SIZE * _grid_h

_player_grid_x: int = 2
_player_grid_y: int = 2
_player_angle: float = 0.0

print(ANSI_HIDECURSOR, end="")

# Game Loop
while not STOP:
    
    try:

        world_img = cellmap_to_image(example_worldmap)
        world_img_pixels = world_img.load()
        draw = ImageDraw.Draw(world_img)

        player_view = Image.new("RGB", (SCREEN_W, SCREEN_H))
        player_view_pixels = player_view.load()

        player_image_x, player_image_y = grid_to_world_coords(_player_grid_x, _player_grid_y)
        player_unit_x, player_unit_y = math.cos(_player_angle), math.sin(_player_angle)
        camera_x = player_image_x - player_unit_x * float(SCREEN_W) / CAM_FOV
        camera_y = player_image_y - player_unit_y * float(SCREEN_W) / CAM_FOV

        # cast rays to find walls
        for screen_x in range(SCREEN_W):
            p = math.sin(0.5 * math.pi * float(screen_x) / float(SCREEN_W)) ** 2
            ray_angle = (_player_angle - (CAM_FOV / 2.0)) + (CAM_FOV * p)
            unit_x, unit_y = math.cos(ray_angle), math.sin(ray_angle)
            
            distance_to_wall = 0.0
            hit_wall = False

            while (not hit_wall) and distance_to_wall < RENDER_DISTANCE_WORLD:
                distance_to_wall += 0.1

                test_world_x = camera_x + unit_x * (float(SCREEN_W) / CAM_FOV + distance_to_wall)
                test_world_y = camera_y + unit_y * (float(SCREEN_W) / CAM_FOV + distance_to_wall)
                test_grid_x, test_grid_y = world_to_grid_coords(test_world_x, test_world_y)
                
                if test_grid_x < 0 or test_grid_x > _grid_w or test_grid_y < 0 or test_grid_y > _grid_h:
                    distance_to_wall = RENDER_DISTANCE_WORLD
                    hit_wall = True
                elif example_worldmap[test_grid_y][test_grid_x] == "1":
                    hit_wall = True

            draw.line((
                camera_x + unit_x * float(SCREEN_W) / CAM_FOV,
                camera_y + unit_y * float(SCREEN_W) / CAM_FOV,
                camera_x + unit_x * (float(SCREEN_W) / CAM_FOV + distance_to_wall),
                camera_y + unit_y * (float(SCREEN_W) / CAM_FOV + distance_to_wall)
            ), fill=(255,0,0))

            ceiling = float(SCREEN_H) * (1.0 - 1.0 / (1.0 + CAM_FOV * distance_to_wall / (SCREEN_W)) )
            floor = SCREEN_H - ceiling

            for screen_y in range(SCREEN_H):
                if screen_y < ceiling:
                    player_view_pixels[screen_x, screen_y] = (0, 0, 0)
                elif screen_y < floor:
                    v = max(0, int(255 * ((RENDER_DISTANCE_WORLD - distance_to_wall) / RENDER_DISTANCE_WORLD) ** 2.0) )
                    player_view_pixels[screen_x, screen_y] = (v, v, v)
                else:
                    player_view_pixels[screen_x, screen_y] = (50, 50, 50)

        # frame = FrameData.generate_from_image_c(world_img.resize((SCREEN_W, SCREEN_H)))
        frame = FrameData.generate_from_image_c(player_view)
        
        print(frame.decode())

        # os.system("bash -c 'read -rn1'")
    
    except KeyboardInterrupt:
        STOP = True
    
    _player_angle += 0.05

print(ANSI_NEWLINE * (SCREEN_H-1))
print(ANSI_SHOWCURSOR)