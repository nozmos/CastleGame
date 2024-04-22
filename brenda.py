'''
-------------------------------------
| Bash Rendering Algorithm (BRendA) |
-------------------------------------

Renders images to bash terminal using UTF-8 characters with ANSI sequences.
'''
from dataclasses import dataclass
import math
import time
import blessed
from blessed.keyboard import Keystroke
import numpy as np
import os
import pathlib
from PIL import Image, ImageDraw
from pynput import keyboard

STOP = False

SCREEN_W, SCREEN_H = 64, 48
HW_STRETCH = 2

ANSI_NEWLINE = "\r\n"
ANSI_ESC = "\033"
ANSI_CURSORUP = lambda n: f"{ANSI_ESC}[{n}A"
ANSI_HIDECURSOR = f"{ANSI_ESC}[?25l"
ANSI_SHOWCURSOR = f"{ANSI_ESC}[?25h"

CELL_SIZE = 48
RENDER_DISTANCE_GRID = 3
RENDER_DISTANCE_WORLD = float(RENDER_DISTANCE_GRID * CELL_SIZE)

CAMERA_DISTANCE_FACTOR: float = 10.0

FOV: float = np.pi
CAM_FOV = FOV / CAMERA_DISTANCE_FACTOR

class CompassDirection:
    NORTH = (0, -1)
    EAST = (1, 0)
    SOUTH = (0, 1)
    WEST = (-1, 0)

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

def shrink_image(source: Image.Image, new_size: tuple, origin: tuple=(0,0), mode: str="RGB", bg_colour: tuple=(0,0,0)) -> Image.Image:
    new_width, new_height = int(new_size[0]), int(new_size[1])
    shrink_x, shrink_y = new_size[0] / source.size[0], new_size[1] / source.size[1]
    shrunk_pixels = source.copy().resize((new_width, new_height)).load()
    
    canvas = Image.new(mode, source.size, color=bg_colour)
    canvas_pixels = canvas.load()

    px = int( (1.0 - shrink_x) * float(origin[0]) ) 
    py = int( (1.0 - shrink_y) * float(origin[1]) ) 

    for i in range(px, min(px + new_width, source.size[0])):
        for j in range(py, min(py + new_height, source.size[1])):
            sx = i - px
            sy = j - py
            canvas_pixels[i, j] = shrunk_pixels[sx, sy]
    return canvas

def paste_image(source: Image.Image, target: Image.Image, xy: tuple) -> Image.Image:
    source_copy, target_copy = source.copy(), target.copy()
    source_map, target_map = source_copy.load(), target_copy.load()
    tx, ty = xy
    for i in range(tx, min(tx + source_copy.size[0], target_copy.size[0])):
        for j in range(ty, min(ty + source_copy.size[1], target_copy.size[1])):
            sx, sy = i - tx, j - ty
            target_map[i, j] = source_map[sx, sy]
    return target_copy

def cellmap_to_raycast_image(
            cellmap: list[str],
            player_grid_xy: tuple,
            view_angle: float,
            viewport_width: int=SCREEN_W,
            viewport_height: int=SCREEN_H,
            fov: float=CAM_FOV,
            raycast_step: float=0.1,
            birds_eye: bool=False,
        ) -> Image.Image:
    global RENDER_DISTANCE_WORLD
    if birds_eye:
        raycast_view = cellmap_to_image(cellmap)
        draw = ImageDraw.Draw(raycast_view)
    
    player_grid_x, player_grid_y = player_grid_xy
    player_view = Image.new("RGB", (viewport_width, viewport_height))
    player_view_pixels = player_view.load()

    player_image_x, player_image_y = grid_to_world_coords(player_grid_x, player_grid_y)
    player_unit_x, player_unit_y = math.cos(view_angle), math.sin(view_angle)
    camera_x = player_image_x - player_unit_x * (float(viewport_width) / fov)
    camera_y = player_image_y - player_unit_y * (float(viewport_width) / fov)

    # cast rays to find walls
    for screen_x in range(viewport_width):
        p = math.sin(0.5 * math.pi * float(screen_x) / float(viewport_width)) ** 2
        ray_angle = (view_angle - (fov / 2.0)) + (fov * p)
        unit_x, unit_y = math.cos(ray_angle), math.sin(ray_angle)
        
        distance_to_wall = 0.0
        hit_wall = False

        while (not hit_wall) and distance_to_wall < RENDER_DISTANCE_WORLD:
            distance_to_wall += raycast_step

            test_world_x = camera_x + unit_x * (float(viewport_width) / fov + distance_to_wall)
            test_world_y = camera_y + unit_y * (float(viewport_width) / fov + distance_to_wall)
            test_grid_x, test_grid_y = world_to_grid_coords(test_world_x, test_world_y)
            
            if test_grid_x < 0 or test_grid_x > _grid_w or test_grid_y < 0 or test_grid_y > _grid_h:
                distance_to_wall = RENDER_DISTANCE_WORLD
                hit_wall = True
            elif cellmap[test_grid_y][test_grid_x] == "1":
                hit_wall = True

        if birds_eye:
            draw.line((
                camera_x + unit_x * (float(viewport_width) / fov),
                camera_y + unit_y * (float(viewport_width) / fov),
                camera_x + unit_x * (float(viewport_width) / fov + distance_to_wall),
                camera_y + unit_y * (float(viewport_width) / fov + distance_to_wall)
            ), fill=(255,0,0))

        else:
            ceiling = float(viewport_height) * (1.0 - 1.3 / (1.0 + fov * distance_to_wall / (viewport_width)) )
            floor = viewport_height - ceiling

            for screen_y in range(viewport_height):
                if screen_y < ceiling:
                    player_view_pixels[screen_x, screen_y] = (0, 0, 0)
                elif screen_y < floor:
                    v = max(0, int(255 * ((RENDER_DISTANCE_WORLD - distance_to_wall) / RENDER_DISTANCE_WORLD) ** 2.0) )
                    player_view_pixels[screen_x, screen_y] = (v, v, v)
                else:
                    player_view_pixels[screen_x, screen_y] = (50, 50, 50)
        
    return raycast_view if birds_eye else player_view



####################################

example_worldmap = [
    "11111",
    "10001",
    "11111"
]

_grid_w, _grid_h = get_grid_size(example_worldmap)
_world_w, _world_h = CELL_SIZE * _grid_w, CELL_SIZE * _grid_h

_player_grid_x: int = 1
_player_grid_y: int = 1

@dataclass
class State:
    player_angle: float = 0.0


def update(prev_state: State, delta_time: int, keys: set[keyboard.Key]):
    # TODO: take into account delta time
    new_player_angle = prev_state.player_angle
    if any(key.char == "a" for key in keys):
        new_player_angle -= 0.04
    if any(key.char == "d" for key in keys):
        new_player_angle += 0.04

    return State(
        player_angle=new_player_angle
    )


def render(term: blessed.Terminal, state: State, height: int, width: int):
    viewport_image = cellmap_to_raycast_image(
        example_worldmap, (_player_grid_x, _player_grid_y), state.player_angle,
        viewport_width=width, viewport_height=height
    )
    # image_arr = np.array(viewport_image.convert("L"), dtype="uint8")
    # for y, row in enumerate(image_arr):
    #     for x, colour in enumerate(row):
    frame = FrameData.generate_from_image(viewport_image)
    print(term.move_xy(0, 0) + frame.decode())
    print(term.move_xy(0, 0) + str(time.time()))


def main(term: blessed.Terminal):
    # assert term.number_of_colors == 1 << 24
    with term.raw(), term.hidden_cursor(), term.fullscreen(), keyboard.Events() as event_provider:
        state = State()
        # Game Loop
        keys_down = set()
        while True:
            while (event := event_provider.get(0)) is not None:
                # assumes events come in time order
                if isinstance(event, keyboard.Events.Press):
                    keys_down.add(event.key)
                elif isinstance(event, keyboard.Events.Release):
                    keys_down.remove(event.key)
            
            state = update(state, None, keys_down)
            render(term, state, SCREEN_H, SCREEN_W)
            if keyboard.Key.ctrl in keys_down and any(key.char == "c" for key in keys_down):
                break


if __name__ == "__main__":
    exit(main(blessed.Terminal()))