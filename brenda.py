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
import sys
from PIL import Image, ImageDraw
from pynput import keyboard

np.set_printoptions(threshold=sys.maxsize)

STOP = False

SCREEN_W, SCREEN_H = 64, 48
HW_STRETCH = 1.5

ANSI_NEWLINE = "\r\n"
ANSI_ESC = "\033"
ANSI_CURSORUP = lambda n: f"{ANSI_ESC}[{n}A"
ANSI_HIDECURSOR = f"{ANSI_ESC}[?25l"
ANSI_SHOWCURSOR = f"{ANSI_ESC}[?25h"

CELL_SIZE = 64
RENDER_DISTANCE_GRID = 5
RENDER_DISTANCE_WORLD = float(RENDER_DISTANCE_GRID * CELL_SIZE)

FOV: float = np.pi / 3.0

MOVE_SPEED: float = CELL_SIZE / 4.0
TURN_SPEED: float = np.pi / 30.0

MIN_COLLISION_DISTANCE = CELL_SIZE / 6.0


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


def worldmap_to_image(worldmap: np.array) -> Image.Image:
    img_h, img_w = worldmap.shape
    img = Image.new("RGB", (img_w, img_h))
    pixels = img.load()
    for i in range(img_w):
        for j in range(img_h):
            if worldmap[j][i] == 1:
                pixels[i, j] = (255, 255, 255)
            else:
                pixels[i, j] = (0, 0, 0)
    return img


def grid_to_world_coords(grid_x: int, grid_y: int) -> tuple:
    return CELL_SIZE * (grid_x + 0.5), CELL_SIZE * (grid_y + 0.5)


def world_to_grid_coords(world_x: int, world_y: int) -> tuple:
    return math.floor(world_x / CELL_SIZE), math.floor(world_y / CELL_SIZE)


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
            image = image.resize((int(SCREEN_W * HW_STRETCH), SCREEN_H)).convert("L")

            image_arr = np.array(image, dtype="uint8")
            
        for row in image_arr:
            row_str = "".join(rgb_to_bash_bg((g, g, g), " ") for g in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
            framedata += bytearray(row_str, 'utf-8')

        return framedata# + bytes(ANSI_CURSORUP(SCREEN_H + 1), "utf-8")

    @classmethod
    def generate_from_image_path_c(cls, imgpath: pathlib.Path | str) -> bytes:
        global SCREEN_W, SCREEN_H
        framedata = bytearray("", "utf-8")

        with Image.open(imgpath) as image:
            # bash chars are not square so image must be streched to keep visual proportion
            image = image.resize((int(SCREEN_W * HW_STRETCH), SCREEN_H))
            image_arr = np.array(image, dtype="uint8")
            
        for row in image_arr:
            row_str = "".join(rgb_to_bash_bg(rgb, " ") for rgb in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
            framedata += bytearray(row_str, 'utf-8')

        return framedata# + bytes(ANSI_CURSORUP(SCREEN_H + 1), "utf-8")
    
    @classmethod
    def generate_from_image(cls, img: Image.Image) -> bytearray:
        global SCREEN_W, SCREEN_H
        framedata = bytearray("", "utf-8")
        image_arr = np.array( img.copy().resize((int(SCREEN_W * HW_STRETCH), SCREEN_H)).convert("L"), dtype="uint8" )
            
        for row in image_arr:
            row_str = "".join(rgb_to_bash_bg((v, v, v), " ") for v in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
            framedata += bytearray(row_str, 'utf-8')

        return framedata# + bytes(ANSI_CURSORUP(SCREEN_H + 1), "utf-8")
    
    @classmethod
    def generate_from_image_c(cls, img: Image.Image) -> bytearray:
        global SCREEN_W, SCREEN_H
        framedata = bytearray("", "utf-8")
        image_arr = np.array( img.copy().resize((int(SCREEN_W * HW_STRETCH), SCREEN_H)), dtype="uint8" )
            
        for row in image_arr:
            row_str = "".join(rgb_to_bash_bg(rgb, " ") for rgb in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
            framedata += bytearray(row_str, 'utf-8')

        return framedata# + bytes(ANSI_CURSORUP(SCREEN_H + 1), "utf-8")

#### DEPRECATED ####

# def generate_rect(x: int, y: int, size: tuple[int, int]) -> bytearray:
#     '''Generates framedata for a rect of given size at top-left coordinates (x, y)'''
#     global SCREEN_W, SCREEN_H
#     framedata = bytearray("", "utf-8")

#     canvas = np.zeros((SCREEN_W, SCREEN_H), dtype="uint8")
#     rect = np.ones(size, dtype="uint8") * 255
#     canvas[x:x+rect.shape[0], y:y+rect.shape[1]] = rect
#     canvas = canvas.transpose()
            
#     for row in canvas:
#         row_str = "".join(rgb_to_bash_bg((g, g, g), " ") for g in row) + f'{ANSI_ESC}[0m{ANSI_NEWLINE}'
#         framedata += bytearray(row_str, 'utf-8')

#     return framedata + bytes(ANSI_CURSORUP(SCREEN_H + 1), "utf-8")


# def shrink_image(source: Image.Image, new_size: tuple, origin: tuple=(0,0), mode: str="RGB", bg_colour: tuple=(0,0,0)) -> Image.Image:
#     new_width, new_height = int(new_size[0]), int(new_size[1])
#     shrink_x, shrink_y = new_size[0] / source.size[0], new_size[1] / source.size[1]
#     shrunk_pixels = source.copy().resize((new_width, new_height)).load()
    
#     canvas = Image.new(mode, source.size, color=bg_colour)
#     canvas_pixels = canvas.load()

#     px = int( (1.0 - shrink_x) * float(origin[0]) ) 
#     py = int( (1.0 - shrink_y) * float(origin[1]) ) 

#     for i in range(px, min(px + new_width, source.size[0])):
#         for j in range(py, min(py + new_height, source.size[1])):
#             sx = i - px
#             sy = j - py
#             canvas_pixels[i, j] = shrunk_pixels[sx, sy]
#     return canvas


# def paste_image(source: Image.Image, target: Image.Image, xy: tuple) -> Image.Image:
#     source_copy, target_copy = source.copy(), target.copy()
#     source_map, target_map = source_copy.load(), target_copy.load()
#     tx, ty = xy
#     for i in range(tx, min(tx + source_copy.size[0], target_copy.size[0])):
#         for j in range(ty, min(ty + source_copy.size[1], target_copy.size[1])):
#             sx, sy = i - tx, j - ty
#             target_map[i, j] = source_map[sx, sy]
#     return target_copy

#######################

def cellmap_to_worldmap(cellmap: list[str]) -> np.array:
    return np.concatenate(      # concatenate rows into full worldmap
        [
            np.concatenate(     # concatenate expanded cells into rows
                [
                    np.array([  # expand single cells into (CELLSIZE x CELLSIZE) arrays
                        [ int(cell) for _ in range(CELL_SIZE) ]
                        for _ in range(CELL_SIZE)
                    ]) for cell in cellrow
                ], axis=1
            ) for cellrow in cellmap
        ], axis=0
    )


def raycast(
        worldmap: np.array,
        obj_xy: tuple,
        angle: float,
        raycast_step: float = 1.0,
    ) -> float:
    '''Returns distance to wall in given direction'''

    obj_x, obj_y = obj_xy
    unit_x, unit_y = math.cos(angle), math.sin(angle)
    world_h, world_w = worldmap.shape
    
    distance_to_wall = 0.0

    while distance_to_wall < RENDER_DISTANCE_WORLD:
        distance_to_wall += raycast_step

        test_world_x = obj_x + unit_x * distance_to_wall
        test_world_y = obj_y + unit_y * distance_to_wall
        
        if test_world_x < 0 or test_world_x > world_h or test_world_y < 0 or test_world_y > world_w:
            distance_to_wall = RENDER_DISTANCE_WORLD
            break
        elif worldmap[math.floor(test_world_y), math.floor(test_world_x)] == 1:
            break

    return distance_to_wall


def worldmap_to_raycast_image(
        worldmap: np.array,
        camera_world_xy: tuple,
        view_angle: float,
        viewport_width: int=SCREEN_W,
        viewport_height: int=SCREEN_H,
        fov: float=FOV,
        raycast_step: float=1.0,
        birds_eye: bool=False,
    ) -> Image.Image:
    '''Generate perspective image of worldmap at given camera position'''
    
    global RENDER_DISTANCE_WORLD
    if birds_eye:
        raycast_view = worldmap_to_image(worldmap)
        draw = ImageDraw.Draw(raycast_view)
    
    camera_world_x, camera_world_y = camera_world_xy
    player_view = Image.new("RGB", (viewport_width, viewport_height))
    player_view_pixels = player_view.load()
    
    distance_to_viewport = viewport_width / (2.0 * math.tan(fov / 2.0))

    world_h, world_w = worldmap.shape

    # cast rays to find walls
    for screen_x in range(viewport_width):
        ray_angle = view_angle - math.atan( math.tan(fov / 2.0) * (1.0 - 2.0 * screen_x / viewport_width) )
        dev_angle = abs(ray_angle - view_angle) # angular deviation from view_angle
        unit_x, unit_y = math.cos(ray_angle), math.sin(ray_angle)
        
        distance_to_wall = raycast(worldmap, camera_world_xy, ray_angle, raycast_step)

        if birds_eye:
            # draw raycast in red
            draw.line((
                camera_world_x,
                camera_world_y,
                camera_world_x + unit_x * distance_to_wall,
                camera_world_y + unit_y * distance_to_wall
            ), fill=(255,0,0))

        else:
            perp_view_to_wall_dist = distance_to_wall * math.cos(dev_angle) - distance_to_viewport
            try:
                wall_height = min(viewport_height, viewport_height * CELL_SIZE / (2 * perp_view_to_wall_dist * math.tan(fov / 2.0)))
            except ZeroDivisionError:
                wall_height = viewport_height
            ceiling = 0.5 * (float(viewport_height) - wall_height)
            floor = viewport_height - ceiling

            for screen_y in range(viewport_height):
                if screen_y < ceiling:
                    player_view_pixels[screen_x, screen_y] = (0, 0, 0)
                elif screen_y < floor:
                    v = max( 0, int(255 * (1.0 - distance_to_wall / RENDER_DISTANCE_WORLD)) )
                    player_view_pixels[screen_x, screen_y] = (v, v, v)
                else:
                    player_view_pixels[screen_x, screen_y] = (50, 50, 50)
    
    if birds_eye:
        # draw player in green
        draw.ellipse((
            camera_world_x - MIN_COLLISION_DISTANCE,
            camera_world_y - MIN_COLLISION_DISTANCE,
            camera_world_x + MIN_COLLISION_DISTANCE,
            camera_world_y + MIN_COLLISION_DISTANCE,
        ), fill=(0,255,0))
    return raycast_view if birds_eye else player_view


def nearest_wall(
        worldmap: np.array,
        obj_xy: tuple,
        raycast_step: float = 1.0,
    ) -> float:
    '''Returns absolute distance and angle to nearest wall'''
    global TURN_SPEED
    obj_x, obj_y = obj_xy
    distance, theta = 0.0, 0.0

    N_RAYS = int(2 * np.pi / TURN_SPEED)
    
    while distance < RENDER_DISTANCE_WORLD:
        distance += raycast_step
        
        for i in range(N_RAYS):
            theta = i * TURN_SPEED
            unit_x, unit_y = math.cos(theta), math.sin(theta)

            test_x = obj_x + unit_x * distance
            test_y = obj_y + unit_y * distance

            if worldmap[math.floor(test_y), math.floor(test_x)] == 1:
                return distance, theta
    
    return distance, theta % 2*np.pi


def collide(
        worldmap: np.array,
        obj_xy: tuple,
        angle: float,
        speed: float = MOVE_SPEED,
        raycast_step: float = 1.0,
        min_collision: float = MIN_COLLISION_DISTANCE
    ) -> tuple:
    '''Return max step vector that does not collide obj with wall'''

    obj_x, obj_y = obj_xy
    unit_x, unit_y = math.cos(angle), math.sin(angle)
    
    n = 0
    while True:
        # find theoretical pos
        new_obj_x = obj_x + unit_x * (speed - raycast_step * n)
        new_obj_y = obj_y + unit_y * (speed - raycast_step * n)

        # if nearest wall is further than min_collision AFTER step, return
        wall_dist, _ = nearest_wall(worldmap, (new_obj_x, new_obj_y), raycast_step)

        if wall_dist > min_collision:
            break

        # otherwise step back by raycast_step until wall_dist is allowable
        n += 1

    return new_obj_x, new_obj_y


def collide_and_slide(
        worldmap: np.array,
        obj_xy: tuple,
        angle: float,
        speed: float = MOVE_SPEED,
        raycast_step: float = 1.0,
        min_collision: float = MIN_COLLISION_DISTANCE
    ) -> tuple:
    '''
    Return max step vector to nearest wall, plus correction vector for sliding along wall.
    If no wall encountered within step_size + MIN_COLLISION_DISTANCE, return xy
    '''

    obj_x, obj_y = obj_xy
    unit_x, unit_y = math.cos(angle), math.sin(angle)

    new_obj_x = obj_x + unit_x * speed
    new_obj_y = obj_y + unit_y * speed

    check_i, check_j = math.floor(new_obj_y), math.floor(new_obj_x)

    # if step does not collide, return
    if worldmap[check_i, check_j] == 0:
        return new_obj_x, new_obj_y

    while True:
        # otherwise, get forward distance to wall
        distance_to_wall = min_collision
        hit_wall = False

        while (not hit_wall) and distance_to_wall < speed:
            distance_to_wall += raycast_step

            test_world_x = obj_x + unit_x * distance_to_wall
            test_world_y = obj_y + unit_y * distance_to_wall
            
            if worldmap[math.floor(test_world_y), math.floor(test_world_x)] == 1:
                hit_wall = True
        
        # find max nearest neighbours in worldmap via overstep (speed - distance_to_wall)
        overstep = math.floor(speed - distance_to_wall)
        check_i, check_j = math.floor(test_world_y), math.floor(test_world_x)
        
        # neighbours should always cojntain a mixture of 0 and 1 (empty/wall)
        m = int(min_collision)
        neighbours = worldmap[(check_i-m):(check_i+m), (check_j-m):(check_j+m)]
        
        # neighbours[m, m] = 2
        x_diff = np.sum(np.ravel(neighbours[0:2*m,0:m])) - np.sum(np.ravel(neighbours[0:2*m,m:2*m]))
        y_diff = np.sum(np.ravel(neighbours[0:m,m:2*m])) - np.sum(np.ravel(neighbours[m:2*m,m:2*m]))
        length = (x_diff**2 + y_diff**2)**0.5

        return test_world_x + overstep * x_diff / length, test_world_y + overstep * y_diff / length

####################################

example_cellmap = [
    "1111111",
    "1000001",
    "1011101",
    "1000101",
    "1011101",
    "1000001",
    "1111111",
]

example_worldmap = cellmap_to_worldmap(example_cellmap)

_grid_w, _grid_h = get_grid_size(example_cellmap)
_world_w, _world_h = CELL_SIZE * _grid_w, CELL_SIZE * _grid_h

_player_grid_x: int = 1
_player_grid_y: int = 1
_player_world_x: float = (_player_grid_x + 0.5) * float(CELL_SIZE)
_player_world_y: float = (_player_grid_y + 0.5) * float(CELL_SIZE)


@dataclass
class State:
    player_angle: float = 0.0
    player_world_x: float = (1 + 0.5) * float(CELL_SIZE)
    player_world_y: float = (1 + 0.5) * float(CELL_SIZE)


def update(prev_state: State, delta_time: int, keys: set[keyboard.Key]):
    # TODO: take into account delta time
    new_player_angle = prev_state.player_angle
    new_player_world_x = prev_state.player_world_x
    new_player_world_y = prev_state.player_world_y

    try:

        if any(key.char == "a" for key in keys):
            new_player_angle = (new_player_angle - TURN_SPEED) % (2.0 * np.pi)
        if any(key.char == "d" for key in keys):
            new_player_angle = (new_player_angle + TURN_SPEED) % (2.0 * np.pi)

        if any(key.char == "w" for key in keys):
            new_player_world_x, new_player_world_y = collide(
                worldmap = example_worldmap,
                obj_xy = (new_player_world_x, new_player_world_y),
                angle = new_player_angle,
                speed = MOVE_SPEED,
            )
            # new_player_world_x += MOVE_SPEED * math.cos(new_player_angle)
            # new_player_world_y += MOVE_SPEED * math.sin(new_player_angle)
        if any(key.char == "s" for key in keys):
            new_player_world_x, new_player_world_y = collide(
                worldmap = example_worldmap,
                obj_xy = (new_player_world_x, new_player_world_y),
                angle = np.pi + new_player_angle,
                speed = MOVE_SPEED,
            )
            # new_player_world_x -= MOVE_SPEED * math.cos(new_player_angle)
            # new_player_world_y -= MOVE_SPEED * math.sin(new_player_angle)
    
    except AttributeError:
        pass

    return State(
        player_angle=new_player_angle,
        player_world_x=new_player_world_x,
        player_world_y=new_player_world_y,
    )


def render(term: blessed.Terminal, state: State, height: int, width: int):
    player_world_xy = (state.player_world_x, state.player_world_y)
    viewport_image = worldmap_to_raycast_image(
        example_worldmap, player_world_xy, state.player_angle,
        viewport_width=48, viewport_height=48, raycast_step=1.0, #birds_eye=True
    )
    frame = FrameData.generate_from_image_c(viewport_image)
    print(term.move_xy(0, 0) + frame.decode())
    # print(term.move_xy(0, 0) + str(time.time()))
    print(term.move_xy(0, 0) + str(np.rad2deg(state.player_angle)))


def main(term: blessed.Terminal):
    # assert term.number_of_colors == 1 << 24

    with term.raw(), term.hidden_cursor(), term.fullscreen(), keyboard.Events() as event_provider:

        state = State()
        keys_down = set()
        t = time.time()

        # Game Loop
        while True:
            while (event := event_provider.get(0)) is not None:
                # assumes events come in time order
                if isinstance(event, keyboard.Events.Press):
                    keys_down.add(event.key)
                elif isinstance(event, keyboard.Events.Release):
                    keys_down.remove(event.key)
            
            state = update(state, t - time.time(), keys_down)
            render(term, state, SCREEN_H, SCREEN_W)

            if keyboard.Key.esc in keys_down:
                break


if __name__ == "__main__":
    exit(main(blessed.Terminal()))