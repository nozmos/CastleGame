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

CELL_SIZE = 64
RENDER_DISTANCE_GRID = 5
RENDER_DISTANCE_WORLD = float(RENDER_DISTANCE_GRID * CELL_SIZE)

FOV: float = np.pi / 3.0

MOVE_SPEED: float = CELL_SIZE * 4.0
TURN_SPEED: float = np.pi / 1.5

MIN_COLLISION_DISTANCE = CELL_SIZE / 6.0

class CellCode:
    EMPTY = 0
    WALL = 1
    THING = 2

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


def image_to_term(img: Image.Image, text_overlay: list[str]) -> str:
    framedata = ""
    image_arr = np.array(img, dtype="uint8")
        
    for y, row in enumerate(image_arr):
        if y < len(text_overlay):
            text_line = text_overlay[y]
        else:
            text_line = ""
        row_str = "".join(rgb_to_bash_bg(rgb, rgb_to_bash_fg((255, 0, 0), text_line[x]) if x < len(text_line) else " ") for x, rgb in enumerate(row)) + "\n"
        framedata += row_str

    return framedata


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
        cell_code = worldmap[math.floor(test_world_y), math.floor(test_world_x)]
        
        if test_world_x < 0 or test_world_x > world_w or test_world_y < 0 or test_world_y > world_h:
            distance_to_wall = RENDER_DISTANCE_WORLD
            break
        elif cell_code != 0:
            break

    return distance_to_wall, cell_code


def worldmap_to_raycast_image(
        worldmap: np.array,
        camera_world_xy: tuple,
        view_angle: float,
        viewport_width: int,
        viewport_height: int,
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
        
        distance_to_wall, cell_code = raycast(worldmap, camera_world_xy, ray_angle, raycast_step)

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
                if wall_height < 0:
                    wall_height = viewport_height
            except ZeroDivisionError:
                wall_height = viewport_height
            ceiling = min( float(viewport_height) / 2.0, 0.5 * (float(viewport_height) - wall_height) )
            floor = viewport_height - ceiling

            for screen_y in range(viewport_height):
                if screen_y < ceiling:
                    v = max( 0, int(150 * (0.4 - screen_y / viewport_height)) )
                    player_view_pixels[screen_x, screen_y] = (v, v, v)
                elif screen_y < floor:
                    v = max( 0, int(255 * (1.0 - distance_to_wall / RENDER_DISTANCE_WORLD)) )
                    player_view_pixels[screen_x, screen_y] = (v, v, v)
                else:
                    v = max( 0, int(150 * (screen_y / viewport_height - 0.6)) )
                    player_view_pixels[screen_x, screen_y] = (v, v, v)
    
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

# FIXME
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
    "111111111",
    "100000001",
    "100000001",
    "100111001",
    "100001001",
    "100001001",
    "111111001",
    "100000001",
    "100000001",
    "100000001",
    "100000001",
    "111111111",
]

example_worldmap = cellmap_to_worldmap(example_cellmap)

@dataclass
class State:
    player_angle: float = 0.0
    player_world_x: float = (1 + 0.5) * float(CELL_SIZE)
    player_world_y: float = (1 + 0.5) * float(CELL_SIZE)
    delta_time: float = 0.0


def update(prev_state: State, delta_time: int, keys: set[keyboard.Key]):
    # TODO: take into account delta time
    new_player_angle = prev_state.player_angle
    new_player_world_x = prev_state.player_world_x
    new_player_world_y = prev_state.player_world_y

    try:

        if any(key.char == "a" for key in keys):
            new_player_angle = (new_player_angle - TURN_SPEED * delta_time) % (2.0 * np.pi)
        if any(key.char == "d" for key in keys):
            new_player_angle = (new_player_angle + TURN_SPEED * delta_time) % (2.0 * np.pi)

        if any(key.char == "w" for key in keys):
            new_player_world_x, new_player_world_y = collide(
                worldmap = example_worldmap,
                obj_xy = (new_player_world_x, new_player_world_y),
                angle = new_player_angle,
                speed = MOVE_SPEED * delta_time,
            )
        if any(key.char == "s" for key in keys):
            new_player_world_x, new_player_world_y = collide(
                worldmap = example_worldmap,
                obj_xy = (new_player_world_x, new_player_world_y),
                angle = np.pi + new_player_angle,
                speed = MOVE_SPEED * delta_time,
            )
    
    except AttributeError:
        pass

    return State(
        player_angle=new_player_angle,
        player_world_x=new_player_world_x,
        player_world_y=new_player_world_y,
        delta_time=delta_time,
    )


def render(term: blessed.Terminal, state: State, width: int, height: int):
    player_world_xy = (state.player_world_x, state.player_world_y)
    viewport_image = worldmap_to_raycast_image(
        example_worldmap, player_world_xy, state.player_angle,
        viewport_width=width, viewport_height=height,
        raycast_step=1.0, #birds_eye=True
    )
    text_overlay = []
    if state.delta_time != 0:
        text_overlay.append(f"FPS: {1/state.delta_time:.1f}")
    text_overlay.append(f"angle: {math.degrees(state.player_angle)}")
    frame = image_to_term(viewport_image, text_overlay)
    print(term.move_xy(0, 0) + frame)
    # print(term.move_xy(0, 0) + str(np.rad2deg(state.player_angle)))


def main(term: blessed.Terminal):
    with term.raw(), term.hidden_cursor(), term.fullscreen(), keyboard.Events() as event_provider:
        state = State()
        keys_down = set()
        delta_time = 0
        t0 = time.time()

        # Game Loop
        while True:
            t1 = time.time()
            delta_time = t1 - t0
            t0 = t1

            # consume keypresses so they don't get printed to terminal after the program exits
            while term.kbhit(0):
                term.getch()

            while (event := event_provider.get(0)) is not None:
                # assumes events come in time order
                if isinstance(event, keyboard.Events.Press):
                    keys_down.add(event.key)
                elif isinstance(event, keyboard.Events.Release):
                    keys_down.remove(event.key)
            state = update(state, delta_time, keys_down)
            render(term, state, 96, 48)


            if keyboard.Key.esc in keys_down:
                break


if __name__ == "__main__":
    exit(main(blessed.Terminal()))