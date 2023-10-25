import math

import numpy as np
from gymnasium import spaces, utils

from miniworld.entity import Box, MeshEnt
from miniworld.miniworld import MiniWorldEnv

from typing import Optional, Tuple
from gymnasium.core import ObsType


from miniworld.entity import Agent

from pyglet.window import key
import pyglet


class Sidewalk(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Walk on a sidewalk up to an object to be collected. Don't walk into the
    street. The goal is to reach the object in as few steps as possible.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +(1 - 0.2 * (step_count / max_episode_steps)) when object reached

    ## Arguments

    ```python
    env = gym.make("MiniWorld-Sidewalk-v0")
    ```

    """

    def __init__(self, time_id, max_episode_time_step=250, rendering=False, wait_for_keypress=False, **kwargs):
        
        self.rendering = rendering
        
        # Select time
        if time_id == 0:
            self.loc_sky_color = [1.0, 0.75, 0.4]
            self.loc_light_amb = [1.0, 0.75, 0.4]
            self.loc_light_color = [1.0, 0.75, 0.4]
            self.wall_tex = "brick_wall"
            self.sidewalk_tex = "concrete_tiles"
            self.road_tex = "asphalt"

        if time_id == 1:
            self.loc_sky_color = [0.4, 0.75, 1.0]
            self.loc_light_amb = [0.4, 0.75, 1.0]
            self.loc_light_color = [0.4, 0.75, 1.0]
            self.wall_tex = "brick_wall"
            self.sidewalk_tex = "concrete_tiles"
            self.road_tex = "asphalt"

        if time_id == 2:
#             self.loc_sky_color = [0.4, 0.75, 0.4]
#             self.loc_light_amb = [0.4, 0.75, 0.4]
#             self.loc_light_color = [0.4, 0.75, 0.4]

            self.loc_sky_color = [1.0, 0.75, 1.0]
            self.loc_light_amb = [1.0, 0.75, 1.0]
            self.loc_light_color = [1.0, 0.75, 1.0]
            self.wall_tex = "brick_wall"
            self.sidewalk_tex = "concrete_tiles"
            self.road_tex = "asphalt"

        if time_id == 3:
            self.loc_sky_color = [0.0, 0.2, 0.4]
            self.loc_light_amb = [0.0, 0.2, 0.4]
            self.loc_light_color = [0.0, 0.2, 0.4]
            self.wall_tex = "brick_wall"
            self.sidewalk_tex = "concrete_tiles"
            self.road_tex = "asphalt"
        
        MiniWorldEnv.__init__(self, max_episode_steps=150, **kwargs)
        utils.EzPickle.__init__(self, **kwargs)            

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)
        
        # Create variables used in human interface
        self.stop_simulation = False  #  the stop_simulation flag will be set to True if user wants to interrupt the simulation
        self.key_pressed = {key.LEFT: False, key.RIGHT: False, key.UP: False}
        self.press_event = False
        self.wait_for_keypress = wait_for_keypress if self.rendering else False
        
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """
        super().reset(seed=seed)

        # Step count since episode start
        self.step_count = 0

        # Create the agent
        self.agent = Agent()

        # List of entities contained
        self.entities = []

        # List of rooms in the world
        self.rooms = []

        # Wall segments for collision detection
        # Shape is (N, 2, 3)
        self.wall_segs = []

        # Generate the world
        self._gen_world()

        # Check if domain randomization is enabled or not
        rand = self.np_random if self.domain_rand else None

        # Randomize elements of the world (domain randomization)
        self.params.sample_many(
            rand, self, ["sky_color", "light_pos", "light_color", "light_ambient"]
        )
        
        # NOTE: added
        self.sky_color = self.loc_sky_color
        self.light_ambient = self.loc_light_amb
        self.light_color = self.loc_light_color
#         print(self.light_pos)

        # Get the max forward step distance
        self.max_forward_step = self.params.get_max("forward_step")

        # Randomize parameters of the entities
        for ent in self.entities:
            ent.randomize(self.params, rand)

        # Compute the min and max x, z extents of the whole floorplan
        self.min_x = min(r.min_x for r in self.rooms)
        self.max_x = max(r.max_x for r in self.rooms)
        self.min_z = min(r.min_z for r in self.rooms)
        self.max_z = max(r.max_z for r in self.rooms)

        # Generate static data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # Pre-compile static parts of the environment into a display list
        self._render_static()

        # Generate the first camera image
        obs = self.render_obs()

        # Return first observation
        return obs, {}
    
        
    def reinit(self, max_steps, wait_for_keypress):
        self.wait_for_keypress = wait_for_keypress if self.rendering else False
        if self.rendering:
            # Init
            self.render_mode = 'pyglet'
            super().render()

            # Decorate key interfaces
            self.on_key_press = self.unwrapped.window.event(self.on_key_press)
            self.on_key_release = self.unwrapped.window.event(self.on_key_release)
            
            self.stop_simulation = False
            
            self.press_event = False
            for key, value in self.key_pressed.items():
                self.key_pressed[key] = False

        self.max_episode_steps = max_steps
        
        return self.reset()
    

    def _gen_world(self):
        sidewalk = self.add_rect_room(
            min_x=-3,
            max_x=0,
            min_z=0,
            max_z=12,
            wall_tex=self.wall_tex,
            floor_tex=self.sidewalk_tex,
            no_ceiling=True,
        )

        self.street = self.add_rect_room(
            min_x=0,
            max_x=6,
            min_z=-80,
            max_z=80,
            floor_tex=self.road_tex,
            no_ceiling=True,
        )

        self.connect_rooms(sidewalk, self.street, min_z=0, max_z=12)

        # Decorative building in the background
        self.place_entity(
            MeshEnt(mesh_name="building", height=30),
            pos=np.array([30, 0, 30]),
            dir=-math.pi,
        )

        for i in range(1, sidewalk.max_z // 2):
            self.place_entity(
                MeshEnt(mesh_name="cone", height=0.75), pos=np.array([1, 0, 2 * i])
            )

        self.box = self.place_entity(
            Box(color="red"),
            room=sidewalk,
            min_z=sidewalk.max_z - 2,
            max_z=sidewalk.max_z,
        )

        self.place_agent(room=sidewalk, min_z=0, max_z=1.5)

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        # Walking into the street ends the episode
        if self.street.point_inside(self.agent.pos):
            reward = 0
            termination = True

        if self.near(self.box):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info
    
    def render(self, mode='rgb_array'):
        if self.rendering:
            self.render_mode = 'pyglet'
            out = super().render()
#             out = super().render()#mode='pyglet')
            self.update_pyglet() # update pyglet info and capture keys
        else:
            self.render_mode = mode
            out = super().render()
#             out = super().render()#mode=mode)
#         print(out)
        return out

    def on_key_press(self, symbol, modifier):
        if symbol == key.ESCAPE: 
            self.stop_simulation = True  # set 'quit' flag if ESCAPE key is pressed
        self.key_pressed[symbol] = True
        self.press_event = True
    
    def on_key_release(self, symbol, modifier):
        self.key_pressed[symbol] = False
        self.press_event = False

    def update_pyglet(self):
        self.window.flip()
        while True:
            pyglet.clock.tick()
            self.window.dispatch_events()
            if (not self.wait_for_keypress) or self.press_event:
                break

    def close(self):
        super().close()
