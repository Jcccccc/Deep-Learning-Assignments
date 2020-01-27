import numpy as np
import itertools as itt
import gym

from gym_minigrid.minigrid import WorldObj, MiniGridEnv, Lava, Grid, Wall, Goal
from gym_minigrid.envs.crossing import CrossingEnv


class SubgoalMiniGridEnv(MiniGridEnv):
    def __init__(self, *args, **kwargs):
        self.dense_reward = kwargs.pop('dense_reward') if 'dense_reward' in kwargs else False
        self.subgoal_reached = False
        self.subgoal_pos = None
        self.subgoal_reward = 0.
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self.subgoal_reached = False
        self.subgoal_pos = None
        super().reset(*args, **kwargs)
        
    def reward_fn(self):
        return 0

    def step(self, action):
        self.step_count += 1
        info = {'success': False}

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal'\
                    and (not self.subgoal_reached or not np.allclose(self.subgoal_pos, fwd_pos)):
                done = True if self.subgoal_reached else False
                info['success'] = True if self.subgoal_reached else False
                if done:
                    reward += self._reward()
                if not self.subgoal_reached:
                    reward += self.subgoal_reward
                self.subgoal_reached = True
                self.subgoal_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()
        
        if self.dense_reward:
            reward += self.reward_fn()

        return obs, reward, done, info

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)


class EuclidRewardMiniGridEnv(MiniGridEnv):
    def reward_fn(self):
        """Computes reward proportional to distance from target."""
        target_pos = np.asarray([self.grid.height - 2, self.grid.width - 2])
        dist = np.linalg.norm(self.agent_pos - target_pos)
        prev_dist = np.linalg.norm(self.prev_pos - target_pos)
        if dist < prev_dist:
            reward = 1
        elif dist > prev_dist:
            reward = -1
        else:
            reward = 0
        return reward / 50      # scale to bound cumulative reward scale

    def step(self, *args, **kwargs):
        self.prev_pos = np.asarray(self.agent_pos).copy()
        obs, reward, done, info = super().step(*args, **kwargs)
        reward += self.reward_fn()
        info['success'] = (self.agent_pos[0] == (self.grid.height - 2) and
                           self.agent_pos[1] == (self.grid.width - 2))
        return obs, reward, done, info


class OptRewardMiniGridEnv(SubgoalMiniGridEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, dense_reward=True)
        self.prev_dist = None
        
    def manhattan_distance(self, x, y):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.prev_dist = None

    def reward_fn(self):
        """Computes reward proportional to shortest path distance from target."""
        target_pos = np.asarray([self.grid.height - 2, self.grid.width - 2])            
            
        if (self.horizontal and self.agent_pos[0] > self.subgoal_pos[0]) \
            or (not self.horizontal and self.agent_pos[1] > self.subgoal_pos[1]):
            dist = self.manhattan_distance(self.agent_pos, target_pos)
        else:
            dist = self.manhattan_distance(self.agent_pos, self.subgoal_pos) + self.manhattan_distance(self.subgoal_pos, target_pos)

        if self.prev_dist is not None:
            reward = (self.prev_dist - dist) / 20
        else:
            reward = 0
        self.prev_dist = dist
        return reward      # scale to bound cumulative reward scale


class SubgoalCrossingEnv(SubgoalMiniGridEnv):
    """
    Environment with wall or lava obstacles, sparse reward.
    """

    def __init__(self, size=9, num_crossings=1, obstacle_type=Lava, seed=None):
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None
        )
        self.subgoal_reward = 0.2

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[:self.num_crossings]  # sample random rivers
        rivers_v = sorted([pos for direction, pos in rivers if direction is v])
        rivers_h = sorted([pos for direction, pos in rivers if direction is h])
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.put_obj(self.obstacle_type(), i, j)

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1]))
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1]))
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, None)
            self.put_obj(Goal(), i, j)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )


class EuclidRewardCrossingEnv(EuclidRewardMiniGridEnv):
    """
    Environment with wall or lava obstacles, dense reward.
    """

    def __init__(self, size=9, num_crossings=1, obstacle_type=Lava, seed=None):
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None
        )

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[:self.num_crossings]  # sample random rivers
        rivers_v = sorted([pos for direction, pos in rivers if direction is v])
        rivers_h = sorted([pos for direction, pos in rivers if direction is h])
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.put_obj(self.obstacle_type(), i, j)

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1]))
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1]))
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, None)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)


class OptRewardCrossingEnv(OptRewardMiniGridEnv):
    """
    Environment with wall or lava obstacles, sparse reward.
    """

    def __init__(self, size=9, num_crossings=1, obstacle_type=Lava, seed=None):
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None
        )

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[:self.num_crossings]  # sample random rivers
        rivers_v = sorted([pos for direction, pos in rivers if direction is v])
        rivers_h = sorted([pos for direction, pos in rivers if direction is h])
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.put_obj(self.obstacle_type(), i, j)

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1]))
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1]))
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, None)
            self.put_obj(Goal(), i, j)
            self.subgoal_pos = np.asarray([i, j])
            self.horizontal = (direction == h)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )


class SimpleSubgoalCrossingEnv(SubgoalCrossingEnv):
    def __init__(self):
        super().__init__(size=5, num_crossings=1, obstacle_type=Wall)


class SimpleEuclidRewardCrossingEnv(EuclidRewardCrossingEnv):
    def __init__(self):
        super().__init__(size=5, num_crossings=1, obstacle_type=Wall)


class SimpleOptRewardCrossingEnv(OptRewardCrossingEnv):
    def __init__(self):
        super().__init__(size=5, num_crossings=1, obstacle_type=Wall)


class HardSubgoalCrossingEnv(SubgoalCrossingEnv):
    def __init__(self):
        super().__init__(size=7, num_crossings=1, obstacle_type=Wall)


class HardEuclidRewardCrossingEnv(EuclidRewardCrossingEnv):
    def __init__(self):
        super().__init__(size=7, num_crossings=1, obstacle_type=Wall)


class HardOptRewardCrossingEnv(OptRewardCrossingEnv):
    def __init__(self):
        super().__init__(size=7, num_crossings=1, obstacle_type=Wall)


class SparseCrossingEnv(CrossingEnv):
    def __init__(self):
        super().__init__(size=7, num_crossings=1, obstacle_type=Wall)

    def step(self, *args, **kwargs):
        self.prev_pos = np.asarray(self.agent_pos).copy()
        obs, reward, done, info = super().step(*args, **kwargs)
        info['success'] = (self.agent_pos[0] == (self.grid.height - 2) and
                           self.agent_pos[1] == (self.grid.width - 2))
        return obs, reward, done, info


class DeterministicCrossingEnv(CrossingEnv):
    def __init__(self):
        super().__init__(size=7, num_crossings=1, obstacle_type=Wall)

    def _gen_grid(self, *args, **kwargs):
        self.seed(33)
        super()._gen_grid(*args, **kwargs)


class DetHardSubgoalCrossingEnv(SubgoalCrossingEnv):
    def __init__(self):
        super().__init__(size=7, num_crossings=1, obstacle_type=Wall)

    def _gen_grid(self, *args, **kwargs):
        self.seed(33)
        super()._gen_grid(*args, **kwargs)


class DetHardEuclidCrossingEnv(EuclidRewardCrossingEnv):
    def __init__(self):
        super().__init__(size=7, num_crossings=1, obstacle_type=Wall)

    def _gen_grid(self, *args, **kwargs):
        self.seed(33)
        super()._gen_grid(*args, **kwargs)


class DetHardOptRewardCrossingEnv(OptRewardCrossingEnv):
    def __init__(self):
        super().__init__(size=7, num_crossings=1, obstacle_type=Wall)

    def _gen_grid(self, *args, **kwargs):
        self.seed(33)
        super()._gen_grid(*args, **kwargs)
        
        
class LavaEnv(SubgoalCrossingEnv):
    def __init__(self):
        super().__init__(size=9, num_crossings=1, obstacle_type=Lava)
        
        
class WallEnv(SubgoalCrossingEnv):
    def __init__(self):
        super().__init__(size=9, num_crossings=1, obstacle_type=Wall)

