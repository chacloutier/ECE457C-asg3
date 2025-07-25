from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False

class HighwayEnvBetter(HighwayEnvFast):
    """
    A variant of highway-v0 with more challenging, interesting behaviour:
        - continuous action but with constraints
        - remove reward for right lane
        - add lane change reward
        - add offroad terminal condition
        - constant reward for being on the road
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 5,
            "duration": 60,  # [s]
            "ego_spacing": 1.5,
            "right_lane_reward": 0,  # No reward for driving on the right-most lanes
            # "lane_change_reward": 0.1,   # The reward received at each lane change action.
            "reward_speed_range": [20, 40],
            # Base reward of 1 per timestep for being on the road and facing straight
            "on_road_reward": 0.5,  # Always reward for being on the road # THIS ISN't USED ACTUALLY
            "reward_heading_deviation": 0.5,  # Reward for driving straight, penalizing deviation in radians from the heading
            "offroad_terminal": True,
            "real_time_rendering": True,  # Enable real-time rendering for better visual feedback
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 40,
                "features": ["presence", "x", "y", "vx", "vy", "heading", "lat_off"],
                "absolute": False
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,  # Allow longitudinal control
                "acceleration_range": [-1, 1],  # Acceleration range for longitudinal control
                "steering_range": [-.1, .1],  # Steering range for lateral control
                "lateral": True,  # Allow lateral control
                "speed_range": [0, 40]  # Target speeds for longitudinal control
            }
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = 0
        if rewards['on_road_reward'] != 0:
            # If the vehicle is not on the road, the reward is zero.
            # Improved calculation for reward items that uses all nonzero ones and scales between the min and max from config
            reward_min_neg = 0; reward_max_pos = 0
            for name, reward in rewards.items():
                weight = self.config.get(name, 0)
                if weight < 0:
                    reward_min_neg += weight
                elif weight > 0:
                    reward_max_pos += weight
                reward += weight * reward

            if self.config["normalize_reward"]:
                reward = utils.lmap(reward,
                                    [reward_min_neg, reward_max_pos],
                                    [0, 1])
        return reward
    
    def _rewards(self, action: Action) -> Dict[Text, float]:
        #TODO: update reward function to reward driving straight
        # override _rewards function, call the parent one and return it
        rewards = super()._rewards(action)
        #add deviation from straight vehicle heading
        rewards["rewards_heading_deviation"] = -abs(float(self.vehicle.heading))

        # NOTE: current lane is self.vehicle.lane_index[2]
        # NOTE: neighbourlanes are self.vehicle.lane_index[0:2]

        return rewards


class HighwayEnvBetterv1_1(HighwayEnvBetter):
    """A variant of HighwayEnvBetter-v1 with harder oponnent vehicles and tighter spacing."""
    
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({ 
                "ego_spacing": 1,
                "vehicles_density": 1,
                "initial_lane_id": 2,  # Start in a different lane
                "vehicles_count": 10
                })
        return config

    # set "ego_spaceing" = 1.5 # to be tighter at start?
    # def _create_vehicles(self) -> None:
    #     """Create some new random vehicles of a given type, and add them on the road."""
    #     other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
    #     other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

    #     self.controlled_vehicles = []
    #     # This is our ego vehicle, which is controlled by the agent
    #     for others in other_per_controlled:
    #         vehicle = Vehicle.create_random(
    #             self.road,
    #             speed=29, # randomize speed a bit?
    #             lane_id=self.config["initial_lane_id"],
    #             spacing=self.config["ego_spacing"]
    #         )
    #         # move the car up a bit
    #         vehicle.position = vehicle.position * 1.2
    #         vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
    #         self.controlled_vehicles.append(vehicle)
    #         self.road.vehicles.append(vehicle)

    #         for _ in range(others):
    #             vehicle = other_vehicles_type.create_random(self.road, 
    #                                                         speed=30+np.rint([23, 26,31,36,25]), 
    #                                                         spacing=1 / self.config["vehicles_density"])
    #             #YAH fix this so it picks some nice random speeds
    #             vehicle.randomize_behavior()
    #             self.road.vehicles.append(vehicle)