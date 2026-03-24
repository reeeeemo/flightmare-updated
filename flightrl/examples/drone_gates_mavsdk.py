from mavsdk import System
import asyncio
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.ppo import PPO
import numpy as np

class Drone_MAVSDK:
    """MavLINK compatible autonomous drone racer. 
    
    Uses a camera and provided telemetry to fly through gates.
    Attributes:
        addr: address for MAVSDK server
        drone: MAVSDK client for responding/sending messages
        model: AI weights that uses telemetry data to output drone actions
        normalizer: trained weights for normalizing observations/rewards
    """
    def __init__(self, addr: str, model_weights: str, normalizer_weights: str, device: str = "cpu"):
        self.addr = addr
        self.drone = System()
        self.model = PPO.load(path=model_weights, device=device)
        self.normalizer = VecNormalize.load(load_path=normalizer_weights, venv=None)
        self.normalizer.training = False
    
    async def connect(self) -> bool:
        """Connect to the MavLINK server"""
        await self.drone.connect(system_address=self.addr)
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Drone Connected!")
                await self.run()
                break
    
    async def run(self):
        """Perform a race with the drone."""

        # ensure drone recieves a heartbeat from the software first
        async for health in self.drone.telemetry.health():
            if health.is_armable:
                print("Drone is armed.")
                break

        print("running")

        while True:
            obs = await self.get_obs()
            action, _states = self.model.predict(self.normalizer.normalize_obs(obs))
            obs, rewards, dones, info = self.model.
    
    async def get_obs(self):
        """TODO: Gather telemetry data from drone."""
        obs = np.array()

    async def get_vid(self):
        """TODO: implement when vision module is implemented"""
        pass


if __name__ == "__main__":
    drone = Drone_MAVSDK("udpin://0.0.0.0:14540")
    asyncio.run(drone.connect())