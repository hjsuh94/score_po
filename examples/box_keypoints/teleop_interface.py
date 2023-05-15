import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os, time

import hydra
from omegaconf import DictConfig

from hydra.utils import get_original_cwd

from pydrake.all import Quaternion
from pydrake.geometry import (
    MeshcatVisualizer,
    StartMeshcat,
    MeshcatPointCloudVisualizer,
    Rgba,
)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import (
    Parser,
    LoadModelDirectives,
    ProcessModelDirectives,
)
from pydrake.all import LeafSystem
from score_po.dynamical_system import DynamicalSystem

import pygame
from pygame.locals import *


class Teleop:
    def __init__(self):
        pygame.init()
        screen_size = 1
        self.screen = pygame.display.set_mode((screen_size, screen_size))

    def get_events(self):
        keys = pygame.key.get_pressed()
        events = dict()
        events["w"] = keys[K_w]
        events["a"] = keys[K_a]
        events["s"] = keys[K_s]
        events["d"] = keys[K_d]
        return events


class MouseKeyboardTeleop(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareVectorOuptutPort(
            "position", 2, self.CalcPositionOutput
        ).disable_caching_by_default()
