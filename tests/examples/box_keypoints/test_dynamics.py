import examples.box_keypoints.box_pushing_system as mut
import numpy as np


class TestPlanarPushingDynamics:
    def test_coordinate_conversion(self):
        system = mut.PlanarPusherSystem()
        reduced_x = np.random.rand(3)
        full_x = np.array([1.0, 0.0, 0.0, 0.0, 0.1, 0.2, system.box_dim[2]])
        
        # Test results of converting back and forth.
        np.testing.assert_allclose(
            system.full_to_planar_coordinates(
                system.planar_to_full_coordinates(reduced_x)),
            reduced_x
        )
        
        np.testing.assert_allclose(
            system.planar_to_full_coordinates(
                system.full_to_planar_coordinates(full_x)),
                    full_x
                )
        
    def test_collision(self):
        system = mut.PlanarPusherSystem()
        np.testing.assert_equal(
            system.is_in_collision(np.zeros(5)), True
        )
        np.testing.assert_equal(
            system.is_in_collision(np.array([0.0, 0.0, 0.0, 0.0, 0.1])), True)
        np.testing.assert_equal(
            system.is_in_collision(np.array([0.0, 0.0, 0.0, 0.1, 0.0])), False)
        
    def test_dynamics(self):
        system = mut.PlanarPusherSystem()
        x = np.array([0.0, 0.0, 0.0, 0.1, 0.0])
        u = np.array([-0.1, 0.0])
        xnext = system.dynamics(x, u)
        np.testing.assert_equal(xnext.shape, (5,))
        assert(xnext[0] < 0.0)
        
        x_batch = np.tile(x, (3, 1))
        u_batch = np.tile(u, (3, 1))
        xnext_batch = system.dynamics_batch(x_batch, u_batch)
        np.testing.assert_equal(xnext_batch.shape, (3, 5))
        
    def test_keypoints(self):
        system = mut.PlanarPusherSystem()
        x = np.random.rand(3)
        keypts = system.get_keypoints(x)
        np.testing.assert_equal(keypts.shape, (2, 5))
        keypts = system.get_keypoints(x, noise_std=0.01)
        np.testing.assert_equal(keypts.shape, (2, 5))

    def test_within_table(self):
        system = mut.PlanarPusherSystem()
        np.testing.assert_equal(
            system.within_table(np.array([0.0, 1.0, 0.0])),
            False)
        np.testing.assert_equal(
            system.within_table(np.array([0.0, 0.5, 0.0])),
            True)
