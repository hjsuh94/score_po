import examples.simple_1d.distance_likelihood as mut

from hydra import initialize, compose

class TestDistanceLikelihood:
    def test_script(self):
        with initialize(config_path="./config"):
            cfg = compose(config_name="distance_likelihood")
            mut.main(cfg)
