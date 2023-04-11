import examples.simple_1d.admissibility_stability as mut

from hydra import initialize, compose

class TestAdmissibilityStability:
    def test_script(self):
        with initialize(config_path="./config"):
            cfg = compose(config_name="admissibility_stability")
            mut.main(cfg)
