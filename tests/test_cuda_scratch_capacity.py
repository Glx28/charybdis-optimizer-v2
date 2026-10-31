import unittest

import numpy as np

from fitness.cuda_kernel import _validate_cuda_scratch_dimensions


class CudaScratchCapacityTests(unittest.TestCase):
    def test_current_layout_dimensions_fit_scratch_caps(self):
        used = _validate_cuda_scratch_dimensions(
            pos_layer=np.array([0, 1, 10], dtype=np.int32),
            shortcut_access_target=np.array([-1, 1, 10], dtype=np.int32),
            shortcut_importance=np.zeros(307, dtype=np.float32),
            app_usage_weight=np.zeros(19, dtype=np.float32),
            group_matrix=np.zeros((79, 3), dtype=np.float32),
        )
        self.assertEqual(used["layers"], 11)

    def test_layer_overflow_fails_before_cuda_launch(self):
        with self.assertRaisesRegex(ValueError, "CUDA fitness scratch capacity exceeded"):
            _validate_cuda_scratch_dimensions(
                pos_layer=np.array([16], dtype=np.int32),
                shortcut_access_target=np.array([-1], dtype=np.int32),
                shortcut_importance=np.zeros(1, dtype=np.float32),
                app_usage_weight=np.zeros(1, dtype=np.float32),
                group_matrix=np.zeros((1, 3), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
