from unittest import TestCase

import time
import numpy as np
from numpy import testing

from quick_parallel import Sort


class QuickParallelTest(TestCase):
    def test_is_quick_sorting_correctly(self):
        for i in range(1, 1001, 100):
            x = np.random.rand(i)
            sorted_x = Sort(x).quick()

            testing.assert_array_equal(sorted_x, np.sort(x))

    def test_is_bubble_sorting_correctly(self):
        for i in range(1, 1001, 100):
            x = np.random.rand(i)
            sorted_x = Sort(x).bubble()

            testing.assert_array_equal(sorted_x, np.sort(x))

    def test_is_quick_parallel_sorting_correctly(self):
        for i in range(1, 1001, 100):
            x = np.random.rand(i)
            sorted_x = Sort(x, n_jobs=4).quick_parallel()

            testing.assert_array_equal(sorted_x, np.sort(x))
