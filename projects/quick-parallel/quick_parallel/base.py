import time
import random
import threading
import numpy as np


class QuickManager:
    def __init__(self, sort_manager, cv=None, phase=0, n_jobs=1):
        self.sort_manager = sort_manager
        self.cv = cv or threading.Condition()
        self.x  = self.sort_manager.x
        self.n_jobs    = n_jobs

        self.intervals = []
        self.probes    = []
        self.intervals_processed = 0

    def run(self):
        for i in range(self.n_jobs):
            probe = QuickProbe(self, i)
            probe.start()

            self.probes.append(probe)

        self.intervals.append((0, len(self.x) - 1))
        with self.cv:
            self.cv.notify()

        for probe in self.probes:
            probe.join()


class QuickProbe(threading.Thread):
    def __init__(self, manager, id):
        super().__init__()

        self.id = id
        self.manager = manager

    def quicksort(self, start, end):
        x = self.manager.x

        # Author: Sue Sentance.
        pivot = x[start]
        left = start + 1
        right = end

        while True:
            while left <= right and x[left] <= pivot:
                left = left + 1
            while x[right] >= pivot and right >= left:
                right = right -1
            if right < left:
                break

            self.manager.sort_manager.swap(left, right)

        self.manager.sort_manager.swap(start, right)

        return right

    def mark_cells(self, count):
        self.manager.intervals_processed += count

        return self

    def run(self):
        x, cv = self.manager.x, self.manager.cv

        while True:
            interval = None

            with cv:
                # Synchronized interval retrieving.
                while not len(self.manager.intervals):
                    if self.manager.intervals_processed >= x.shape[0]:
                        return

                    cv.wait()

                start, end = self.manager.intervals.pop(0)

            pivot = self.quicksort(start, end)

            with cv:
                if start < pivot - 1:
                    self.manager.intervals.append((start, pivot - 1))
                else:
                    self.mark_cells(pivot - start + 1)

                if pivot + 1 < end:
                    self.manager.intervals.append((pivot + 1, end))
                else:
                    self.mark_cells(pivot - start + 1)

                cv.notify_all()


class Sort:
    """Sorts an array x using the called method.

        Options are:
            -- "bubble"
            -- "quick"
            -- "bubble-parallel"
            -- "quick-parallel"
    """

    def __init__(self, x, n_jobs=1):
        self.x = x[:]
        self.n_jobs = n_jobs

    def swap(self, i, j):
        t = self.x[i]
        self.x[i] = self.x[j]
        self.x[j] = t

    def bubble(self):
        for length in range(len(self.x), 1, -1):
            for i in range(length - 1):
                if self.x[i] > self.x[i+1]:
                    self.swap(i, i+1)

        return self.x

    def bubble_parallel(self):
        raise NotImplementedError

    def quick(self):
        self._quicksort(0, len(self.x) - 1)
        return self.x

    def quick_parallel(self):
        QuickManager(self, n_jobs=self.n_jobs).run()
        return self.x

    def tim(self):
        return np.sort(self.x)

    def _quicksort(self, start, end):
        # Author: Sue Sentance.
        x = self.x

        if start < end:
            pivot = self._partition(start, end)
            self._quicksort(start, pivot - 1)
            self._quicksort(pivot + 1, end)

    def _partition(self, start, end):
        # Author: Sue Sentance.
        x = self.x

        pivot = x[start]
        left = start + 1
        right = end

        while True:
            while left <= right and x[left] <= pivot:
                left = left + 1
            while x[right] >= pivot and right >= left:
                right = right -1
            if right < left:
                break

            self.swap(left, right)

        self.swap(start, right)

        return right
