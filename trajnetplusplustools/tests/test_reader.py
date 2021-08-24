import numpy as np
import pytest
import trajnettools
from trajnettools.data import TrackRow


def test_scene_to_xy():
    paths = [
        [TrackRow(0, 1, 1.0, 1.0), TrackRow(10, 1, 1.0, 1.0), TrackRow(20, 1, 1.0, 1.0)],
        [TrackRow(10, 2, 2.0, 2.0), TrackRow(20, 2, 2.0, 2.0)],
        [TrackRow(0, 3, 3.0, 3.0), TrackRow(10, 3, 3.0, 3.0)],
    ]

    xy = trajnettools.Reader.paths_to_xy(paths)
    assert xy == pytest.approx(np.array([
        [[1.0, 1.0], [np.nan, np.nan], [3.0, 3.0]],
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        [[1.0, 1.0], [2.0, 2.0], [np.nan, np.nan]],
    ]), nan_ok=True)


