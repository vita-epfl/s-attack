import trajnettools


def test_average_l2():
    path1 = [trajnettools.TrackRow(0, 0, x, 0.0) for x in range(21)]
    path2 = [trajnettools.TrackRow(0, 0, x, 1.0) for x in range(21)]
    assert trajnettools.metrics.average_l2(path1, path2) == 1.0


def test_final_l2():
    path1 = [trajnettools.TrackRow(0, 0, x, 0.0) for x in range(21)]
    path2 = [trajnettools.TrackRow(0, 0, x, 1.0) for x in range(21)]
    assert trajnettools.metrics.final_l2(path1, path2) == 1.0


def test_collision():
    path1 = [trajnettools.TrackRow(x, 0, x, x) for x in range(21)]
    path2 = [trajnettools.TrackRow(x, 0, x, 21-x) for x in range(21)]
    assert trajnettools.metrics.collision(path1, path2) == 1
