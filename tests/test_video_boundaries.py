import pytest

from app.video.boundaries import resolve_boundaries


def resolve(left, right, **kwargs):
    return resolve_boundaries(10, 30, 60, left, right, kwargs.pop('confidence', .8),
                              kwargs.pop('context_complete', True), minimum=5, maximum=30, **kwargs)


def test_valid_tightening_and_extension():
    assert resolve(12, 28)['applied'] == [12, 28]
    assert resolve(8, 32, context_start=5, context_end=35)['accepted']


@pytest.mark.parametrize('left,right,options', [
    (30, 10, {}), (-1, 20, {}), (10, 61, {}), (10, 31, {}),
    (10, 14, {}), (None, 20, {}), (float('nan'), 20, {}),
    (12, 28, {'confidence': .1}), (12, 28, {'context_complete': False}),
    (12, 28, {'enabled': False}),
])
def test_invalid_recommendations_fall_back(left, right, options):
    result = resolve(left, right, **options)
    assert not result['accepted']
    assert result['applied'] == [10, 30]
    assert result['reason']


def test_maximum_duration():
    result = resolve_boundaries(0, 50, 60, 0, 40, .8, True, 5, 30)
    assert not result['accepted']
    assert result['reason'] == 'duration outside limits'
