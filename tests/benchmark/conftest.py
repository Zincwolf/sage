"""Apply the production attention policy only to this benchmark test suite."""

import pytest

from weight_vq.inference.attention import preferred_sdpa


@pytest.fixture(autouse=True)
def sdpa_backend_policy(request):
    if request.node.get_closest_marker("cuda") is None:
        yield
    else:
        with preferred_sdpa():
            yield
