# To do, the following code is not valid

import falcon
from falcon import testing
import pytest

from eeris_nilm.app import api


@pytest.fixture
def client():
    return testing.TestClient(api)


# pytest will inject the object returned by the "client" function
# as an additional parameter.
def test_eeris_nilm_put(client):
    doc = {}  # json or whatever object is for the put function

    response = client.simulate_put('/')  # Change the route to fit definition
    result_doc = {}  # json simulate response

    assert result_doc == doc
    assert response.status == falcon.HTTP_OK
