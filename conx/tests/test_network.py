from conx import Network

def test_network():
    net = Network(2, 2, 1)

    assert net is not None
