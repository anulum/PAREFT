import numpy as np
from pareft.probes import save_psd_csv

def test_psd(tmp_path):
    t = np.linspace(0, 10, 10000)
    x = np.sin(2*np.pi*3*t)
    f, P = save_psd_csv(t, x, tmp_path/"psd.csv")
    assert f.shape[0] == P.shape[0]
    assert (f > 0).any()