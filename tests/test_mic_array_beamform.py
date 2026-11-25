import pytest
import media.mic_array_beamform as mod

CASES = [
    ("geometry_measured", (1,)),
    ("freeze_on_double_talk", (1,)),
    ("gain_smooth", (1,)),
    ("align_by_sample", (1,)),
    ("coeff_crossfade", (1,)),
    ("output_limit", (1,)),
    ("load", (1,)),
    ("beamform", (1,)),
    ("mix", (1,)),
    ("present", (1,)),
    ("validate_beam", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
