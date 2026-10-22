from pathlib import Path
import numpy as np
from PIL import Image
from ml.features import extract_features
from ml.dataset import validate_image

def test_feature_vector_is_stable():
    work = Path("artifacts/test_tmp"); work.mkdir(parents=True, exist_ok=True); path = work / "feature_image.png"; Image.new("RGB", (24, 32), (30, 120, 40)).save(path)
    one = extract_features(str(path)); two = extract_features(str(path))
    assert one.ndim == 1 and one.size > 0 and np.allclose(one, two)

def test_validate_image():
    work = Path("artifacts/test_tmp"); work.mkdir(parents=True, exist_ok=True); path = work / "validate_image.jpg"; Image.new("RGB", (10, 12), "green").save(path)
    result = validate_image(path)
    assert result["readable"] is True and result["width"] == 10
