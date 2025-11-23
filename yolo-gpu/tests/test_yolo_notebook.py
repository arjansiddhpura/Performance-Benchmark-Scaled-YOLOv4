# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os
import pathlib

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor


### Pytest markers
@pytest.mark.ipus("1")
def test_yolo_v4(tmp_path):
    """
    Test the notebook for object detection with YOLOv4 model (notebook_yolo.ipynb)
    which is meant for Paperspace.
    """
    working_path = pathlib.Path(__file__).parents[1]
    os.environ["NUM_AVAILABLE_IPU"] = "4"
    notebook_filename = working_path / "notebook_yolo.ipynb"
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{working_path}"}})
