# -*- coding: utf-8 -*-

import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache

import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.logger import create_small_table
import pdb

from fsdet.evaluation.evaluator import DatasetEvaluator

class GroceryDatasetEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name): # initial needed variables
        self._dataset_name = dataset_name

    def reset(self): # reset predictions
        self._predictions = []

    def process(self, inputs, outputs): # prepare predictions for evaluation
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            if "instances" in output:
                prediction["instances"] = output["instances"]
            self._predictions.append(prediction)

    def evaluate(self): # evaluate predictions
        pdb.set_trace()
        # results = evaluate_predictions(self._predictions)
        return {
            'AP': 0.25,
            'AP50': 0.50,
            'AP75': 0.75
        }
        return {
            "AP": results["AP"],
            "AP50": results["AP50"],
            "AP75": results["AP75"],
        }