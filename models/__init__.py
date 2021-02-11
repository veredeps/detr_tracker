# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .redetector import build as build_redetector
from .redetector import build_from_detr

def build_model(args):
    return build(args)


def build_redetectr_model(args):
    return build_redetector(args)


def build_redetector_from_detr(args, detr_model, criterion):
    return build_from_detr(args, detr_model, criterion)