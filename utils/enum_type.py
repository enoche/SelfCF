# -*- coding: utf-8 -*-
# @Time   : 2020/8/9
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
#
# Update by xin
# @Email: enoche.chow@gmail.com
"""
#######################
"""

from enum import Enum


class ModelType(Enum):
    """Type of models.
    - ``GENERAL``: General Recommendation, DEFAULT
    - ``SEQUENTIAL``: Sequential Recommendation
    """
    GENERAL = 0
    SEQUENTIAL = 1

