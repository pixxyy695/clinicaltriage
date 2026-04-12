# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pydantic import BaseModel
from typing import Optional, Literal, List


class Observation(BaseModel):
    case_id: str
    symptoms: str
    history: List[str]
    status: str


class Action(BaseModel):
    type: Literal["ask", "triage"]

    question: Optional[str] = None

    urgency: Optional[Literal["low", "medium", "high"]] = None
    department: Optional[
        Literal["cardiology", "neurology", "pulmonology", "general", "gastroenterology"]
    ] = None

    next_step: Optional[
        Literal[
            "ECG",
            "CT scan",
            "oxygen support",
            "basic checkup",
            "ultrasound",
            "blood test",
            "rest and hydration"
        ]
    ] = None

    confidence: Optional[float] = None