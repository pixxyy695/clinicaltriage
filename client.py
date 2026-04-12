# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import Action, Observation


class MyEnv(
    EnvClient[Action, Observation, State]
):

    def _step_payload(self, action: Action) -> Dict:
        return {
            "type": action.type,
            "urgency": action.urgency,
            "department": action.department,
            "next_step": action.next_step
        }

    def _parse_result(self, payload: Dict) -> StepResult[Observation]:
        obs_data = payload.get("observation", {})

        observation = Observation(
            case_id=obs_data.get("case_id"),
            symptoms=obs_data.get("symptoms"),
            history=obs_data.get("history", []),
            status=obs_data.get("status")
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )