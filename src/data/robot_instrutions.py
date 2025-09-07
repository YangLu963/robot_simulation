
from typing import List, Optional, Literal
from pydantic import BaseModel

class ObjectAttributes(BaseModel):
    color: Optional[str] = None
    shape: Optional[str] = None
    size: Optional[str] = None

class Object(BaseModel):
    id: int
    name: str
    attributes: Optional[ObjectAttributes] = None

class Precondition(BaseModel):
    type: Literal["object_visible", "object_reachable", "object_grasped"]
    object_id: int

class Target(BaseModel):
    object_id: int
    grasp_pose: Optional[List[float]] = None

class Destination(BaseModel):
    object_id: Optional[int] = None
    relation: Optional[Literal["left", "right", "above", "below"]] = None
    offset: Optional[List[float]] = None

class Step(BaseModel):
    action: Literal["pick", "place", "move", "open", "close"]
    target: Optional[Target] = None
    destination: Optional[Destination] = None
    preconditions: Optional[List[Precondition]] = []

class SafetyConstraint(BaseModel):
    type: Literal["avoid_collision", "force_limit"]
    parameters: Optional[dict] = None

class RobotInstruction(BaseModel):
    intent: Literal["transfer", "arrange", "clean", "fetch"]
    objects: List[Object]
    steps: List[Step]
    safety_constraints: List[SafetyConstraint] = []
