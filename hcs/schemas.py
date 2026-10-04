from typing import Literal
from pydantic import BaseModel, Field

class Check(BaseModel):
     
    id: str = Field(description="The check's id, copied exactly from the input, e.g. 'TL1'.")
    check: str = Field(description="The check's text, copied exactly from the input.")
    evidence: str | None = Field(
        default=None,
        description="An exact quote from the CV relevant to this check, or null if the CV has none.",
    )
    passed: bool = Field(description="True only if the evidence clearly proves the check.") 
    
Decision = Literal["strong_no", "lean_no", "lean_hire", "strong_hire"]

class Review(BaseModel):
    checks: list[Check] = Field(description="One entry per check, in the order given.")
    decision: Decision = Field(description="The overall decision, made after judging every check.")
    
    