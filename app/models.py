from pydantic import BaseModel, Field
from typing import Dict, Optional

class ProjectDetails(BaseModel):
title: str = Field(..., description="Project title")
location: str = Field(..., description="Project location")
duration: int = Field(..., description="Project duration in months", gt=0)
budget: Optional[str] = Field(None, description="Project budget")
description: str = Field(..., description="Detailed project description")

class TenderResponse(BaseModel):
sections: Dict[str, str]
