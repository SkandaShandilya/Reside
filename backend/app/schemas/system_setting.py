from pydantic import BaseModel, Field


class SystemSettingResponse(BaseModel):
    ai_analysis_fps: int
    confidence_threshold: float
    alert_cooldown_seconds: int

    class Config:
        from_attributes = True


class SystemSettingUpdate(BaseModel):
    ai_analysis_fps: int = Field(ge=1, le=30)
    confidence_threshold: float = Field(ge=0.5, le=0.99)
    alert_cooldown_seconds: int = Field(ge=5, le=600)
