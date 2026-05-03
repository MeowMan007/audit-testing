"""
Configuration Module for AccessLens.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "AccessLens"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Model Config
    MODEL_PATH: str = "dataset/accessibility_model.pth"
    CONFIDENCE_THRESHOLD: float = 0.4
    
    # Database (use /app/data in Docker for writable storage)
    DATABASE_PATH: str = "data/audit_history.db"
    
    # Hugging Face AI Insights
    HF_TOKEN: str = ""
    
    # CORS
    CORS_ORIGINS: list = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
