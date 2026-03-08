"""
Centralized configuration for Operation Antigravity Grid.
All configuration is validated at load time with Pydantic.
"""
import os
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, validator
import yaml
import logging

logger = logging.getLogger(__name__)


class FirestoreConfig(BaseModel):
    """Firebase Firestore configuration"""
    project_id: str = Field(default="antigravity-grid")
    collection_prefix: str = Field(default="antigravity")
    service_account_path: Optional[str] = Field(default="config/service_account.json")
    
    @validator('service_account_path')
    def validate_service_account(cls, v):
        if v and not os.path.exists(v):
            logger.warning(f"Service account file not found: {v}. Will use ADC.")
        return v


class TelemetryConfig(BaseModel):
    """Telemetry collection configuration"""
    collection_interval_seconds: int = Field(default=5, ge=1, le=300)
    max_telemetry_age_minutes: int = Field(default=60, ge=5, le=1440)
    
    # Process monitoring
    target_process_names: List[str] = Field(default=["python", "antigravity-helper"])
    exclude_process_names: List[str] = Field(default=["systemd", "dbus-daemon"])
    
    # GPU monitoring
    enable_gpu_monitoring: bool = Field(default=True)
    gpu_vendors: List[Literal["nvidia", "amd", "intel"]] = Field(default=["nvidia", "amd"])
    
    # Safety thresholds
    max_cpu_percent_per_process: float = Field(default=95.0, ge=1.0, le=100.0)
    min_memory_mb_available: int = Field(default=1024, ge=128)


class BrainConfig(BaseModel):
    """Orchestrator Brain configuration"""
    prediction_horizon_seconds: int = Field(default=60, ge=30, le=300)
    model_retrain_interval_hours: int = Field(default=24, ge=1, le=168)
    
    # Resource allocation
    target_efficiency_gain_percent: float = Field(default=10.0, ge=1.0, le=30.0)
    min_slice_duration_seconds: int = Field(default=30, ge=10)
    max_slices_concurrent: int = Field(default=4, ge=1, le=16)
    
    # Safety controllers
    pid_kp: float = Field(default=0.8, ge=0.1, le=2.0)  # Proportional gain
    pid_ki: float = Field(default=0.2, ge=0.0, le=1.0)  # Integral gain
    pid_kd: float = Field(default=0.1, ge=0.0, le=0.5)  # Derivative gain
    circuit_breaker_threshold: float = Field(default=2.0, ge=1.1, le=5.0)  # Latency multiplier


class SimulationConfig(BaseModel):
    """Market simulation configuration"""
    ray_head_address: str = Field(default="auto")  # "auto", "localhost:6379", or IP
    docker_image: str = Field(default="antigravity/simulation-worker:latest")
    simulation_timeout_seconds: int = Field(default=300, ge=30)
    
    # Exchange configuration
    supported_exchanges: List[str] = Field(default=["binance", "coinbase", "kraken"])
    max_concurrent_backtests: int = Field(default=8, ge=1)
    
    # Cloud bursting
    enable_cloud_burst: bool = Field(default=False)
    cloud_provider: Literal["aws", "gcp", "azure", "none"] = Field(default="none")
    max_cloud_spend_daily: float = Field(default=10.0, ge=0.0)


class AntigravityConfig(BaseModel):
    """Root configuration object"""
    firestore: FirestoreConfig = Field(default_factory=FirestoreConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    brain: BrainConfig = Field(default_factory=BrainConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    
    # System identification
    host_id: str = Field(default_factory=lambda: os.uname().nodename)
    deployment_mode: Literal["development", "staging", "production"] = Field(default="development")
    
    @classmethod
    def from_yaml(cls, path: str = "config/antigravity.yaml") -> "AntigravityConfig":
        """Load configuration from YAML file"""
        if not os.path.exists(path):
            logger.warning(f"Config file {path} not found, using defaults")
            return cls()
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    def to_yaml(self, path: str = "config/antigravity.yaml"):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)


# Global configuration instance
config = AntigravityConfig.from_yaml()