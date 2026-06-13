import os
import uuid
import logging
import asyncio
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, BackgroundTasks, HTTPException, status

from gtda.time_series import TakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceImage

class TDAFeatureEngine:
    def __init__(self, dim: int = 3, delay: int = 2, res: int = 50):
        self.res = res
        self.embedder = TakensEmbedding(dimension=dim, time_delay=delay)
        self.homology = VietorisRipsPersistence(homology_dimensions=[0, 1])
        self.vectorizer = PersistenceImage(n_bins=res)

    def transform(self, series: np.ndarray) -> np.ndarray:
        try:
            x = np.nan_to_num(series).reshape(1, -1)
            embedded = self.embedder.fit_transform(x)
            diagrams = self.homology.fit_transform(embedded)
            images = self.vectorizer.fit_transform(diagrams)
            mi, ma = np.min(images), np.max(images)
            return (images - mi) / (ma - mi + 1e-8)
        except Exception:
            raise

class TopologicalCNN(nn.Module):
    def __init__(self, res: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))

class TaskResponse(BaseModel):
    task_id: str
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    result: Optional[float] = None
    error: Optional[str] = None

class TimeSeriesInput(BaseModel):
    series: List[float] = Field(..., min_items=20)

    @field_validator('series')
    @classmethod
    def validate_signal(cls, v: List[float]) -> List[float]:
        if np.std(v) < 1e-6:
            raise ValueError("Signal variance too low")
        return v

class InferenceOrchestrator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.res = 50
        self.engine = TDAFeatureEngine(res=self.res)
        self.model = TopologicalCNN(res=self.res).to(self.device)
        self.model.eval()
        self.tasks: Dict[str, TaskResponse] = {}
        self.executor = ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) // 2))

    async def process_task(self, task_id: str, data: List[float]):
        try:
            self.tasks[task_id].status = "PROCESSING"
            loop = asyncio.get_running_loop()
            features = await loop.run_in_executor(
                self.executor, self.engine.transform, np.array(data)
            )
            feat_tensor = torch.as_tensor(features, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                pred = self.model(feat_tensor).cpu().item()
            self.tasks[task_id].result = float(pred)
            self.tasks[task_id].status = "SUCCESS"
        except Exception as e:
            self.tasks[task_id].status = "FAILED"
            self.tasks[task_id].error = str(e)

app = FastAPI(title="TDA-ML Platform")
orchestrator = InferenceOrchestrator()

@app.post("/v1/inference", status_code=status.HTTP_202_ACCEPTED, response_model=TaskResponse)
async def submit_inference(payload: TimeSeriesInput, bg: BackgroundTasks):
    t_id = str(uuid.uuid4())
    task = TaskResponse(task_id=t_id, status="QUEUED")
    orchestrator.tasks[t_id] = task
    bg.add_task(orchestrator.process_task, t_id, payload.series)
    return task

@app.get("/v1/status/{task_id}", response_model=TaskResponse)
async def get_status(task_id: str):
    if task_id not in orchestrator.tasks:
        raise HTTPException(status_code=404)
    return orchestrator.tasks[task_id]

@app.get("/health")
async def health():
    return {"status": "ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)

