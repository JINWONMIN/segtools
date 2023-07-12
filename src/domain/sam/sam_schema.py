from pydantic import BaseModel


class PointsGenerator(BaseModel):
    points_per_side: str
    points_per_batch: str
    session_id: str
