import pydantic as _pydantic
from typing import Optional
from fastapi import UploadFile
from typing import List


class _PromptBase(_pydantic.BaseModel):
    seed: Optional[float] = -1
    num_inference_steps: int = 30
    guidance_scale: float = 12
    strength: float = 0.6


class BabyCreate(_PromptBase):
    encoded_mom_imgs: List[str]
    encoded_dad_imgs: List[str]
    img_height: int = 512
    gender: str = 'female'
    power_of_dad: int = 50
    ethnicity: str = 'unknown'
    focal_length: float = 0.0
    total_number_of_photos: int = 3
    gamma: float = 0.47
    eta: float = 0.4
    token: str = '1230pol>EUe208tq'
