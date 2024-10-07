from pydantic import BaseModel
from typing import Optional

#Request
class get_info(BaseModel):
    image: str

class update_like(BaseModel):
    id: int
    like: Optional[int] = None

class update_comment(BaseModel):
    id: int
    comment : str

class ItemCreated(get_info):
    pass

#Response
class GlassResponse(get_info):
    id: int
    type : str
    image: str

