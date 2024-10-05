from pydantic import BaseModel
from typing import Optional

#Request
class get_info(BaseModel):
    image: str

class update_reflect(BaseModel):
    id: int
    like : int
    comment : str

class ItemCreated(get_info):
    pass

#Response
class ItemResponse(update_reflect):
    id : int
    like : int
    comment : str

class GlassResponse(get_info):
    id: int
    type : str
    image: str

