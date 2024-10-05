from .prediction import *
from .pydantic_models import *
from .database import *
from .schema import *
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from sqlalchemy.orm import joinedload

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predicted(info: ItemCreated, db: Session = Depends(get_db)):
    try:
        face_detect = face_crop(base64_to_image(info.image))
        if face_detect is not None:
            image_tensor = transform_image(face_detect)
            predicted_mobile = predict_MobileNet(image_tensor)
            predicted_yolo = predict_YOLO(face_detect)
            vote_score = vote(predicted_mobile, predicted_yolo)
            
            db_item = user_reflection(
                img=info.image,
                create_at=str(datetime.now()),
                mobilenet_score=str(predicted_mobile),
                yolov8_score=str(predicted_yolo),
                vote_score=str(vote_score)
            )
            
            
            db_select = (
                db.query(face_glasses, glasses_product)
                .join(glasses_product, face_glasses.glasses_type == glasses_product.glasses_type)
                .filter(face_glasses.face_type == vote_score["class"])
                .options(joinedload(face_glasses.glasses_class))
                .all()
            )

            
            db.add(db_item)
            db.commit()
            db.refresh(db_item)

            products = []
            for fg, gp in db_select:
                products.append({
                    "glasses_type": fg.glasses_type,
                    "suitability": fg.suitability,
                    "glasses_img": gp.glasses_img
                })

            return {
                    "id": db_item.id,
                    "mscore": predicted_mobile["score"],
                    "yscore": predicted_yolo["score"],
                    "vscore": vote_score,
                    "dt": db_item.create_at,
                    "products": products
                }
        
        else:
            db_item = user_reflection(
                img=info.image,
                create_at=str(datetime.now())
            )
            db.add(db_item)
            db.commit()
            db.refresh(db_item)
            return "No face detected"
    except:
        raise HTTPException(status_code=400, detail="Input the image with base64")


@app.get("/test_db")
def test_db(db: Session = Depends(get_db)):
    items = db.query(glasses_product).all()
    return items

@app.put("/update_reflection")
async def update_user_reflection(item: update_reflect, db: Session = Depends(get_db)):
    db_item = db.query(user_reflection).filter(user_reflection.id == item.id).first()
    if db_item:
        for key, value in item.model_dump().items():
            setattr(db_item, key, value)
        db.commit()
        db.refresh(db_item)
        return 'Update Success!'
    else:
        raise HTTPException(status_code=404, detail="Item not found")

@app.get("/glasses_item/{item_id}", response_model=GlassResponse)
def get_item(item_id: int, db: Session = Depends(get_db)):
    db_item = db.query(glasses_product).filter(glasses_product.id == item_id).first()
    return GlassResponse(id=db_item.id, type=db_item.glasses_type, image=db_item.glasses_img)


