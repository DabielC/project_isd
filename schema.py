from .database import *
from sqlalchemy import *
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey

class face_glasses(Base):
    __tablename__ = 'face_glasses'
    id = Column(Integer, primary_key=True)
    face_type = Column(String, ForeignKey('face_shape.face_type'))
    glasses_type = Column(String, ForeignKey('glasses_class.glasses_type'))
    suitability = Column(String)

    # Relationships
    face_shape = relationship("face_shape", back_populates="face_glasses")
    glasses_class = relationship("glasses_class", back_populates="face_glasses")


class face_shape(Base):
    __tablename__ = 'face_shape'
    face_type = Column(String, primary_key=True)

    # Relationship
    face_glasses = relationship("face_glasses", back_populates="face_shape")


class glasses_class(Base):
    __tablename__ = 'glasses_class'
    glasses_type = Column(String, primary_key=True)

    # Relationships
    face_glasses = relationship("face_glasses", back_populates="glasses_class")
    glasses_product = relationship("glasses_product", back_populates="glasses_class")


class glasses_product(Base):
    __tablename__ = 'glasses_product'
    id = Column(Integer, primary_key=True)
    glasses_type = Column(String, ForeignKey('glasses_class.glasses_type'))
    glasses_img = Column(String)

    # Relationships
    glasses_class = relationship("glasses_class", back_populates="glasses_product")


class user_reflection(Base):
    __tablename__ = 'user_reflection'
    id = Column(Integer, primary_key=True)
    like = Column(Integer)
    comment = Column(String)
    img = Column(String)
    create_at = Column(String)
    mobilenet_score = Column(String)
    yolov8_score = Column(String)
    vote_score = Column(String)
