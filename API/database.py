from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session


# Define the database URL for the SQLite database
SQLALCHEMY_DATABASE_URL = "sqlite:///./../database/isd.db"

# Create an engine that connects to the database
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Session object that can be used to interact with the database
session = Session(engine)

# Base class for declarative class definitions
Base = declarative_base()

# Dependency to get a database session
def get_db():
    db = SessionLocal()  # Instantiate a new database session
    try:
        yield db  # Yield the session to be used in the calling function
    finally:
        db.close()  # Ensure that the session is closed after use
