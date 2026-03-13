"""
Database Service Handler
Manages MongoDB connections, User Model, Auth Utils, and History records.
"""
import os
import atexit
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
import jwt
from pydantic import BaseModel, EmailStr, Field
from bson import ObjectId

# Load environment logic
from dotenv import load_dotenv
load_dotenv()

# Basic MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "xlite_db")

# Setup JWT parameters
SECRET_KEY = os.getenv("SECRET_KEY", "fallback_secret_key_default")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class Database:
    client: AsyncIOMotorClient = None
    db = None

db = Database()

def connect_to_mongo():
    try:
        db.client = AsyncIOMotorClient(MONGODB_URL)
        db.db = db.client[MONGODB_DB_NAME]
        print(f"Connected to MongoDB database: {MONGODB_DB_NAME}")
    except Exception as e:
        print(f"Could not connect to MongoDB: {e}")

def close_mongo_connection():
    if db.client:
        db.client.close()
        print("Closed MongoDB connection")

class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return str(ObjectId(v))

# Modern Pydantic v2 approach for ObjectId
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic_core import core_schema, core_schema as cs
from typing import Any

class PyObjectIdAnnotated:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.str_schema()
        )

    @classmethod
    def validate(cls, v: Any) -> ObjectId:
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

class UserInDB(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    username: str
    email: EmailStr
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

# Prediction History Model
class PredictionRecord(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    user_id: str
    filename: str
    original_image_path: str
    heatmap_image_path: Optional[str] = None
    predictions: List[Dict[str, Any]]  # List of {disease, prob, risk}
    pdf_report_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

# Auth Utilities
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
