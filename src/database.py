from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "")
DATABASE_NAME = os.getenv("DATABASE_NAME", "xfirealert")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "alerts")

client = None
db = None
collection = None
DB_AVAILABLE = False

def connect_db():
    global client, db, collection, DB_AVAILABLE
    
    if not MONGODB_URI:
        print("MongoDB URI not found. Running in offline mode.")
        return False
    
    try:
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            maxPoolSize=10
        )
        client.admin.command('ping')
        
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        DB_AVAILABLE = True
        print("MongoDB Connected Successfully")
        return True
        
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        DB_AVAILABLE = False
        return False

connect_db()

def log_alert(p_risk, p_det, p_final, alert, features):
    if not DB_AVAILABLE:
        return None
    
    try:
        record = {
            "timestamp": datetime.utcnow(),
            "weather_risk": p_risk,
            "vision_confidence": p_det,
            "final_probability": p_final,
            "alert_level": alert,
            "top_features": features
        }
        result = collection.insert_one(record)
        return result.inserted_id
    except Exception as e:
        print(f"Failed to log alert: {e}")
        return None

def get_all_alerts(limit=100):
    if not DB_AVAILABLE:
        return []
    
    try:
        alerts = list(collection.find().sort("timestamp", -1).limit(limit))
        for alert in alerts:
            alert["_id"] = str(alert["_id"])
        return alerts
    except Exception as e:
        print(f"Failed to fetch alerts: {e}")
        return []

def get_alerts_by_level(alert_level, limit=50):
    if not DB_AVAILABLE:
        return []
    
    try:
        alerts = list(collection.find({"alert_level": alert_level}).sort("timestamp", -1).limit(limit))
        for alert in alerts:
            alert["_id"] = str(alert["_id"])
        return alerts
    except Exception as e:
        print(f"Failed to fetch alerts: {e}")
        return []

def get_alert_stats():
    if not DB_AVAILABLE:
        return None
    
    try:
        total = collection.count_documents({})
        high = collection.count_documents({"alert_level": "HIGH ALERT"})
        moderate = collection.count_documents({"alert_level": "MODERATE WARNING"})
        low = collection.count_documents({"alert_level": "LOW RISK"})
        
        return {
            "total_alerts": total,
            "high_alerts": high,
            "moderate_alerts": moderate,
            "low_alerts": low
        }
    except Exception as e:
        print(f"Failed to get stats: {e}")
        return None

def delete_alert(alert_id):
    if not DB_AVAILABLE:
        return False
    
    try:
        from bson.objectid import ObjectId
        result = collection.delete_one({"_id": ObjectId(alert_id)})
        return result.deleted_count > 0
    except Exception as e:
        print(f"Failed to delete alert: {e}")
        return False

def close_connection():
    global client
    if client:
        client.close()
        print("MongoDB connection closed")
