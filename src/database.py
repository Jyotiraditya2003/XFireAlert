from pymongo import MongoClient
from datetime import datetime
import os

MONGO_URI = "mongodb+srv://xfire:xfire@123@xfire.xgepeni.mongodb.net/?appName=xfire"

client = None
collection = None

# Building safe connection
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client["xfirealert"]
    collection = db["alerts"]

    # testing connection
    client.server_info()

    DB_AVAILABLE = True
    print("MongoDB Connected Successfully")

except Exception as e:
    DB_AVAILABLE = False
    print("MongoDB not available. Running in offline mode.")


def log_alert(p_risk, p_det, p_final, alert, features):

    # If database found offline → skips logging
    if not DB_AVAILABLE:
        return

    try:
        record = {
            "timestamp": datetime.utcnow(),
            "weather_risk": p_risk,
            "vision_confidence": p_det,
            "final_probability": p_final,
            "alert_level": alert,
            "top_features": features
        }

        collection.insert_one(record)

    except Exception:
        # Fixing Crash
        pass