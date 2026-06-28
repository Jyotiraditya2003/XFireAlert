import sys
sys.path.insert(0, 'src')

from database import DB_AVAILABLE, log_alert, get_all_alerts, get_alert_stats, close_connection

print("=" * 50)
print("MongoDB Connection Test")
print("=" * 50)

if DB_AVAILABLE:
    print("Status: CONNECTED")
    
    test_id = log_alert(
        p_risk=0.75,
        p_det=0.85,
        p_final=0.80,
        alert="HIGH ALERT",
        features={"Temperature": 0.12, "RH": -0.08}
    )
    
    if test_id:
        print(f"Test alert inserted with ID: {test_id}")
    
    stats = get_alert_stats()
    if stats:
        print(f"\nDatabase Stats:")
        print(f"  Total alerts: {stats['total_alerts']}")
        print(f"  High alerts: {stats['high_alerts']}")
        print(f"  Moderate alerts: {stats['moderate_alerts']}")
        print(f"  Low alerts: {stats['low_alerts']}")
    
    close_connection()
else:
    print("Status: NOT CONNECTED")
    print("\nTo connect, create a .env file with:")
    print("  MONGODB_URI=mongodb+srv://<user>:<password>@cluster.mongodb.net/...")
    print("  DATABASE_NAME=xfirealert")
    print("  COLLECTION_NAME=alerts")

print("=" * 50)
