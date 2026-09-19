from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from river import linear_model, preprocessing
import networkx as nx
import math
import datetime

app = FastAPI(title="GeoGuide Luxor - GeoAI Engine")

# السماح للواجهة بالاتصال بالخادم محلياً أو بعد الرفع (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. تهيئة نموذج التعلّم المستمر اللحظي (Online Learning Pipeline)
model = preprocessing.StandardScaler() | linear_model.LinearRegression()

# 2. بناء شبكة تجريبية مبسطة (Graph)
G = nx.Graph()

# نماذج الطلبات
class RouteRequest(BaseModel):
    origin: list       # [lat, lng]
    destination: list  # [lat, lng]
    timestamp: str = None

class FeedbackRequest(BaseModel):
    route_id: str
    actual_duration_seconds: float

def haversine_distance(coord1, coord2):
    """حساب المسافة الهندسية التقريبية بالمتر بين نقطتين"""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371000  # نصف قطر الأرض بالمتر
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

@app.get("/")
def home():
    return {"message": "GeoGuide Luxor GeoAI Backend is running successfully!"}

@app.post("/api/get-smart-route")
def get_smart_route(req: RouteRequest):
    now = datetime.datetime.now()
    hour = now.hour
    is_weekend = 1 if now.weekday() in [4, 5] else 0

    dist_meters = haversine_distance(req.origin, req.destination)
    
    # ميزات التنبؤ للنموذج اللحظي
    features = {
        'distance_m': dist_meters,
        'hour': hour,
        'is_weekend': is_weekend
    }
    
    # التنبؤ بزمن العبور بالثواني عبر نموذج الـ AI
    predicted_seconds = model.predict_one(features)
    
    # إذا كان النموذج في بدايته، نعتمد متوسط سرعة افتراضي (30 كم/س داخل المدينة)
    if predicted_seconds is None or predicted_seconds <= 0:
        predicted_seconds = (dist_meters / 8.33)  # 8.33 م/ث = 30 كم/س

    eta_minutes = round(predicted_seconds / 60, 1)

    # مخرجات المسار بصيغة GeoJSON
    geojson_data = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [req.origin[1], req.origin[0]],         # [lng, lat]
                [req.destination[1], req.destination[0]]
            ]
        },
        "properties": {
            "distance_meters": round(dist_meters, 1),
            "eta_minutes": eta_minutes
        }
    }

    return {
        "route_id": f"route_{int(now.timestamp())}",
        "eta_minutes": eta_minutes,
        "geojson": geojson_data
    }

@app.post("/api/feedback-trip")
def feedback_trip(data: FeedbackRequest):
    now = datetime.datetime.now()
    features = {
        'distance_m': 1000.0, # يتم مطابقتها مع مسافة المسار
        'hour': now.hour,
        'is_weekend': 1 if now.weekday() in [4, 5] else 0
    }
    
    # التعلّم اللحظي وتحديث وزن النموذج مباشرة
    model.learn_one(features, data.actual_duration_seconds)
    
    return {
        "status": "success",
        "message": "Model updated in real-time with actual trip duration."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)