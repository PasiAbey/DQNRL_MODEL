from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.data.tables import TableServiceClient
from datetime import datetime, timezone, timedelta
import uuid
import json
import os
import numpy as np
import joblib
from rl_agent import RLAgent, get_rl_state_vector

app = FastAPI(title="Skill-Quest API (Double Engine)")

# Configure Azure Table Storage
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
pending_client = None
replay_client = None

if CONNECTION_STRING:
    table_service = TableServiceClient.from_connection_string(conn_str=CONNECTION_STRING)
    pending_client = table_service.create_table_if_not_exists(table_name="PendingInteractions")
    replay_client = table_service.create_table_if_not_exists(table_name="ExperienceReplay")

# ---------------------------------------------------------
# LOAD THE DOUBLE BRAINS
# ---------------------------------------------------------
# 1. Load the RL Agent
agent = RLAgent()
rl_model_path = os.path.join(os.path.dirname(__file__), "trained_rl_agent.pth")
agent.load_pretrained_model(rl_model_path)

# 2. Load the Risk Predictor
risk_model_path = os.path.join(os.path.dirname(__file__), "trained_risk_model.joblib")
risk_model = joblib.load(risk_model_path)

# ---------------------------------------------------------
# RISK MATH FUNCTIONS
# ---------------------------------------------------------
def calculate_engagement(active_minutes, quiz_accuracy, modules_done, days_since_last_login):
    time_score = min(active_minutes / 60.0, 1.0)
    decay_factor = np.exp(-0.5 * days_since_last_login)
    raw_engagement = (0.5 * time_score) + (0.3 * quiz_accuracy) + (0.2 * (modules_done > 0))
    return raw_engagement * decay_factor

def calculate_reward_score(recent_points, total_badges_count):
    points_value = np.tanh(recent_points / 500.0)
    badge_value = 1.0 if total_badges_count > 0 else 0.0
    return (0.7 * points_value) + (0.3 * badge_value)

# ---------------------------------------------------------
# API ROUTES
# ---------------------------------------------------------
class PredictRequest(BaseModel):
    level: str
    duration_norm: float
    consecutive_norm: float
    daily_xp_norm: float
    active_minutes: float
    quiz_accuracy: float
    modules_done: int
    days_since_last_login: int
    recent_points: float
    total_badges_count: int

class FeedbackRequest(BaseModel):
    interaction_id: str
    engaged: bool

@app.get("/")
async def root():
    return {"status": "online", "service": "Skill-Quest Inference API"}

@app.post("/predict")
async def predict_action(req: PredictRequest):
    interaction_id = str(uuid.uuid4())
    
    # Calculate Risk
    eng_score = calculate_engagement(req.active_minutes, req.quiz_accuracy, req.modules_done, req.days_since_last_login)
    rew_score = calculate_reward_score(req.recent_points, req.total_badges_count)
    student_features = np.array([[eng_score, rew_score]])
    retention_prob = risk_model.predict_proba(student_features)[0][1]
    calculated_risk_score = 1.0 - retention_prob 
    
    # Get Action
    state_vector = get_rl_state_vector(
        req.level, req.duration_norm, calculated_risk_score, 
        req.quiz_accuracy, req.consecutive_norm, req.daily_xp_norm
    )
    action_id = agent.choose_action(state_vector)
    
    # Save to Azure
    if pending_client:
        entity = {
            "PartitionKey": "LiveInteractions",
            "RowKey": interaction_id,
            "StateVector": json.dumps(state_vector.tolist()), 
            "ActionId": action_id,
            "Timestamp_UTC": datetime.now(timezone.utc).isoformat()
        }
        pending_client.create_entity(entity=entity)
    
    return {
        "interaction_id": interaction_id, 
        "action_id": action_id,
        "risk_score": float(calculated_risk_score)
    }

@app.post("/feedback")
async def receive_feedback(req: FeedbackRequest):
    if not pending_client or not replay_client:
        return {"status": "error", "message": "Database not connected."}

    try:
        # Retrieve short-term memory
        entity = pending_client.get_entity(partition_key="LiveInteractions", row_key=req.interaction_id)
        
        # 13-Hour Expiration Check
        saved_time = datetime.fromisoformat(entity["Timestamp_UTC"])
        if datetime.now(timezone.utc) - saved_time > timedelta(hours=13):
            pending_client.delete_entity(partition_key="LiveInteractions", row_key=req.interaction_id)
            return {"status": "ignored", "message": "Interaction expired."}
            
        # Unpack saved data
        old_state = json.loads(entity["StateVector"])
        action_taken = entity["ActionId"]
        
        # Calculate Reward
        reward = 10.0 if req.engaged else -10.0
        
        # Move to permanent ExperienceReplay table for batch training
        replay_entity = {
            "PartitionKey": "BatchData",
            "RowKey": str(uuid.uuid4()),
            "State": json.dumps(old_state),
            "Action": int(action_taken),
            "Reward": float(reward),
            "NextState": json.dumps(old_state),
            "Done": True,
            "Timestamp_UTC": datetime.now(timezone.utc).isoformat()
        }
        replay_client.create_entity(entity=replay_entity)
        
        # Clean up short-term database
        pending_client.delete_entity(partition_key="LiveInteractions", row_key=req.interaction_id)
        
        return {"status": "success", "message": f"Experience saved for batch training with reward: {reward}"}
        
    except Exception as e:
        raise HTTPException(status_code=404, detail="Interaction ID not found.")