from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.data.tables import TableServiceClient
from datetime import datetime, timezone, timedelta
import uuid
import json
import os
from rl_agent import RLAgent, get_rl_state_vector

app = FastAPI(title="Skill-Quest RL API")

# Configure Azure Table Storage
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
pending_client = None
replay_client = None

if CONNECTION_STRING:
    table_service = TableServiceClient.from_connection_string(conn_str=CONNECTION_STRING)
    # Table 1: Short-term memory for the 12-hour delay
    pending_client = table_service.create_table_if_not_exists(table_name="PendingInteractions")
    # Table 2: Long-term memory for the nightly Batch Training script
    replay_client = table_service.create_table_if_not_exists(table_name="ExperienceReplay")

# Initialize the RL Agent and load the pre-trained weights
agent = RLAgent()
model_path = os.path.join(os.path.dirname(__file__), "trained_rl_agent.pth")
agent.load_pretrained_model(model_path)

# Define API request formats
class PredictRequest(BaseModel):
    level: str
    duration_norm: float
    risk_score: float
    quiz_norm: float
    consecutive_norm: float
    daily_xp_norm: float

class FeedbackRequest(BaseModel):
    interaction_id: str
    engaged: bool

@app.get("/")
async def root():
    """Simple health check so the main domain doesn't return a 404."""
    return {"status": "online", "service": "Skill-Quest Inference API"}

@app.post("/predict")
async def predict_action(req: PredictRequest):
    interaction_id = str(uuid.uuid4())
    
    # Generate state and get action using the pre-trained brain
    state_vector = get_rl_state_vector(
        req.level, req.duration_norm, req.risk_score, 
        req.quiz_norm, req.consecutive_norm, req.daily_xp_norm
    )
    action_id = agent.choose_action(state_vector)
    
    # Save the 12-hour memory to Azure Table Storage
    if pending_client:
        entity = {
            "PartitionKey": "LiveInteractions",
            "RowKey": interaction_id,
            "StateVector": json.dumps(state_vector.tolist()), 
            "ActionId": action_id,
            "Timestamp_UTC": datetime.now(timezone.utc).isoformat()
        }
        pending_client.create_entity(entity=entity)
    
    return {"interaction_id": interaction_id, "action_id": action_id}

@app.post("/feedback")
async def receive_feedback(req: FeedbackRequest):
    if not pending_client or not replay_client:
        return {"status": "error", "message": "Database not connected."}

    try:
        # 1. Retrieve the short-term memory
        entity = pending_client.get_entity(partition_key="LiveInteractions", row_key=req.interaction_id)
        
        # 2. 13-Hour Expiration Check Workaround
        saved_time = datetime.fromisoformat(entity["Timestamp_UTC"])
        if datetime.now(timezone.utc) - saved_time > timedelta(hours=13):
            pending_client.delete_entity(partition_key="LiveInteractions", row_key=req.interaction_id)
            return {"status": "ignored", "message": "Interaction expired."}
            
        # 3. Unpack the saved data
        old_state = json.loads(entity["StateVector"])
        action_taken = entity["ActionId"]
        
        # 4. Assign the reward based on engagement
        reward = 10.0 if req.engaged else -10.0
        
        # 5. Save experience to long-term storage for the nightly batch training
        replay_entity = {
            "PartitionKey": "BatchData",
            "RowKey": str(uuid.uuid4()),
            "State": json.dumps(old_state),
            "Action": int(action_taken),
            "Reward": float(reward),
            "NextState": json.dumps(old_state), # Using old_state as terminal state
            "Done": True,
            "Timestamp_UTC": datetime.now(timezone.utc).isoformat()
        }
        replay_client.create_entity(entity=replay_entity)
        
        # 6. Clean up the short-term database to save budget
        pending_client.delete_entity(partition_key="LiveInteractions", row_key=req.interaction_id)
        
        return {"status": "success", "message": f"Experience saved for batch training with reward: {reward}"}
        
    except Exception as e:
        raise HTTPException(status_code=404, detail="Interaction ID not found.")