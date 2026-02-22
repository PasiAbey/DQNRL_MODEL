from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.data.tables import TableServiceClient
from datetime import datetime, timezone, timedelta
import uuid
import json
import os
from rl_agent import RLAgent, get_rl_state_vector

app = FastAPI(title="New RL Platform API")

# Configure Azure Table Storage
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if CONNECTION_STRING:
    table_service = TableServiceClient.from_connection_string(conn_str=CONNECTION_STRING)
    table_client = table_service.create_table_if_not_exists(table_name="PendingInteractions")

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
    if CONNECTION_STRING:
        entity = {
            "PartitionKey": "LiveInteractions",
            "RowKey": interaction_id,
            "StateVector": json.dumps(state_vector.tolist()), 
            "ActionId": action_id,
            "Timestamp_UTC": datetime.now(timezone.utc).isoformat()
        }
        table_client.create_entity(entity=entity)
    
    return {"interaction_id": interaction_id, "action_id": action_id}

@app.post("/feedback")
async def receive_feedback(req: FeedbackRequest):
    if not CONNECTION_STRING:
        return {"status": "error", "message": "Database not connected."}

    try:
        # Retrieve the memory
        entity = table_client.get_entity(partition_key="LiveInteractions", row_key=req.interaction_id)
        
        # 13-Hour Expiration Check Workaround
        saved_time = datetime.fromisoformat(entity["Timestamp_UTC"])
        if datetime.now(timezone.utc) - saved_time > timedelta(hours=13):
            table_client.delete_entity(partition_key="LiveInteractions", row_key=req.interaction_id)
            return {"status": "ignored", "message": "Interaction expired."}
            
        # Unpack the saved data
        old_state = json.loads(entity["StateVector"])
        action_taken = entity["ActionId"]
        
        # Assign the reward based on engagement
        reward = 10.0 if req.engaged else -10.0
        
        # Save to agent memory for future retraining
        agent.remember(old_state, action_taken, reward, next_state=old_state, done=True)
        
        # Clean up database to save budget
        table_client.delete_entity(partition_key="LiveInteractions", row_key=req.interaction_id)
        
        return {"status": "success", "message": f"Learned with reward: {reward}"}
        
    except Exception as e:
        raise HTTPException(status_code=404, detail="Interaction ID not found.")