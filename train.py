import os
import json
import uuid
import torch
from datetime import datetime, timezone, timedelta
from azure.data.tables import TableServiceClient
from rl_agent import RLAgent

# Configuration
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
MODEL_PATH = "trained_rl_agent.pth"
MAX_MEMORIES = 2000  # The maximum number of past experiences to keep in Azure

def run_nightly_training():
    print("=" * 50)
    print("🚀 Starting Nightly Batch Training & Auto-Sweep Process")
    print("=" * 50)

    if not CONNECTION_STRING:
        print("❌ Error: AZURE_STORAGE_CONNECTION_STRING is not set.")
        return

    try:
        # 1. Connect to Azure Table Storage
        table_service = TableServiceClient.from_connection_string(conn_str=CONNECTION_STRING)
        pending_client = table_service.get_table_client(table_name="PendingInteractions")
        replay_client = table_service.get_table_client(table_name="ExperienceReplay")
    except Exception as e:
        print(f"❌ Error connecting to Azure: {e}")
        return

    # ==========================================
    # STEP 1: THE AUTO-SWEEP (12-Hour Timeout)
    # ==========================================
    print("🧹 STEP 1: Sweeping expired interactions (12+ hours old)...")
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=12)
    swept_count = 0

    try:
        for entity in pending_client.list_entities():
            entity_time_str = entity.get("Timestamp_UTC")
            if not entity_time_str:
                continue
            
            try:
                entity_time = datetime.fromisoformat(entity_time_str)
            except ValueError:
                continue 

            # If it's older than 12 hours, punish it with -10.0
            if entity_time < cutoff_time:
                try:
                    state_vector_str = entity["StateVector"]
                    action_id = entity["ActionId"]

                    replay_entity = {
                        "PartitionKey": "BatchData",
                        "RowKey": str(uuid.uuid4()),
                        "State": state_vector_str,
                        "Action": int(action_id),
                        "Reward": -10.0,
                        "NextState": state_vector_str,
                        "Done": True,
                        "Timestamp_UTC": datetime.now(timezone.utc).isoformat()
                    }
                    replay_client.create_entity(entity=replay_entity)
                    pending_client.delete_entity(partition_key=entity["PartitionKey"], row_key=entity["RowKey"])
                    swept_count += 1
                except Exception as e:
                    print(f"⚠️ Error processing expired row {entity.get('RowKey')}: {e}")

        print(f"✅ Swept {swept_count} ignored interactions and marked them as failures (-10.0).")
    except Exception as e:
        print(f"⚠️ Error during auto-sweep: {e}")

    # ==========================================
    # STEP 2: FETCH & TRAIN
    # ==========================================
    print("📡 STEP 2: Fetching experiences from Azure Table Storage...")
    try:
        entities = list(replay_client.list_entities())
    except Exception as e:
        print(f"❌ Error reading table: {e}")
        return

    if not entities:
        print("💤 No new experiences found today. Skipping training.")
        return

    print(f"📊 Found {len(entities)} total experiences in the memory bank.")

    # Initialize Agent and Load Current Brain
    agent = RLAgent()
    if os.path.exists(MODEL_PATH):
        agent.load_pretrained_model(MODEL_PATH)
        print(f"🧠 Loaded existing model from {MODEL_PATH}")
    else:
        print("⚠️ No existing model found. Training from scratch.")

    # Load ALL Data into the Agent's Short-Term Memory
    for entity in entities:
        try:
            state = json.loads(entity["State"])
            action = entity["Action"]
            reward = float(entity["Reward"])
            next_state = json.loads(entity["NextState"])
            done = entity["Done"]
            agent.remember(state, action, reward, next_state, done)
        except Exception as e:
            print(f"⚠️ Skipping corrupted row {entity.get('RowKey')}: {e}")

    if not agent.memory:
        print("⚠️ No valid memories to train on.")
        return

    # Train the Neural Network using Random Sampling from the Buffer
    batch_size = min(32, len(agent.memory))
    # Run training proportional to how much data we have (approx 1 epoch)
    num_updates = max(1, len(entities) // batch_size) 
    print(f"⚙️ Running {num_updates} training batches (Size: {batch_size})...")
    
    for _ in range(num_updates):
        agent.replay(batch_size=batch_size)

    # Save the Updated Brain
    print(f"💾 Saving updated model weights to {MODEL_PATH}...")
    torch.save({
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon
    }, MODEL_PATH)

    # ==========================================
    # STEP 3: MANAGE THE ROLLING BUFFER
    # ==========================================
    print("🧹 STEP 3: Managing Experience Replay database size...")
    
    # Sort all experiences by their timestamp (oldest first)
    entities.sort(key=lambda x: x.get("Timestamp_UTC", ""))
    
    if len(entities) > MAX_MEMORIES:
        excess_count = len(entities) - MAX_MEMORIES
        rows_to_delete = entities[:excess_count] # Grab the oldest rows
        deleted_count = 0
        for row in rows_to_delete:
            try:
                replay_client.delete_entity(partition_key=row["PartitionKey"], row_key=row["RowKey"])
                deleted_count += 1
            except Exception as e:
                print(f"⚠️ Failed to delete old row: {e}")
        print(f"✅ Cleaned up {deleted_count} old memories. Kept the freshest {MAX_MEMORIES}.")
    else:
        print(f"✅ Database has {len(entities)} memories. No cleanup needed yet.")

    print("🎉 Nightly training complete! The AI is now smarter and the database is optimized.")
    print("=" * 50)

if __name__ == "__main__":
    run_nightly_training()