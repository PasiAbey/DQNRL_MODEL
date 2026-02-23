import os
import json
import torch
from azure.data.tables import TableServiceClient
from rl_agent import RLAgent

# Configuration
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
TABLE_NAME = "ExperienceReplay"
MODEL_PATH = "trained_rl_agent.pth"

def run_nightly_training():
    print("=" * 50)
    print("🚀 Starting Nightly Batch Training Process")
    print("=" * 50)

    if not CONNECTION_STRING:
        print("❌ Error: AZURE_STORAGE_CONNECTION_STRING is not set.")
        return

    try:
        # 1. Connect to Azure Table Storage
        table_service = TableServiceClient.from_connection_string(conn_str=CONNECTION_STRING)
        replay_client = table_service.get_table_client(table_name=TABLE_NAME)
    except Exception as e:
        print(f"❌ Error connecting to Azure: {e}")
        return

    print("📡 Fetching experiences from Azure Table Storage...")
    try:
        # Fetch all saved experiences
        entities = list(replay_client.list_entities())
    except Exception as e:
        print(f"❌ Error reading table (it may be empty or not exist yet): {e}")
        return

    if not entities:
        print("💤 No new experiences found today. Skipping training.")
        return

    print(f"📊 Found {len(entities)} new experiences. Initializing RL Agent...")

    # 2. Initialize Agent and Load Current Brain
    agent = RLAgent()
    if os.path.exists(MODEL_PATH):
        agent.load_pretrained_model(MODEL_PATH)
        print(f"🧠 Loaded existing model from {MODEL_PATH}")
    else:
        print("⚠️ No existing model found. Training from scratch.")

    # 3. Load Data into the Agent's Short-Term Memory
    processed_rows = []
    for entity in entities:
        try:
            state = json.loads(entity["State"])
            action = entity["Action"]
            reward = entity["Reward"]
            next_state = json.loads(entity["NextState"])
            done = entity["Done"]

            agent.remember(state, action, reward, next_state, done)
            processed_rows.append(entity)
        except Exception as e:
            print(f"⚠️ Skipping corrupted row {entity.get('RowKey')}: {e}")

    # 4. Train the Neural Network
    # We dynamically set the batch size. If we only have 10 rows today, we batch 10.
    batch_size = min(32, len(agent.memory))
    
    # If we had a busy day on Skill-Quest, we run the training loop multiple times
    num_updates = max(1, len(processed_rows) // batch_size)
    print(f"⚙️ Running {num_updates} training epoch(s) with batch size {batch_size}...")
    
    for _ in range(num_updates):
        agent.replay(batch_size=batch_size)

    # 5. Save the Updated Brain
    print(f"💾 Saving updated model weights to {MODEL_PATH}...")
    torch.save({
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon
    }, MODEL_PATH)

    # 6. Clean Up the Database (Crucial for the free tier)
    print("🧹 Cleaning up processed experiences from Azure...")
    deleted_count = 0
    for row in processed_rows:
        try:
            replay_client.delete_entity(partition_key=row["PartitionKey"], row_key=row["RowKey"])
            deleted_count += 1
        except Exception as e:
            print(f"⚠️ Failed to delete row {row['RowKey']}: {e}")

    print(f"✅ Successfully deleted {deleted_count} rows.")
    print("🎉 Nightly training complete! The AI is now smarter.")
    print("=" * 50)

if __name__ == "__main__":
    run_nightly_training()