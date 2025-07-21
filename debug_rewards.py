from environment.pothole_env2 import VideoBasedPotholeEnv

env = VideoBasedPotholeEnv(split='train', verbose=True)

print("🧪 EMERGENCY REWARD SYSTEM TEST")
print("="*50)

rewards_received = []
for i in range(20):
    obs, info = env.reset()
    
    # Test all actions
    for action in range(5):
        obs, reward, done, truncated, info = env.step(action)
        rewards_received.append(reward)
        
        print(f"Action {action}: Reward = {reward}, "
              f"Confidence = {info.get('detection_confidence', 'N/A'):.3f}, "
              f"Ground Truth = {info.get('ground_truth_has_pothole', 'N/A')}")
        
        if done or truncated:
            break

print(f"\n📊 Rewards Summary:")
print(f"Unique rewards: {set(rewards_received)}")
print(f"Total +10: {rewards_received.count(10)}")
print(f"Total -5: {rewards_received.count(-5)}")
print(f"Total -20: {rewards_received.count(-20)}")
print(f"Total 0: {rewards_received.count(0)}")

if len(set(rewards_received)) == 1 and rewards_received[0] == 0:
    print("❌ CRITICAL: Reward system completely broken!")
else:
    print("✅ Reward system working!")
