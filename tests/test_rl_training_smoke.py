import os
import shutil
import pytest
from src.trainers.train_rl import train, load_config

def test_rl_training_smoke():
    """
    Runs a short training session to ensure no crashes.
    """
    # Create temp config
    config = load_config("configs/rl.yaml")
    config['training']['total_timesteps'] = 1000 # Short run
    config['training']['log_dir'] = "logs/test_rl"
    config['training']['model_dir'] = "models/test_rl"
    
    # Save temp config
    import yaml
    with open("configs/test_rl.yaml", 'w') as f:
        yaml.dump(config, f)
        
    # Patch train to use valid config path mechanism or just patch load_config?
    # Actually train() calls load_config() with default or expects arg?
    # train_rl.py has hardcoded "configs/rl.yaml" default in make_env.
    # We should probably refactor train_rl to accept config path, but for now let's strict check.
    # To avoid changing src too much, let's just make sure "configs/rl.yaml" is valid (it is).
    # But we want to run SHORT training.
    
    # Override by monkeypatching load_config in the module?
    from src.trainers import train_rl
    original_load = train_rl.load_config
    
    def mock_load(path="configs/rl.yaml"):
        # return our short config regardless of path
        return config
        
    train_rl.load_config = mock_load
    
    try:
        train_rl.train()
    except Exception as e:
        pytest.fail(f"Training failed with error: {e}")
    finally:
        # Cleanup
        if os.path.exists("logs/test_rl"):
            shutil.rmtree("logs/test_rl")
        if os.path.exists("models/test_rl"):
            shutil.rmtree("models/test_rl")
        if os.path.exists("configs/test_rl.yaml"):
            os.remove("configs/test_rl.yaml")
        if os.path.exists("vec_normalize.pkl"):
            os.remove("vec_normalize.pkl")
            
        train_rl.load_config = original_load
