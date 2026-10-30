import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "lerobot/smolvla_base"

policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()

# Check what the model expects before building obs
print(policy.config.input_features)
print(policy.config.output_features)

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": device}},
)

obs = {
    "observation.images.camera1": torch.rand(1, 3, 480, 640),
    "observation.images.camera2": torch.rand(1, 3, 480, 640),
    "observation.images.camera3": torch.rand(1, 3, 480, 640),
    "observation.state": torch.zeros(1, 6),
    "task": "pick up the cube",
}

with torch.no_grad():
    batch = preprocess(obs)               # tokenizes task, normalizes, moves to device
    action = policy.select_action(batch)  # normalized action
    action = postprocess(action)          # unnormalized, real units

print(action.shape, action)