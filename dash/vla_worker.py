"""SmolVLA service scaffold. /predict stays disabled until a TIAGo adapter is implemented."""
import base64, io, os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

MODEL_ID=os.getenv("VLA_MODEL_ID","lerobot/smolvla_base")
DEVICE=os.getenv("VLA_DEVICE","cuda" if torch.cuda.is_available() else "cpu")
app=FastAPI(title="TIAGo VLA Service")
policy=None
load_error=None

class Request(BaseModel):
    task:str
    telemetry:dict
    image_base64:str
    image_mime_type:str="image/jpeg"

@app.on_event("startup")
def startup():
    global policy,load_error
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy=SmolVLAPolicy.from_pretrained(MODEL_ID).to(DEVICE).eval()
    except Exception as e: load_error=f"{type(e).__name__}: {e}"

@app.get("/health")
def health():
    return {"model_id":MODEL_ID,"device":DEVICE,"model_loaded":policy is not None,"load_error":load_error,"tiago_adapter_configured":False}

@app.get("/model-info")
def model_info():
    if policy is None: raise HTTPException(503,detail=load_error or "Model not loaded")
    def info(fs):
        return {str(k):{"type":str(getattr(v,"type","")),"shape":list(getattr(v,"shape",[]))} for k,v in (fs or {}).items()}
    return {"model_id":MODEL_ID,"device":DEVICE,"input_features":info(getattr(policy.config,"input_features",{})),"output_features":info(getattr(policy.config,"output_features",{}))}

@app.post("/predict")
def predict(req:Request):
    if policy is None: raise HTTPException(503,detail=load_error or "Model not loaded")
    try:
        raw=base64.b64decode(req.image_base64,validate=True)
        im=Image.open(io.BytesIO(raw)).convert("RGB"); im.load()
    except Exception as e: raise HTTPException(400,detail=f"Invalid image: {e}")
    # Intentionally do not invent a state/action mapping.
    raise HTTPException(501,detail={"message":"Model is loaded, but TIAGo observation/action adapter is not configured.","image_size":list(im.size),"telemetry_keys":list(req.telemetry.keys()),"next_step":"Inspect /model-info and implement the TIAGo adapter for this checkpoint."})
