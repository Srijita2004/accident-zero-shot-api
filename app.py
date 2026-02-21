from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
from PIL import Image
import io
import torch

app = FastAPI()

MODEL_NAME = "google/siglip2-base-patch16-224"

device = 0 if torch.cuda.is_available() else -1
clf = pipeline("zero-shot-image-classification", model=MODEL_NAME, device=device)

ACCIDENT_PROMPTS = [
    # Road / vehicle accidents
    "a road traffic accident scene",
    "a car crash collision",
    "a highway pile-up accident",
    "a motorcycle crash accident",
    "a truck rollover accident",
    "a bus accident scene",
    "a pedestrian hit by a vehicle",
    "a rear-end collision accident",
    "a head-on collision accident",
    "a side-impact t-bone collision",

    # Fire & explosion accidents
    "a building fire accident with flames and smoke",
    "a vehicle fire accident",
    "an electrical fire accident",
    "an industrial explosion accident",
    "a gas cylinder blast accident",
    "a chemical explosion accident",

    # Industrial / construction accidents
    "a worker fall from height accident",
    "a construction site accident scene",
    "a scaffold collapse accident",
    "a crane accident scene",
    "a heavy object falling accident",
    "an electrical shock accident",

    # Human fall / medical emergency
    "a slip and fall accident",
    "an elderly fall accident",
    "a sudden collapse medical emergency",
    "a fainting incident",
    "a person lying unconscious emergency",

    # Water-related accidents
    "a drowning incident emergency",
    "a vehicle submerged in water accident",
    "a flood-related accident scene",
    "a boat collision accident",
]

NORMAL_PROMPTS = [
    # Normal road scenes (non-accident)
    "a normal road scene with traffic",
    "vehicles parked normally on the roadside",
    "a traffic jam with no accident",
    "cars slowing at a speed breaker",
    "night driving on a normal road",

    # Normal fire-like scenes (not accident)
    "a controlled bonfire",
    "a kitchen flame while cooking",
    "fireworks celebration",
    "a controlled industrial flame in a safe setting",

    # Normal human activities (non-accident)
    "a person sitting on the ground normally",
    "a person sleeping normally",
    "a person doing yoga pose",
    "a worker bending down to pick something safely",
    "people playing sports",

    # Industrial non-accident
    "normal machine operation in a factory",
    "safe construction work with safety gear",
    "a crane lifting materials safely",
    "workers wearing helmets and safety vests working normally",

    # General safe scenes
    "a normal scene with no accident",
    "a safe environment with no emergency",
]

def group_score(results, group):
    s = 0.0
    group = set(group)
    for r in results:
        if r["label"] in group:
            s += float(r["score"])
    return s

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    data = await image.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    labels = ACCIDENT_PROMPTS + NORMAL_PROMPTS
    out = clf(img, candidate_labels=labels)

    if isinstance(out, dict) and "labels" in out:
        results = [{"label": l, "score": s} for l, s in zip(out["labels"], out["scores"])]
    else:
        results = out

    accident_score = group_score(results, ACCIDENT_PROMPTS)
    normal_score = group_score(results, NORMAL_PROMPTS)

    pred = "accident" if accident_score >= normal_score else "non_accident"
    conf = float(accident_score / (accident_score + normal_score + 1e-9))

    return JSONResponse({
        "prediction": pred,
        "confidence": conf,
        "accident_score": float(accident_score),
        "normal_score": float(normal_score),
        "top_matches": results[:5]
    })
