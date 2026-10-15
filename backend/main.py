from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from predictor import DogBreedPredictor

MAX_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models once when the server starts
    state["predictor"] = DogBreedPredictor()
    print("Models loaded. Server ready.")
    yield
    state.clear()


app = FastAPI(title="AI-Powered Dog Breed Detection", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        return JSONResponse(
            status_code=400,
            content={"status": "error",
                     "message": "Unsupported file type. Please upload a JPG, PNG or WEBP image."},
        )

    data = await file.read()
    if len(data) == 0:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "The uploaded file is empty."},
        )
    if len(data) > MAX_BYTES:
        return JSONResponse(
            status_code=413,
            content={"status": "error", "message": "The file is too large. Maximum size is 10 MB."},
        )

    return state["predictor"].predict(data)