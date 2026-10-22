from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from .config import settings
from .db.mongodb import connect_to_mongo, close_mongo_connection
from .api.upload import router as upload_router
from .api.inference import router as inference_router
from .api.jobs import router as jobs_router

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Deepfake Detection API", version="0.1.0")

origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
	await connect_to_mongo()
	logger.info("Deepfake Detection API started")
	logger.info(f"Face detection enabled: {settings.USE_FACE_DETECTION}")
	if settings.USE_FACE_DETECTION:
		logger.info(f"Face detection settings: min_size={settings.MIN_FACE_SIZE}, "
				   f"confidence={settings.FACE_DETECTION_CONFIDENCE}, "
				   f"padding={settings.FACE_PADDING}")


@app.on_event("shutdown")
async def shutdown_event():
	await close_mongo_connection()


@app.get("/")
async def root():
	return {"status": "ok"}


app.include_router(upload_router)
app.include_router(inference_router)
app.include_router(jobs_router)








