import os
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Global variables for model artifact
MODEL_PATH = os.path.join("models", "delay_model.joblib")
model_artifact = None


# Modern FastAPI Lifespan Handler (Replaces @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup loading and shutdown teardown for microservice resources."""
    global model_artifact
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run 'python src/train.py' first."
        )

    model_artifact = joblib.load(MODEL_PATH)
    print("Model artifact successfully loaded into FastAPI application!")
    yield
    # Cleanup logic on application shutdown
    model_artifact = None


# Initialize FastAPI with lifespan context
app = FastAPI(
    title="E-Commerce Delivery Delay Risk API",
    description="Microservice predicting shipping delay risk based on order spatial and physical properties.",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic V2 Schema Definition
class OrderPredictionInput(BaseModel):
    total_freight_value: float = Field(
        ...,
        json_schema_extra={"example": 18.50},
        description="Total freight shipping charge in BRL",
    )
    total_item_price: float = Field(
        ...,
        json_schema_extra={"example": 120.00},
        description="Total cost of items in cart",
    )
    num_items: int = Field(
        ...,
        ge=1,
        json_schema_extra={"example": 1},
        description="Number of distinct items in order",
    )
    product_weight_g: float = Field(
        ...,
        json_schema_extra={"example": 800.0},
        description="Average product weight in grams",
    )
    product_volume_cm3: float = Field(
        ...,
        json_schema_extra={"example": 3000.0},
        description="Product volumetric footprint in cm^3",
    )
    distance_km: float = Field(
        ...,
        json_schema_extra={"example": 450.2},
        description="Haversine distance between customer and seller",
    )
    is_same_state: int = Field(
        ...,
        ge=0,
        le=1,
        json_schema_extra={"example": 0},
        description="1 if customer and seller in same state, else 0",
    )
    estimated_days: float = Field(
        ...,
        json_schema_extra={"example": 14.0},
        description="Promised shipping duration set at checkout",
    )
    purchase_dow: int = Field(
        ...,
        ge=0,
        le=6,
        json_schema_extra={"example": 4},
        description="Day of week order was placed (0=Mon, 6=Sun)",
    )
    purchase_hour: int = Field(
        ...,
        ge=0,
        le=23,
        json_schema_extra={"example": 14},
        description="Hour of purchase (0-23)",
    )


class PredictionResponse(BaseModel):
    delay_probability: float
    is_high_risk: bool
    risk_level: str


@app.get("/")
def health_check():
    """Health check endpoint to verify microservice status."""
    return {"status": "online", "model_loaded": model_artifact is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_delay_risk(payload: OrderPredictionInput):
    """Predicts delivery delay risk probability for an incoming order payload."""
    if model_artifact is None:
        raise HTTPException(
            status_code=500, detail="Model artifact is not loaded."
        )

    try:
        # Pydantic V2 uses payload.model_dump() instead of payload.dict()
        input_data = pd.DataFrame([payload.model_dump()])

        # Ensure feature column ordering matches training schema
        feature_names = model_artifact["feature_names"]
        input_data = input_data[feature_names]

        # Score probability with saved model
        model = model_artifact["model"]
        delay_prob = float(model.predict_proba(input_data)[0, 1])

        is_high_risk = delay_prob >= 0.40

        if delay_prob < 0.25:
            risk_level = "LOW"
        elif delay_prob < 0.45:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return PredictionResponse(
            delay_probability=round(delay_prob, 4),
            is_high_risk=is_high_risk,
            risk_level=risk_level,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))