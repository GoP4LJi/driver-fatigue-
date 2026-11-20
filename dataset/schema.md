# Driver Fatigue and Distraction Dataset Schema

## Overview
This document defines the formal tabular feature schema used for training, validating, and benchmarking machine learning classifiers in the Driver Fatigue and Distraction Detection System.

## Data Schema Specification

Each record in `dataset/driver_features.csv` represents a single discrete video frame analysis vector.

| Column Name | Data Type | Permissible Range | Typical Alert Range | Description |
|:---|:---|:---|:---|:---|
| `timestamp` | `float` (Unix epoch) | `> 0.0` | N/A | Monotonic or epoch timestamp in seconds when the frame was acquired. |
| `ear` | `float` | `0.00` to `0.50` | `0.25` - `0.38` | Bilateral average Eye Aspect Ratio (EAR). Values `< 0.25` signify eye closure. |
| `mar` | `float` | `0.00` to `1.50` | `0.05` - `0.45` | Mouth Aspect Ratio (MAR). Values `> 0.60` signify mouth opening / yawning. |
| `pitch` | `float` (degrees) | `-90.0` to `+90.0` | `-10.0` to `+10.0` | Head pitch angle. Positive = nodding downward (slumping); negative = tilting upward. |
| `yaw` | `float` (degrees) | `-90.0` to `+90.0` | `-15.0` to `+15.0` | Head yaw angle. Positive = turning right; negative = turning left. |
| `roll` | `float` (degrees) | `-90.0` to `+90.0` | `-15.0` to `+15.0` | Head roll angle. Positive = tilting clockwise; negative = counter-clockwise. |
| `state` | `string` (enum) | `ALERT`, `DROWSY`, `DISTRACTED` | `ALERT` | Ground-truth physiological and behavioral driver state label. |

## Target Class Definitions

1. **`ALERT`**:
   - The driver is attentive, eyes open (`ear >= 0.25`), mouth closed/neutral (`mar <= 0.50`), and head oriented forward facing the road (`|pitch| <= 15°`, `|yaw| <= 20°`, `|roll| <= 20°`).

2. **`DROWSY`**:
   - The driver exhibits physiological markers of fatigue or sleepiness:
     - Prolonged eye closure (`ear < 0.25`).
     - Frequent or sustained yawning (`mar > 0.60`).
     - Head slumping forward / nodding down (`pitch > 15°`).

3. **`DISTRACTED`**:
   - The driver is not looking at the road ahead:
     - Head turned away (`|yaw| > 20°`), looking at passengers, mirrors, or side windows.
     - Head tilted excessively down (`pitch > 20°`) or up (`pitch < -20°`), e.g., viewing mobile devices or center console.

## Dataset Limitations & Ethics Note
- **Synthetic Data Disclaimer**: Synthetic datasets generated for testing the software pipeline must **never** be presented as real-world validated training data.
- **Genuine Data Requirements**: Real-world deployment models must be trained on ethically collected, diverse driver benchmark datasets (e.g., NTHU-Driver-Drowsiness-Detection or YawDD) capturing diverse lighting conditions, ethnicities, facial hair, and eyewear.
