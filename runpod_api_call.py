import os
import torch
import runpod
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pickle
import numpy as np
import time
import base64
import io

# -------------------------------------------------------------
# GLOBAL CONFIG
# -------------------------------------------------------------
_dir = os.getcwd()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'CUDA Available: {"Yes" if torch.cuda.is_available() else "No"}')
EMBED_PATH = f"{_dir}/steven_embeddings.pkl"
import urllib.request
urllib.request.urlretrieve('https://picsum.photos/800/600', 'downloaded_image.jpg')


# -------------------------------------------------------------
# LAZY LOAD (RunPod requirement)
# -------------------------------------------------------------
def lazy_load():
    """
    Loads models and embeddings once per container.
    Required for RunPod Serverless.
    """
    global mtcnn, resnet, steven_embeddings

    if "mtcnn" not in globals():
        print("[INIT] Loading MTCNN + ResNet models...")
        mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=DEVICE)
    if "resnet" not in globals():
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

    if "steven_embeddings" not in globals():
        print("[INIT] Loading Steven embeddings...")
        if os.path.exists(EMBED_PATH):
            with open(EMBED_PATH, "rb") as f:
                steven_embeddings = pickle.load(f)
            print(f"[INIT] Loaded {len(steven_embeddings)} embeddings.")
        else:
            return 'No pickle embeddings found - could not reach file'


# -------------------------------------------------------------
# Your original functions (UNCHANGED)
# -------------------------------------------------------------
def embed_image(image_path):
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)

    if face is None:
        return None

    face = face.unsqueeze(0).to(DEVICE)
    emb = resnet(face)
    return emb.detach().cpu().numpy()[0]


def load_steven_embeddings():
    if not os.path.exists(EMBED_PATH):
        raise FileNotFoundError(f"Embeddings file not found at {EMBED_PATH}")

    with open(EMBED_PATH, "rb") as f:
        steven_embeddings = pickle.load(f)

    print(f"[INFO] Loaded {len(steven_embeddings)} Steven embeddings")
    return steven_embeddings


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def is_steven(emb, steven_embeddings, STEVEN_THRESHOLD: float = 0.3):
    best_score = -1
    for ref in steven_embeddings:
        score = cosine_similarity(emb, ref)
        best_score = max(best_score, score)

    return best_score >= STEVEN_THRESHOLD, best_score


# -------------------------------------------------------------
# RUNPOD HANDLER (only updated to use lazy-loaded objects)
# -------------------------------------------------------------
def runpod_inference(event):
    try:
        lazy_load()  # <-- RunPod serverless requirement

        # Extract input
        job_input = event.get("input", {})
        if not job_input:
            return {"error": "No input provided"}

        image_data = job_input.get("image", "downloaded_image.jpg")
        threshold = job_input.get("threshold", 0.3)

        if not image_data:
            return {"error": "No image provided in input"}

        # Decode base64
        if isinstance(image_data, str):
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            return {"error": "Invalid image format"}

        frame = np.array(image)
        results = []

        # Detect faces
        face_detect_time = time.time()
        boxes, probs = mtcnn.detect(frame)
        face_detect_duration = time.time() - face_detect_time
        print(f'Took {face_detect_duration:.4f} seconds to detect faces')

        if boxes is not None:
            face_embeddings_time = time.time()
            faces = mtcnn(frame)
            face_embed_duration = time.time() - face_embeddings_time
            print(f'Took {face_embed_duration:.4f} seconds to embed faces')

            if faces is not None:
                if faces.ndim == 3:
                    faces = faces.unsqueeze(0)

                for idx, box in enumerate(boxes):
                    if idx >= len(faces):
                        continue

                    f_tensor = faces[idx]
                    fac_rec_time = time.time()

                    with torch.no_grad():
                        emb = resnet(f_tensor.unsqueeze(0).to(DEVICE)).detach().cpu().numpy()[0]
                        is_steven_flag, score = is_steven(emb, steven_embeddings, threshold)
                        fac_rec_duration = time.time() - fac_rec_time
                        print(f'Took {fac_rec_duration:.4f} seconds for face recognition')

                        result = {
                            "detected_face_boxes": [float(x) for x in box],
                            "is_steven":is_steven_flag,
                            "confidence": score,
                            "probability": probs[idx]
                        }
                        results.append(result)

        return {
            "faces": results
        }

    except Exception as e:
        print(f"[ERROR] Error in runpod_inference: {str(e)}")
        return {"error": str(e)}


# -------------------------------------------------------------
# START RUNPOD SERVERLESS
# -------------------------------------------------------------
if __name__ == "__main__":
    runpod.serverless.start({"handler": runpod_inference})
