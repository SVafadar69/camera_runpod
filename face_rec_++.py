face_api_key = 'GKnDCXeJb4N9lWZDWuBB2hkPnYqgkYWy'
face_api_secret = 'lG_QMEkrLLOUC5uZ1rzX2u6ysQwJC6p2'

import cv2
import base64
import asyncio
import aiohttp
import threading
import time
import os

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
FACEPP_API_KEY = face_api_key
FACEPP_API_SECRET = face_api_secret
FACEPP_COMPARE_URL = "https://api-us.faceplusplus.com/facepp/v3/compare"

REFERENCE_IMAGE_PATH = os.path.join(os.getcwd(), os.listdir("/home/ryanpc/Desktop/try_again_mein_face")[0])   # your known face
MAX_COMPARISONS = 5                      # total API hits


# ---------------------------------------------------------
# Helper: Encode image → Base64
# ---------------------------------------------------------
def img_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# ---------------------------------------------------------
# Async Face++ Compare Request
# ---------------------------------------------------------
async def compare_faces(session, ref_b64, frame_b64, idx):
    payload = {
        "api_key": FACEPP_API_KEY,
        "api_secret": FACEPP_API_SECRET,
        "image_base64_1": ref_b64,
        "image_base64_2": frame_b64
    }

    try:
        async with session.post(FACEPP_COMPARE_URL, data=payload, timeout=20) as resp:
            result = await resp.json()
            print(f"[COMPARE {idx}] Result → {result}")

    except Exception as e:
        print(f"[COMPARE {idx}] ERROR:", e)


# ---------------------------------------------------------
# Async worker that processes incoming frames
# ---------------------------------------------------------
async def async_face_worker(ref_b64, frame_queue):
    comparison_count = 0

    async with aiohttp.ClientSession() as session:
        while comparison_count < MAX_COMPARISONS:
            frame_b64 = await frame_queue.get()
            comparison_count += 1

            # launch async task
            asyncio.create_task(compare_faces(
                session, ref_b64, frame_b64, comparison_count
            ))

        print("Reached max comparison count. No more Face++ calls.")


# ---------------------------------------------------------
# Threaded wrapper to run asyncio loop separately from OpenCV
# ---------------------------------------------------------
def start_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


# ---------------------------------------------------------
# MAIN OPENCV LOOP
# ---------------------------------------------------------
def main():
    # Load reference image
    ref_img = cv2.imread(REFERENCE_IMAGE_PATH)
    ref_b64 = img_to_base64(ref_img)

    # Create asyncio loop for API requests
    loop = asyncio.new_event_loop()
    frame_queue = asyncio.Queue(loop=loop)

    # Start async event loop in background thread
    t = threading.Thread(target=start_async_loop, args=(loop,), daemon=True)
    t.start()

    # Start async worker inside event loop
    asyncio.run_coroutine_threadsafe(async_face_worker(ref_b64, frame_queue), loop)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("[INFO] Starting camera stream...")

    comparisons_sent = 0
    sent_limit = MAX_COMPARISONS

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break

        cv2.imshow("Live Feed", frame)

        # Send frame for face comparison (5 times max)
        if comparisons_sent < sent_limit:
            frame_b64 = img_to_base64(frame)

            # Push into async queue (non-blocking)
            asyncio.run_coroutine_threadsafe(frame_queue.put(frame_b64), loop)
            comparisons_sent += 1
            print(f"[INFO] Sent {comparisons_sent}/{sent_limit} frame for Face++ comparison")

        # Quit with q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    loop.stop()
    print("[INFO] Finished.")


if __name__ == "__main__":
    main()

