from fastapi import FastAPI, status, File, Form, UploadFile, Cookie
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import torch
import zipfile
import numpy as np
from io import BytesIO
from PIL import Image
from base64 import b64encode, b64decode
from pydantic import BaseModel

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

sam_checkpoint = "sam_vit_h_4b8939.pth"  # "sam_vit_h_4b8939.pth" or "sam_vit_l_0b3195.pth" or "sam_vit_b_01ec64.pth
model_type = "vit_h"  # "vit_l" or "vit_h"
device = "cuda:3"

print("Loading model")
SAM = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
print("Finishing loading")

predictor = SamPredictor(SAM)

app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

segmented_mask = []
interactive_mask = []
mask_input = None

GLOBAL_IMAGE = None
GLOBAL_MASK = None
GLOBAL_ZIPBUFFER = None

sessions = {}

class SessionData(BaseModel):
    image: np.ndarray
    mask_input: np.ndarray
    segmented_mask: list
    interactive_mask: list
    zip_buffer: BytesIO

@app.post("/image")
async def process_images(
    image: UploadFile = File(...),
    session_id: str = Cookie(None)
):
    global segmented_mask, interactive_mask
    global GLOBAL_IMAGE, GLOBAL_MASK, GLOBAL_ZIPBUFFER
    import os
    f = "./embedding_cached.pt"

    segmented_mask = []
    interactive_mask = []

    # Read the image and mask data as bytes
    image_data = await image.read()
    image_data = BytesIO(image_data)
    img = np.array(Image.open(image_data))
    print("get image", img.shape)
    GLOBAL_IMAGE = img[:, :, :-1]
    GLOBAL_MASK = None
    GLOBAL_ZIPBUFFER = None

    session = sessions.get(session_id)
    if session is None:
        session = SessionData(
            image=GLOBAL_IMAGE,
            mask_input=None,
            segmented_mask=[],
            interactive_mask=[],
            zip_buffer=None
        )
        sessions[session_id] = session
    else:
        session.image = GLOBAL_IMAGE
        session.mask_input = None
        session.segmented_mask = []
        session.interactive_mask = []
        session.zip_buffer = None

    # produce an image embedding by calling SamPredictor.set_image
    predictor.set_image(GLOBAL_IMAGE)

    # Return a JSON response
    return JSONResponse(
        content={
            "message": "Images received successfully",
        },
        status_code=101,
        headers={
            "Set-Cookie": f"session_id={session_id}; Path=/",
        },
    )

@app.post("/undo")
async def undo_mask(
    session_id: str = Cookie(None)
):
    session = sessions.get(session_id)
    if session is not None:
        session.segmented_mask.pop()

    return JSONResponse(
        content={
            "message": "Clear successfully",
        },
        status_code=500,
    )

from fastapi import Request
@app.post("/click")
async def click_images(
    request: Request,
    session_id: str = Cookie(None)
):
    session = sessions.get(session_id)
    if session is None:
        return JSONResponse(
            content={
                "message": "Session expired",
            },
            status_code=401,
        )

    form_data = await request.form()
    type_list = [int(i) for i in form_data.get("type").split(',')]
    click_list = [int(i) for i in form_data.get("click_list").split(',')]

    point_coords = np.array(click_list, np.float32).reshape(-1, 2)
    point_labels = np.array(type_list).reshape(-1)

    if len(point_coords) == 1:
        session.mask_input = None

    masks_, scores_, logits_ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=session.mask_input,
        multimask_output=True,
    )

    best_idx = np.argmax(scores_)
    res = masks_[best_idx]
    session.mask_input = logits_[best_idx][None, :, :]

    len_prompt = len(point_labels)
    len_mask = len(session.interactive_mask)
    if len_mask == 0 or len_mask < len_prompt:
        session.interactive_mask.append(res)
    else:
        session.interactive_mask[len_prompt - 1] = res

    # Return a JSON response
    res = Image.fromarray(res)
    return JSONResponse(
        content={
            "masks": pil_image_to_base64(res),
            "message": "Images processed successfully"
        },
        status_code=100,
    )

@app.post("/finish_click")
async def finish_interactive_click(
    mask_idx: int = Form(...),
    session_id: str = Cookie(None)
):
    session = sessions.get(session_id)
    if session is None:
        return JSONResponse(
            content={
                "message": "Session expired",
            },
            status_code=401,
        )

    session.segmented_mask.append(session.interactive_mask[mask_idx])
    session.interactive_mask = list()

    return JSONResponse(
        content={
            "message": "Finish successfully",
        },
        status_code=200,
    )

@app.post("/rect")
async def rect_images(
    start_x: int = Form(...),
    start_y: int = Form(...),
    end_x: int = Form(...),
    end_y: int = Form(...),
    session_id: str = Cookie(None)
):
    session = sessions.get(session_id)
    if session is None:
        return JSONResponse(
            content={
                "message": "Session expired",
            },
            status_code=401,
        )

    masks_, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array([[start_x, start_y, end_x, end_y]]),
        multimask_output=False
    )

    res = Image.fromarray(masks_[0])
    print(masks_[0].shape)

    return JSONResponse(
        content={
            "masks": pil_image_to_base64(res),
            "message": "Images processed successfully"
        },
        status_code=300,
    )

class PointsGenerator(BaseModel):
    points_per_side: str
    points_per_batch: str

@app.post("/everything")
async def seg_everything(
    params: PointsGenerator,
    session_id: str = Cookie(None)
):
    session = sessions.get(session_id)
    if session is None:
        return JSONResponse(
            content={
                "message": "Session expired",
            },
            status_code=401,
        )

    params = dict(params)
    pps = int(params['points_per_side'])
    ppb = int(params['points_per_batch'])

    mask_generator = SamAutomaticMaskGenerator(
        model=SAM,
        points_per_side=pps,
        points_per_batch=ppb
    )

    masks = mask_generator.generate(session.GLOBAL_IMAGE)
    assert len(masks) > 0, "No masks found"

    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    print(len(sorted_anns))

    img = np.zeros(
        (
            sorted_anns[0]['segmentation'].shape[0],
            sorted_anns[0]['segmentation'].shape[1]
        ),
        dtype=np.uint8
    )
    for idx, ann in enumerate(sorted_anns, 0):
        img[ann['segmentation']] = idx % 255 + 1

    res = Image.fromarray(img)
    session.GLOBAL_MASK = res

    zip_buffer = BytesIO()
    PIL_GLOBAL_IMAGE = Image.fromarray(session.GLOBAL_IMAGE)
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, ann in enumerate(sorted_anns, 0):
            left, top, right, bottom = (
                ann["bbox"][0],
                ann["bbox"][1],
                ann["bbox"][0] + ann["bbox"][2],
                ann["bbox"][1] + ann["bbox"][3]
            )
            cropped = PIL_GLOBAL_IMAGE.crop((left, top, right, bottom))

            transparent = Image.new("RGBA", cropped.size, (0, 0, 0, 0))

            mask = Image.fromarray(ann["segmentation"].astype(np.uint8) * 255)
            mask_cropped = mask.crop((left, top, right, bottom))

            result = Image.composite(
                cropped.convert("RGBA"),
                transparent,
                mask_cropped
            )

            result_bytes = BytesIO()
            result.save(result_bytes, format="PNG")
            result_bytes.seek(0)
            zip_file.writestr(f"seg_{idx}.png", result_bytes.read())

    zip_buffer.seek(0)
    session.GLOBAL_ZIPBUFFER = zip_buffer

    return JSONResponse(
        content={
            "masks": pil_image_to_base64(session.GLOBAL_MASK),
            "zipfile": b64encode(session.GLOBAL_ZIPBUFFER.getvalue()).decode("utf-8"),
            "message": "Images processed successfully"
        },
        status_code=400,
    )

@app.get("/assets/{path}/{file_name}", response_class=FileResponse)
async def read_assets(path, file_name):
    return f"assets/{path}/{file_name}"

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return read_content('segDrawer.html')

import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)

class Session:
    def __init__(self, GLOBAL_IMAGE):
        self.GLOBAL_IMAGE = GLOBAL_IMAGE
        self.GLOBAL_MASK = None
        self.GLOBAL_ZIPBUFFER = None
        self.segmented_mask = []
        self.interactive_mask = []
        self.mask_input = None

sessions = {}


@app.post("/image")
async def process_images(
    image: UploadFile = File(...),
    session_id: str = Cookie(None)
):
    global segmented_mask, interactive_mask
    global GLOBAL_IMAGE, GLOBAL_MASK, GLOBAL_ZIPBUFFER

    segmented_mask = []
    interactive_mask = []

    # Read the image data as bytes
    image_data = await image.read()
    image_data = BytesIO(image_data)
    img = np.array(Image.open(image_data))
    GLOBAL_IMAGE = img[:, :, :-1]
    GLOBAL_MASK = None
    GLOBAL_ZIPBUFFER = None

    session = sessions.get(session_id)
    if session is None:
        session = Session(GLOBAL_IMAGE)
        sessions[session_id] = session
    else:
        session.GLOBAL_IMAGE = GLOBAL_IMAGE
        session.GLOBAL_MASK = None
        session.GLOBAL_ZIPBUFFER = None
        session.segmented_mask = []
        session.interactive_mask = []
        session.mask_input = None

    # produce an image embedding by calling SamPredictor.set_image
    predictor.set_image(GLOBAL_IMAGE)

    # Return a JSON response
    return JSONResponse(
        content={
            "message": "Images received successfully",
        },
        status_code=101,
        headers={
            "Set-Cookie": f"session_id={session_id}; Path=/",
        },
    )
