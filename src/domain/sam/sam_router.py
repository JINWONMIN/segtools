from fastapi import APIRouter, File, Form, UploadFile, Request
from fastapi.responses import JSONResponse
from segment_anything import SamAutomaticMaskGenerator
import zipfile
import numpy as np
from io import BytesIO
from PIL import Image
from base64 import b64encode

from domain.sam.sam_schema import PointsGenerator
from domain.utils.utils import pil_image_to_base64
from app import predictor_0, predictor_1, predictor_2, predictor_3, sessions, SAM_3


router = APIRouter(
    prefix="/api/sam"
)


@router.post("/image")
async def process_images(
    image: UploadFile = File(...),
    session_id: str = Form(...),
):
    global sessions
    import os
    f = "./embedding_cached.pt"

    # Read the image and mask data as bytes
    image_data = await image.read()
    image_data = BytesIO(image_data)
    img = np.array(Image.open(image_data))
    print("get image", img.shape)
    GLOBAL_IMAGE = img[:,:,:-1]
    
    session = sessions.get(session_id)
    if session is None:
        session = {
            "image": GLOBAL_IMAGE,
            "mask_input": None,
            "segmented_mask": [],
            "interactive_mask": [],
            "zip_buffer": None,
            "global_mask": None,
            "features": None,
            "res": None,
            "original_size": None,
            "input_size": None
        }
        sessions[session_id] = session
        predictor_3.set_image(sessions[session_id]['image'])
        sessions[session_id]['features'] = predictor_3.features
        sessions[session_id]['original_size'] = predictor_3.original_size
        sessions[session_id]['input_size'] = predictor_3.input_size
        
        
    # produce an image embedding by calling SamPredictor.set_image
    # if os.path.exists(cache_path):
    #     predictor.load_embedding(cache_path)
    #     print(predictor.is_image_set)
    # else:
    #     predictor.set_image(GLOBAL_IMAGE)
    #     predictor.save_embedding(cache_path)
    #     print("finish setting image")
    
    # Return a JSON response
    return JSONResponse(
        content={
            "message": "Images received successfully",
        },
        status_code=200,
    )
    

@router.post("/undo")
async def undo_mask(
    session_id: str = Form(...),
):
    global sessions
    sessions[session_id]["segmented_mask"].pop()

    return JSONResponse(
        content={
            "message": "Clear successfully",
        },
        status_code=200,
    )
    
    
@router.post("/click")
async def click_images(
    request: Request,
    session_id: str = Form(...),
):  
    global sessions

    session = sessions.get(session_id)
    
    form_data = await request.form()
    type_list = [int(i) for i in form_data.get("type").split(',')]
    click_list = [int(i) for i in form_data.get("click_list").split(',')]

    point_coords = np.array(click_list, np.float32).reshape(-1, 2)
    point_labels = np.array(type_list).reshape(-1)
    
    if (len(point_coords) == 1):
        session['mask_input'] = None
    predictor_3.features = session['features']
    predictor_3.original_size = session['original_size']
    predictor_3.input_size = session['input_size']
    
    masks_, scores_, logits_ = predictor_3.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=session['mask_input'],
        multimask_output=True,
    )

    best_idx = np.argmax(scores_)
    res = masks_[best_idx]
    mask_input = logits_[best_idx][None, :, :]

    len_prompt = len(point_coords)
    len_mask = len(session['interactive_mask'])
    if len_mask == 0 or len_mask < len_prompt:
        session["interactive_mask"].append(res)
    else:
        session['interactive_mask'][len_prompt-1] = res
    

    # Return a JSON response
    # res = Image.fromarray(res)
    session['res'] = Image.fromarray(res)
    sessions[session_id] = session
    return JSONResponse(
        content={
            "masks": pil_image_to_base64(session['res']),
            "message": "Images processed successfully"
        },
        status_code=200,
    )


@router.post("/finish_click")
async def finish_interactive_click(
    mask_idx: int = Form(...),
    session_id: str = Form(...),
):
    global sessions

    session = sessions.get(session_id)
    
    interactive_mask = session['interactive_mask'][mask_idx]
        
    session['segmented_mask'].append(interactive_mask)
    session['interactive_mask'] = list()
    
    sessions[session_id] = session

    return JSONResponse(
        content={
            "message": "Finish successfully",
        },
        status_code=200,
    )


@router.post("/rect")
async def rect_images(
    start_x: int = Form(...), # horizontal
    start_y: int = Form(...), # vertical
    end_x: int = Form(...), # horizontal
    end_y: int = Form(...),  # vertical
    session_id: str = Form(...)
):
    global sessions
    
    session = sessions.get(session_id)

    predictor_3.features = session['features']
    predictor_3.original_size = session['original_size']
    predictor_3.input_size = session['input_size']
    
    masks_, _, _ = predictor_3.predict(
        point_coords=None,
        point_labels=None,
        box=np.array([[start_x, start_y, end_x, end_y]]),
        multimask_output=False
    )
    
    res = Image.fromarray(masks_[0])
    print(masks_[0].shape)
    # res.save("res.png")
    # Return a JSON response
    return JSONResponse(
        content={
            "masks": pil_image_to_base64(res),
            "message": "Images processed successfully"
        },
        status_code=200,
    )


@router.post("/everything")
async def seg_everything(
    params: PointsGenerator
):
    """
        segmentation : the mask
        area : the area of the mask in pixels
        bbox : the boundary box of the mask in XYWH format
        predicted_iou : the model's own prediction for the quality of the mask
        point_coords : the sampled input point that generated this mask
        stability_score : an additional measure of mask quality
        crop_box : the crop of the image used to generate this mask in XYWH format
    """    
    global SAM_0, SAM_1, SAM_2, SAM_3, sessions
    # if type(GLOBAL_MASK) != type(None):
    #     return JSONResponse(
    #         content={
    #             "masks": pil_image_to_base64(GLOBAL_MASK),
    #             "zipfile": b64encode(GLOBAL_ZIPBUFFER.getvalue()).decode("utf-8"),
    #             "message": "Images processed successfully"
    #         },
    #         status_code=200,
    #     )

    params = dict(params)
    pps = int(params['points_per_side'])
    ppb = int(params['points_per_batch'])
    session_id = params['session_id']
    
    session = sessions.get(session_id)
    
    mask_generator = SamAutomaticMaskGenerator(model=SAM_3,
                                               points_per_side=pps,
                                               points_per_batch=ppb)

    masks = mask_generator.generate(session['image'])
    assert len(masks) > 0, "No masks found"

    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    print(len(sorted_anns))

    # Create a new image with the same size as the original image
    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]), dtype=np.uint8)
    for idx, ann in enumerate(sorted_anns, 0):
        img[ann['segmentation']] = idx % 255 + 1 # color can only be in range [1, 255]
    
    res = Image.fromarray(img)
    session['global_mask'] = res

    # Save the original image, mask, and cropped segments to a zip file in memory
    zip_buffer = BytesIO()
    PIL_GLOBAL_IMAGE = Image.fromarray(session['image'])
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Cut out the segmented regions as smaller squares
        for idx, ann in enumerate(sorted_anns, 0):
            left, top, right, bottom = ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]
            cropped = PIL_GLOBAL_IMAGE.crop((left, top, right, bottom))

            # Create a transparent image with the same size as the cropped image
            transparent = Image.new("RGBA", cropped.size, (0, 0, 0, 0))

            # Create a mask from the segmentation data and crop it
            mask = Image.fromarray(ann["segmentation"].astype(np.uint8) * 255)
            mask_cropped = mask.crop((left, top, right, bottom))

            # Combine the cropped image with the transparent image using the mask
            result = Image.composite(cropped.convert("RGBA"), transparent, mask_cropped)

            # Save the result to the zip file
            result_bytes = BytesIO()
            result.save(result_bytes, format="PNG")
            result_bytes.seek(0)
            zip_file.writestr(f"seg_{idx}.png", result_bytes.read())

    # move the file pointer to the beginning of the file so we can read whole file
    zip_buffer.seek(0)
    
    session['zip_buffer'] = zip_buffer
    sessions[session_id] = session
    
    mask_generator.predictor.reset_image()
                                               
    # Return a JSON response
    return JSONResponse(
        content={
            "masks": pil_image_to_base64(session['global_mask']),
            "zipfile": b64encode(session['zip_buffer'].getvalue()).decode("utf-8"),
            "message": "Images processed successfully"
        },
        status_code=200,
    )
