from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from segment_anything import sam_model_registry, SamPredictor
from domain.utils.utils import read_content
from domain.sam import sam_router


sessions = {}

sam_checkpoint = "./models/sam_vit_h_4b8939.pth" # "sam_vit_h_4b8939.pth" or "sam_vit_l_0b3195.pth" or "sam_vit_b_01ec64.pth
model_type = "vit_h" # "vit_l" or "vit_h"
# device = "cuda" if torch.cuda.is_available() else "cpu"
device_0 = "cuda:0"
device_1 = "cuda:1"
device_2 = "cuda:2"
device_3 = "cuda:3"

print("Loading model")
SAM_0 = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device_0)
SAM_1 = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device_1)
SAM_2 = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device_2)
SAM_3 = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device_3)
print("Finishing loading")

predictor_0 = SamPredictor(SAM_0)
predictor_1 = SamPredictor(SAM_1)
predictor_2 = SamPredictor(SAM_2)
predictor_3 = SamPredictor(SAM_3)


if __name__ == "__main__":
    
    app = FastAPI(debug=True)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    app.include_router(sam_router.router)


    @app.get("/assets/{path}/{file_name}", response_class=FileResponse)
    async def read_assets(path, file_name):
        return f"assets/{path}/{file_name}"

    @app.get("/", response_class=HTMLResponse)
    async def read_index():
        return read_content('segDrawer.html')

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
