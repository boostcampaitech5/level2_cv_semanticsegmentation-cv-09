import bentoml
from bentoml.io import Text, Image 

segment_runner = bentoml.pytorch.get("pytorch_unet").to_runner()

svc = bentoml.Service(
    name="Bone_Image_Segmentor",
    runners=[segment_runner],
)


@svc.api(input=Text(), output=Image())
async def predict(path: str) -> str:
    assert isinstance(path, str)
    return await segment_runner.async_run(path)