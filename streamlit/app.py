import streamlit as st
import numpy as np
import torch
import albumentations as A
from stream_utils import out2img, do_gradcam, label2mask, compare_loss,gt_predict_diff
from PIL import Image
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
import json

st.set_page_config(initial_sidebar_state="expanded")
st.title("Bone Segmentation ")
st.subheader("CV-09 Team model")

with st.sidebar:
    mode = st.sidebar.radio(
        "Select Mode",
        ("Output","GradCAM","CompareLoss")
    )

def main():
    global CAM_READY
    box_size = st.radio(
        'Select input image size',
        (512,1024,2048)
    )
    
    img_type = st.radio(
        "Select input img type",
        ("Gray","RGB")
    )
    if img_type == "Gray":
        img_type="L"
    
    device = st.radio(
        "Select device",
        ("cpu","cuda")
    )
    
    # uploaded_model = st.file_uploader("Upload your model.", accept_multiple_files=False, type=['pth','pt'])
    uploaded_model = "/opt/ml/level2_cv_semanticsegmentation-cv-09/checkpoint/hrnet_ocr_best.pt"
    if uploaded_model is not None:
        model = torch.load(uploaded_model,map_location='cpu')
        model.to(device)
        model.eval()

    uploaded_image = st.file_uploader("Upload your image.", accept_multiple_files=False, type=['png','jpg'])
            
    uploaded_json = st.file_uploader("Upload your json",accept_multiple_files=False, type=['json'])
    
    if uploaded_json is not None:
        annotations = json.load(uploaded_json)
        annotations = annotations['annotations']
    if mode != "Output":
        id = st.selectbox("Select Class ID",[i for i in range(29)])
        
    button = st.button("Start inference",)
    with st.spinner("Wait for it..."):
        if button:
            button = False
            if uploaded_model is not None and uploaded_image is not None:
                img = Image.open(uploaded_image)
                img = np.array(img.convert(img_type))
                ori_img = img.copy()
                
                tf = A.Compose([
                    A.Resize(box_size,box_size),
                    ])
                
                if uploaded_json is not None:
                    label = np.zeros((img.shape[0],img.shape[1],29), dtype=np.uint8)
                    mask = label2mask(annotations, label)
                    
                    aug = tf(image=img,mask=mask)
                    img = aug['image']
                    mask = aug['mask']
                    mask = mask.transpose(2,0,1)
                    mask = torch.from_numpy(mask).float().unsqueeze(0)
                    mask = mask.to(device)
                else:
                    img = tf(image=img)['image']
                    
                img = img/255.
                rgb_img = np.float32(img)
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).float().unsqueeze(0)
                img = img.to(device)
                
                gradcams, output = do_gradcam(model,img)
                
                st.success("Done!")
                
                if mode == "Output":
                    out_img = out2img(output)
                    gts, diff = gt_predict_diff(mask, output)
                    col1, col2 = st.columns(2)
                    col1.image(ori_img, caption="Origin")
                    col2.image(out_img, caption=mode)
                    col3, col4 = st.columns(2)
                    col3.image(gts, caption="GT")
                    col4.image(diff, caption="GT-Predict Diff")
                    st.text(f"number of diff pixel {np.count_nonzero(diff)}")

                elif mode == "GradCAM":
                    out_img = show_cam_on_image(rgb_img, gradcams[id], use_rgb=True)
                    col1, col2 = st.columns(2)
                    col1.image(ori_img, caption="Origin")
                    col2.image(out_img, caption=mode)
                    
                elif mode == "CompareLoss":
                    gt_maps,loss_maps = compare_loss(rgb_img, mask, output)
                    col1, col2 = st.columns(2)
                    col1.image(gt_maps[id], caption="Ground Truth")
                    col2.image(loss_maps[id], caption=mode)
                    
                
            else:
                st.text("we have no data (model, img)")
                
    
if __name__=="__main__":
    main()