import streamlit as st 

def ensemble():
    uploaded_model = st.file_uploader("Upload your model.", accept_multiple_files=True, type=['pth','pt'])
    if uploaded_model is not None:

        for model in uploaded_model:
            resize = st.number_input('Insert input resolution value(512, 1024, ... etc.)')

            test_dataset = XRayInferenceDataset(
            data_dir=args.data_dir,
            transforms=transform
            )
        
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=2,
                shuffle=False,
                num_workers=2,
                drop_last=False
            )