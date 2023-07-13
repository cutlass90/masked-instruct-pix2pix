import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from pipeline_stable_diffusion_masked_instruct_pix2pix import StableDiffusionMaskedInstructPix2PixPipeline as Pipe
import torch

st.set_page_config(layout="wide")
st.title("StableDiffusionMaskedInstructPix2PixPipeline Demo")
DEVICE = 'cuda:0'
PROMPT = 'add a knight helmet'
NEGATIVE_PROMPT = ''

def main():
    if 'pipeline' not in st.session_state:
        with st.spinner(text="Loading models. Please wait..."):
            st.session_state['pipeline'] = Pipe.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16,
                                                                safety_checker=None).to(DEVICE)
    col1, col2 = st.columns(2)

    input_image = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    prompt = st.sidebar.text_input("Prompt", PROMPT)
    negative_prompt = st.sidebar.text_input("Negative prompt", NEGATIVE_PROMPT)
    guidance_scale = st.sidebar.slider("Guidance Scale", min_value=1, max_value=30, value=10, step=1)
    image_guidance_scale = st.sidebar.slider("Image Guidance Scale", min_value=0., max_value=5., value=1.5, step=0.1)
    steps = st.sidebar.slider("Inference Steps", min_value=5, max_value=100, value=30, step=1)
    stroke_width = st.sidebar.slider("Stroke width: ", 10, 50, 30)
    if input_image is not None:
        input_image = Image.open(input_image)

        with col1:
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_color="rgba(0, 255, 0, 0.5)",
                stroke_width=stroke_width,
                background_image=input_image,
                update_streamlit=True,
                height=input_image.height,
                width=input_image.width,
                drawing_mode='freedraw',
                point_display_radius=0,
                key="canvas",
            )

        if st.button('RUN'):
            with st.spinner("Working, wait a moment..."):
                mask = (canvas_result.image_data[:, :, 3] > 60).astype(np.float32)
                result_image = st.session_state['pipeline'](prompt, negative_prompt=negative_prompt, image=input_image, num_inference_steps=steps,
                                                            guidance_scale=guidance_scale, mask=mask, image_guidance_scale=image_guidance_scale).images[0]

                with col2:
                    st.image(result_image)


if __name__ == "__main__":
    main()