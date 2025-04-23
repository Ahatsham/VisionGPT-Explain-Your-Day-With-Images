import gradio as gr
from scripts.inference import infer

def generate_summary(image):
    return infer(image)

demo = gr.Interface(fn=generate_summary, inputs=gr.Image(type="filepath"), outputs="text")
demo.launch()