from transformers import CLIPVisionModel, AutoTokenizer, AutoModelForCausalLM
import torch

vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
language_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

def train(image_inputs, texts):
    for img_input, txt in zip(image_inputs, texts):
        img_features = vision_encoder(**img_input).last_hidden_state
        inputs = tokenizer(txt, return_tensors="pt")
        outputs = language_model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss
        loss.backward()
        print("Loss:", loss.item())