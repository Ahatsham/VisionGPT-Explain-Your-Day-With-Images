from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, AutoTokenizer, AutoModelForCausalLM

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
language_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

def infer(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    img_features = vision_encoder(**inputs).last_hidden_state
    prompt = "Describe this person's day based on the image."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated = language_model.generate(input_ids, max_new_tokens=100)
    return tokenizer.decode(generated[0], skip_special_tokens=True)