# VisionGPT: (Visual Diary Generator)

![GitHub Repo stars](https://img.shields.io/github/stars/ahatsham/VisionGPT?style=social)

**VisionGPT** is a multimodal project where screenshots, activity graphs, or mobility maps are fed into a Vision-Language model to generate natural language summaries describing a user's day.

## ✨ Features
- Fine-tuned LLaVA-style architecture.
- Simulated mobile data as input.
- Textual summaries describing behavior

## 🔧 Setup
```bash
git clone https://github.com/ahatsham/VisionGPT.git
cd VisionGPT
pip install -r requirements.txt
```

## 📊 Demo
```bash
python app/gradio_demo.py
```

## 🚀 Training setup
```bash
python scripts/train.py
```

## 📁 Dataset
Structure your dataset like this:
```
dataset/
  raw_images/
    user1_day1.png
    user1_day2.png
  annotations.json
```

## 🤝 Contributing
Pull requests are welcome! Star ⭐ and share if you find this helpful!
