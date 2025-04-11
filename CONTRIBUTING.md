# 🤝 Contributing to Immunova

Thank you for your interest in contributing to **Immunova** — a cutting-edge, open-source platform for multi-modal analysis and treatment modeling in pancreatic cancer. This project is built for collaboration, and we welcome your ideas, issues, and improvements.

---

## 🧠 What You Can Work On

We welcome contributions in any of the following areas:

- 📊 Improving model architectures (CNNs, Transformers, GNNs)
- ⚙️ Optimizing training and validation pipelines
- 🌐 Enhancing the frontend (UI, interactivity, performance)
- 🧬 Expanding support for clinical/genomic datasets
- 🧪 Improving validation with real-world patient data
- ☁️ Enhancing deployment (Docker, Azure, scalable APIs)
- 📖 Writing documentation, tutorials, or examples

---

## 🛠 Getting Started

1. **Fork** the repository  
2. **Clone** your fork:
   ```
   git clone https://github.com/YOUR_USERNAME/immunova.git
   ```
3. **Navigate** into the project directory:
   ```
   cd immunova
   ```
4. **Create a new branch** for your feature or fix:
   ```
   git checkout -b feature/my-feature-name
   ```
5. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

6. **Make your changes locally**

7. **Push your branch**:
   ```
   git push origin feature/my-feature-name
   ```

8. **Open a Pull Request** from your fork to the `main` branch

---

## ✅ Code Style & Practices

- Use **clear, readable code** with comments where needed  
- Prefer **type hints** and docstrings  
- Follow **PEP8** style guide  
- Format code using:
   ```
   black .
   ```
- Keep commits atomic and descriptive  
- Add **tests** for your changes if applicable  
- Validate performance on synthetic or real data if needed  

---

## 📁 Project Structure

```
til_classification/         # TIL CNNs + Grad-CAM
treatment_response/         # Imaging/genomics/fusion models
survival_analysis/          # Transformer survival models
drug_optimization/          # GNN + RL drug strategies
validation/                 # Clinical/external dataset validators
frontend/                   # Web frontend (HTML/CSS/JS)
deployment/                 # Azure deployment configs
```

---

## 🧪 Testing

Please include basic tests for any new models or pipelines.

Run existing tests using:
```
pytest
```

Or include a script that validates your module on dummy/sample data.

---

## 🙋 Need Help?

Open an issue or start a discussion. You can also contact the core maintainers via email listed in the repo or project README.

We’re excited to collaborate with you — let’s build the future of precision oncology together. 💙
