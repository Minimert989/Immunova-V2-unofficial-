# 🧬 Immunova

**Immunova** is an open-source platform for multi-modal analysis and treatment modeling in pancreatic cancer. It integrates histopathology, genomics, survival data, and drug-response modeling into one unified, modular framework.

---

## 🚀 Quickstart

```bash
# Train a TIL classifier
python til_classification/train.py

# Run inference on genomics + imaging
python treatment_response/fusion_model.py

# Validate survival predictions
python survival_analysis/km_validation.py
```

---

## 🧪 Testing

```bash
pytest
```

---

## 🤝 Contributing

We welcome contributions! Please check out [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started.

---

## 📁 Project Structure

```
immunova/
├── til_classification/         # TIL CNNs + Grad-CAM
│   ├── train.py
│   └── ...
├── treatment_response/         # Imaging/genomics/fusion models
│   ├── fusion_model.py
│   └── ...
├── survival_analysis/          # Transformer survival models
│   ├── km_validation.py
│   └── ...
├── drug_optimization/          # GNN + RL drug strategies
│   └── ...
├── validation/                 # Clinical/external dataset validators
│   └── ...
├── frontend/                   # Web frontend (HTML/CSS/JS)
│   └── ...
├── deployment/                 # Azure deployment configs
│   └── ...
├── tests/                      # Unit tests and model tests
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 👩‍🔬 Acknowledgements

Built by young researchers and contributors in the fight against pancreatic cancer. Datasets and validation supported by collaborations with leading institutions and clinical data sources.

---

## 💙 Join Us

Let’s push the boundaries of computational oncology — together.
