# DenSiSeg: Density-Aware Progressive Smoke Segmentation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Torch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)

This repository contains the official PyTorch implementation of **DenSiSeg**, a novel smoke segmentation framework that explicitly models smoke density variations via a progressive optimization scheme and background-guided representation learning.

📄 **Paper:** _[Title]_  
✒️ **Authors:** [Your Name], et al.  
📚 **To appear in:** *Pattern Recognition, 2025*  
🔗 **PDF:** [Link to the paper]  
📬 **Contact:** [your-email@example.com]

---

## 🔥 Highlights

- 🌀 **Density-aware prediction** using a novel cosine-based estimation module.
- 🧠 **Background-guided learning** with SAM-derived feature alignment.
- 📈 **Soft contrastive learning** to improve intra-class separability.
- 🎯 Achieves SOTA performance on **SYN70K**, **Smoke5K**, and **Rekon** datasets.

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/DenSiSeg.git
cd DenSiSeg

# Create environment (optional but recommended)
conda create -n densiseg python=3.8
conda activate densiseg

# Install dependencies
pip install -r requirements.txt
