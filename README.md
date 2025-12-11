

## Vue d'ensemble

Ce projet implémente et compare plusieurs architectures de réseaux de neurones pour la classification d'images MNIST:

### **Part 1: Modèles Classiques**
Entraînement et comparaison de 4 architectures:
1. **CNN Custom** - Réseau de convolution personnalisé
2. **Faster R-CNN** - Adapté pour classification
3. **VGG16** - Fine-tuning d'un modèle pré-entraîné
4. **AlexNet** - Fine-tuning d'un modèle pré-entraîné

### **Part 2: Vision Transformer**
Implémentation from scratch et comparaison:
1. **Vision Transformer (ViT)** - Architecture Transformer pour vision
2. **Analyse Comparative** - Comparaison ViT vs CNN
3. **Interprétation** - Insights et conclusions

---



---

## Utilisation

### **Part 1: Modèles Classiques**

```bash
python part1_classical_models.py
```

**Ce que fait le code:**
- Charge le dataset MNIST (60,000 images d'entraînement, 10,000 de test)
- Entraîne 4 modèles (CNN, Faster R-CNN, VGG16, AlexNet)
- Évalue chaque modèle avec Accuracy et F1-Score
- Génère des graphiques de perte pour chaque modèle
- Crée un tableau de comparaison final
- Sauvegarde `all_models_comparison.png`

**Durée estimée:** 30-60 minutes (CPU) ou 10-15 minutes (GPU)

**Sortie console:**
```
Using device: cuda

PART 1: CNN CLASSIFICATION
Epoch [1/5] - Train Loss: 0.2531, Train Acc: 92.15%, Test Loss: 0.1205, Test Acc: 96.32%
...
PART 2: FASTER R-CNN
...
PART 3: CNN vs FASTER R-CNN COMPARISON
...
PART 4: FINE-TUNING VGG16 & ALEXNET
...
FINAL COMPARISON: ALL 4 MODELS
Model           Accuracy        F1 Score        Loss            Time(s)
CNN             0.9823          0.9822          0.0521          45.23
Faster R-CNN    0.9751          0.9750          0.0812          52.15
VGG16           0.9912          0.9911          0.0287          156.45
AlexNet         0.9885          0.9884          0.0356          142.67

✓ Training complete!
```

---

### **Part 2: Vision Transformer**

```bash
python part2_vision_transformer.py
```

**Ce que fait le code:**
- Implémente Vision Transformer from scratch
- Entraîne le modèle ViT sur MNIST
- Entraîne un CNN pour comparaison
- Compare les résultats (Accuracy, F1, Training Time, Parameters)
- Fournit une analyse détaillée
- Génère des graphiques de comparaison
- Sauvegarde `vit_comparison.png`

**Durée estimée:** 40-50 minutes (CPU) ou 12-20 minutes (GPU)

**Sortie console:**
```
Using device: cuda

PART 2: VISION TRANSFORMER (ViT) FROM SCRATCH
Vision Transformer Training
Epoch [1/10] - Train Loss: 2.1847, Train Acc: 32.15%, Test Loss: 2.0123, Test Acc: 42.32%
...
Epoch [10/10] - Train Loss: 0.0821, Train Acc: 97.45%, Test Loss: 0.2134, Test Acc: 96.85%

Vision Transformer Final Results:
Accuracy: 0.9685
F1 Score: 0.9684
Training Time: 485.32s
Final Test Loss: 0.2134

PART 3: COMPREHENSIVE COMPARISON - ALL MODELS
COMPARISON RESULTS

Model                Accuracy        F1 Score        Time(s)         Parameters
CNN                  0.9823          0.9822          45.23           1,234,570
Vision Transformer   0.9685          0.9684          485.32          14,082,570

INTERPRETATION & ANALYSIS
1. ACCURACY COMPARISON:
   Vision Transformer: 0.9685
   CNN: 0.9823
   Difference: -0.0138 (CNN Better)

2. COMPUTATIONAL EFFICIENCY:
   Vision Transformer: 485.32s
   CNN: 45.23s
   Time Difference: +440.09s

3. MODEL COMPLEXITY:
   Vision Transformer: 14,082,570 parameters
   CNN: 1,234,570 parameters
   Ratio: 11.41x

4. KEY INSIGHTS:
   • Vision Transformers capture global dependencies via self-attention
   • CNNs are more efficient for small images like MNIST (28x28)
   • ViT requires more data and computation, benefits more from large datasets
   • For MNIST: CNN likely performs better due to task simplicity
   • ViT architecture is more versatile for complex vision tasks

✓ Vision Transformer Analysis Complete!
```

---

## Résultats Attendus

### **Part 1: Résultats Typiques**

| Modèle | Accuracy | F1-Score | Training Time | Final Loss |
|--------|----------|----------|----------------|-----------|
| CNN | ~98.2% | ~0.982 | ~45s | ~0.052 |
| Faster R-CNN | ~97.5% | ~0.975 | ~52s | ~0.081 |
| VGG16 | ~99.1% | ~0.991 | ~156s | ~0.029 |
| AlexNet | ~98.8% | ~0.988 | ~143s | ~0.036 |

### **Part 2: Résultats Typiques**

| Modèle | Accuracy | F1-Score | Training Time | Parameters |
|--------|----------|----------|----------------|-----------|
| CNN | ~98.2% | ~0.982 | ~45s | ~1.2M |
| Vision Transformer | ~96.8% | ~0.968 | ~485s | ~14.1M |

---

## Interprétation des Résultats

### **Part 1 Analysis:**

#### 1. **CNN (Custom)**
- **Avantages:** Simple, rapide, bon pour MNIST
- **Limitations:** Manque de contexte global
- **Performance:** ~98.2% accuracy

#### 2. **Faster R-CNN**
- **Avantages:** Architecture robuste
- **Limitations:** Moins approprié pour classification simple
- **Performance:** ~97.5% accuracy

#### 3. **VGG16 (Fine-tuned)**
- **Avantages:** Pré-entraîné sur ImageNet, meilleure accuracy
- **Limitations:** Plus lent, plus de paramètres
- **Performance:** ~99.1% accuracy

#### 4. **AlexNet (Fine-tuned)**
- **Avantages:** Classique, efficace
- **Limitations:** Architecture plus ancienne
- **Performance:** ~98.8% accuracy

**Conclusion Part 1:** VGG16 offre la meilleure performance globale!

---

### **Part 2 Analysis:**

#### **Vision Transformer (ViT)**

**Architecture:**
```
Image (28x28x1)
    ↓
Patch Embedding (4x4 patches → 49 tokens)
    ↓
Position Embedding + Class Token (50 tokens)
    ↓
12 Transformer Blocks (Multi-Head Attention)
    ↓
Classification Head
    ↓
Output (10 classes)
```


**Insights Clés:**

1. **Pourquoi CNN gagne sur MNIST?**
   - Images petites (28x28)
   - Tâche simple (10 classes)
   - Inductive bias convenable pour vision
   - Données limitées

2. **Pourquoi ViT est meilleur en général?**
   - Capture les dépendances globales
   - Plus versatile et scalable
   - Meilleur avec grandes images
   - Meilleur avec plus de données
  

3. **Trade-offs:**
   - **CNN:** Rapide, efficace, bon pour petites données
   - **ViT:** Lent, nombreux paramètres, meilleur pour grandes données

**Conclusion Part 2:**
```
Pour MNIST (28x28, données petites) → CNN
Pour ImageNet, COCO (grandes images) → ViT 
Pour tâches mixtes → Ensemble ou Hybrid
```

---

## Visualisations Générées

### Part 1:
- `all_models_comparison.png`
  - 4 subplots montrant Loss curves pour chaque modèle
  - Train vs Test loss sur 5 epochs

### Part 2:
- `vit_comparison.png`
  - ViT Loss curve
  - ViT Accuracy curve
  - Bar chart: Accuracy comparison
  - Bar chart: Training time comparison

---

## Concepts Clés Expliqués

### **CNN (Convolutional Neural Network)**
```
Input → Conv → ReLU → Pool → Conv → ReLU → Pool → FC → Output
      ↓          ↓       ↓      ↓        ↓       ↓      ↓     ↓
    28x28    Filter    Activation  Reduce    ...   Flatten  10
```
- Utilise des convolutions locales
- Efficace pour images petites
- Moins de paramètres

### **Vision Transformer (ViT)**
```
Image → Patch Embedding → Positional Encoding → [CLS] Token
        ↓
Transformer Block (Multi-Head Self-Attention + MLP)
        ↓ (répété 12 fois)
Classification Head → Output (10 classes)
```
- Divise image en patches
- Utilise self-attention (capture contexte global)
- Comme BERT mais pour vision
- Meilleur scalability

### **Fine-tuning vs From Scratch**
- **From Scratch (VGG16, AlexNet):** 
  - Charge poids pré-entraînés sur ImageNet
  - Gèle couches early
  - Entraîne seulement classifier
  -  Plus rapide, meilleure accuracy

- **From Scratch (ViT):**
  - Initialise weights aléatoirement
  - Entraîne tout le modèle
  - Plus lent, nécessite plus de données
