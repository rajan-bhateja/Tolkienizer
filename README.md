# 🧙 Tolkienizer  

Tolkienizer is a fine-tuned language model that transforms plain English into writing styled after **J.R.R. Tolkien**.  
The project explores the use of **LLaMA 3.2 1B Instruct model** with **parameter-efficient fine-tuning** (LoRA via [Unsloth](https://github.com/unslothai/unsloth)), using a custom dataset.  

> ⚠️ This project is built **for educational and portfolio purposes only**. It is not intended for commercial use.  

---

## ✨ Features
- Converts modern English text into Tolkien-inspired prose.  
- Fine-tuned on **2,500+ quotes** created through a **custom data generation pipeline**.  
- Optimized training setup using **LoRA** to make it feasible on limited hardware (Google Colab).  
- Demonstrates how to fine-tune large language models for **stylistic transformations**.  

---

## 🚀 How It Works
1. **Data Generation**  
   - Extracted and cleaned Tolkien quotes from public-domain sources. (Goodreads)
   - Generated structured pairs of **prompt (plain English)** and **completion (Tolkien-style text)**.  

2. **Model Fine-Tuning**  
   - Base model: *LLaMA 3.2 1B Instruct* 
   - Fine-tuned with **LoRA adapters** using Unsloth’s `SFTTrainer`.  
   - Configured with sequence packing, gradient accumulation, and efficient optimizers.  

3. **Inference**  
   - Input: `"The sun is rising over the valley."`  
   - Output: `"Behold, the golden sun arose, casting its first light upon the deep valley and its silent woods."`  

---

## ⚙️ Installation
```bash
# Clone the repo
git clone https://github.com/rajan-bhateja/Tolkienizer.git
cd Tolkienizer

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Usage
```
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "path_to_your_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "The stars are shining in the night sky."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## ⚠️ Limitations & Future Work

- **Stylistic accuracy**: While the model captures Tolkien’s tone, it sometimes drifts into generic fantasy language.  
- **Data constraints**: The dataset (~2,500 pairs) is relatively small, which limits fluency and diversity of generations.  
- **Model size**: Training was done on smaller instruction-tuned models (≤1B parameters), which naturally restricts the depth of expression compared to larger models.  
- **Inference quality**: Longer outputs may lose coherence, especially in multi-paragraph generations.  

### Future Work (Highly Unlikely)
- Scale up training with **larger LLMs** (7B+ parameters).  
- Expand dataset with more diverse and context-rich Tolkien-inspired examples.  
- Explore **RAG pipelines** to ground text in Tolkien’s lore.  
- Test more advanced fine-tuning strategies (e.g., QLoRA with better hyperparameter tuning, PEFT combinations).
