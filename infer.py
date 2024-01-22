from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import torch

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", tokenizer_class="LlamaTokenizer")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
gpu_id = 4
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

while True:
    user_input = input("You: ")

    if user_input.lower() == 'exit':
        break

    input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(input_ids, max_length=200, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Model: {generated_text}")

