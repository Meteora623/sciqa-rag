def answer_query_baseline(query):
    prompt = f"""You are a helpful scientific assistant. Please answer the following question clearly and concisely.

### Question:
{query}

### Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()