# Text Summarization using Google's FLAN-T5 Model

This project demonstrates how to use Google's FLAN-T5 model for text summarization using the Hugging Face Transformers library.

## Requirements

- Python 3.x
- transformers
- numpy

Install the required packages:
-bash
-pip install transformers numpy

## Usage

1. Import the necessary modules:
   ```python
   from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
   import numpy as np
   ```

2. Set the checkpoint for the pre-trained model:
   ```python
   checkpoint = "google/flan-t5-small"
   ```

3. Prepare your text to be summarized.

4. Load the tokenizer and model:
   ```python
   tokenizer = AutoTokenizer.from_pretrained(checkpoint)
   model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
   ```

5. Tokenize the input text:
   ```python
   summarization_input = "summarize: " + your_text
   tokens_input = tokenizer.encode(summarization_input, return_tensors="pt", truncation=True)
   ```

6. Generate the summary:
   ```python
   summary_ids = model.generate(tokens_input, min_length=40, max_length=130)
   summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
   ```

7. Print the summary.

## Example

The provided code includes an example text about large language models and their carbon footprint. Run the script to see the summarization in action.

## Note

This example uses the "google/flan-t5-small" model. You can experiment with other T5 variants by changing the `checkpoint` variable.