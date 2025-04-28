# Import modules & set checkpoint (pre-trained model)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
import argparse
import os
import sys
import textwrap

# Setup command line arguments
parser = argparse.ArgumentParser(description='Summarize text from a file using T5 model')
parser.add_argument('--file', type=str, help='Path to the input text file', default=None)
parser.add_argument('--prompt', type=str, help='Custom prompt for summarization', default='summarize')
parser.add_argument('--min_length', type=int, help='Minimum summary length', default=40)
parser.add_argument('--max_length', type=int, help='Maximum summary length', default=150)
parser.add_argument('--model', type=str, help='Model checkpoint to use', default='google/flan-t5-base')
parser.add_argument('--temperature', type=float, help='Sampling temperature', default=0.7)
parser.add_argument('--num_beams', type=int, help='Number of beams for beam search', default=4)
args = parser.parse_args()

print(f"Using model: {args.model}")
print(f"Prompt: '{args.prompt}'")
print(f"Min length: {args.min_length}, Max length: {args.max_length}")
print(f"Temperature: {args.temperature}, Num beams: {args.num_beams}")

try:
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

def summarize_text(text, prompt=args.prompt, min_length=args.min_length, 
                  max_length=args.max_length, temperature=args.temperature,
                  num_beams=args.num_beams):
    """
    Summarize the provided text using the T5 model
    
    Args:
        text (str): Text to summarize
        prompt (str): Prompt for summarization
        min_length (int): Minimum length of summary
        max_length (int): Maximum length of summary
        temperature (float): Sampling temperature
        num_beams (int): Number of beams for beam search
        
    Returns:
        str: Generated summary
    """
    try:
        # Clean the text (remove extra whitespace)
        text = ' '.join(text.split())
        
        # Prepare the input
        summarization_input = f"{prompt}: {text}"
        tokens_input = tokenizer.encode(summarization_input, return_tensors="pt", truncation=True)
        
        # Generate summary with improved parameters
        summary_ids = model.generate(
            tokens_input, 
            min_length=min_length, 
            max_length=max_length,
            temperature=temperature,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            do_sample=(temperature > 0)
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Error: Could not generate summary."

# Function to format text for display
def format_text_display(text, max_display_length=500):
    if len(text) > max_display_length:
        return text[:max_display_length] + "..."
    return text

# Use a provided file if specified, otherwise use the example text
if args.file and os.path.exists(args.file):
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            text_to_summarize = f.read()
        print(f"Summarizing text from file: {args.file}")
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
else:
    if args.file:
        print(f"Warning: File '{args.file}' not found. Using example text instead.")
    
    # Default example text if no file is provided
    text_to_summarize = "Large language models (LLMs) have a dirty secret: they require vast amounts of energy to train and run. What's more, it's still a bit of a mystery exactly how big these models' carbon footprints really are. AI startup Hugging Face believes it's come up with a new, better way to calculate that more precisely, by estimating emissions produced during the model's whole life cycle rather than just during training. It could be a step toward more realistic data from tech companies about the carbon footprint of their AI products at a time when experts are calling for the sector to do a better job of evaluating AI's environmental impact. Hugging Face's work is published in a non-peer-reviewed paper. To test its new approach, Hugging Face estimated the overall emissions for its own large language model, BLOOM, which was launched earlier this year. It was a process that involved adding up lots of different numbers: the amount of energy used to train the model on a supercomputer, the energy needed to manufacture the supercomputer's hardware and maintain its computing infrastructure, and the energy used to run BLOOM once it had been deployed. The researchers calculated that final part using a software tool called CodeCarbon, which tracked the carbon dioxide emissions BLOOM was producing in real time over a period of 18 days. Hugging Face estimated that BLOOM's training led to 25 metric tons of carbon dioxide emissions. But, the researchers found, that figure doubled when they took into account the emissions produced by the manufacturing of the computer equipment used for training, the broader computing infrastructure, and the energy required to actually run BLOOM once it was trained."
    print("Using example text (no file provided)")

print("\nInput Text:")
print("-" * 80)
print(format_text_display(text_to_summarize))
print("-" * 80)

# Generate and display summary
print("\nGenerating summary...")
summary = summarize_text(text_to_summarize)
print("\nSummary:")
print("-" * 80)
# Format the summary for better display with proper line wrapping
wrapped_summary = textwrap.fill(summary, width=80)
print(wrapped_summary)
print("-" * 80)