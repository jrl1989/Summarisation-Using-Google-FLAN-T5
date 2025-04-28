# Text Summarization with Google T5

This project provides a simple utility for summarizing text using Google's T5 model. It allows you to summarize text from a file using customizable prompts and parameters.

## Features

- Summarize text from files or use an example text
- Customize summarization prompts
- Adjust minimum and maximum summary length
- Control generation parameters (temperature, beam search)
- Select different T5 model variants
- Error handling and formatted output

## Requirements

Install the required dependencies:

```bash
pip install transformers torch numpy
```

## Usage

Basic usage with default parameters:

```bash
python "Summarisation Using Google T5.py"
```

Summarize text from a file:

```bash
python "Summarisation Using Google T5.py" --file sample_text.txt
```

Customize the summarization:

```bash
python "Summarisation Using Google T5.py" --file sample_text.txt --prompt "provide a detailed summary of" --min_length 100 --max_length 200
```

Control generation parameters:

```bash
python "Summarisation Using Google T5.py" --file sample_text.txt --temperature 0.8 --num_beams 5
```

Use a different T5 model variant:

```bash
python "Summarisation Using Google T5.py" --model google/flan-t5-large
```

## Available Arguments

- `--file`: Path to input text file (optional)
- `--prompt`: Custom prompt for summarization (default: "summarize")
- `--min_length`: Minimum summary length (default: 40)
- `--max_length`: Maximum summary length (default: 150)
- `--temperature`: Sampling temperature (default: 0.7)
- `--num_beams`: Number of beams for beam search (default: 4)
- `--model`: Model checkpoint to use (default: google/flan-t5-base)

## Examples of Custom Prompts

- `summarize`
- `provide a detailed summary of`
- `explain the key points in`
- `what are the main ideas in`
- `create a concise summary of`
- `extract the most important information from`

## Available T5 Models

- `google/flan-t5-small` (smallest, fastest)
- `google/flan-t5-base` (default)
- `google/flan-t5-large`
- `google/flan-t5-xl`
- `google/flan-t5-xxl` (largest, best quality)

Note: Larger models provide better quality summaries but require more computational resources.

## Understanding Parameters

- **Temperature**: Controls randomness in generation. Lower values (e.g., 0.3) give more deterministic outputs, while higher values (e.g., 0.9) produce more diverse summaries.
- **Number of beams**: Controls the beam search algorithm. Higher values explore more possibilities and generally produce better summaries but take longer to generate.
- **Min/Max length**: Control the length of the generated summary in tokens.

## Error Handling

The script includes error handling for:
- Invalid file paths
- Model loading issues
- Summarization errors
- File reading problems