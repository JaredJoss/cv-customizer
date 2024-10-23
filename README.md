# CV Customizer

A Python tool that uses AI to customize your CV/resume based on job descriptions. The tool extracts key phrases from a job description and intelligently incorporates them into your CV while maintaining accuracy and honesty.

## Features

- Supports multiple AI providers (OpenAI and Ollama)
- Two-step customization process:
  1. Extract important keywords from job description
  2. Modify CV to incorporate relevant keywords
- Configurable model selection for each step
- Command-line interface with various options
- Support for environment variables for API keys

## Prerequisites

- Python 3.8+
- OpenAI API key (if using OpenAI models)
- Ollama installed (if using Ollama models)

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd cv_customizer
```

2. Install required packages:
```bash
pip install langchain openai python-dotenv
```

3. If using Ollama, install it:
```bash
# On macOS
brew install ollama

# On Linux
curl https://ollama.ai/install.sh | sh
```

4. Set up environment variables:
Create a .env file with the following variables:
```
OPENAI_API_KEY=your_api_key_here
```


## Usage
### Basic Usage:
```bash
python cv_customizer.py
```

### Using OpenAI models:
```bash
python cv_customizer.py \
    --job-desc job_description.txt \
    --cv my_cv.txt \
    --keyword-provider openai \
    --cv-provider openai \
    --keyword-model gpt-3.5-turbo \
    --cv-model gpt-4 \
    --verbose
```

### Using Ollama models:
```bash
python cv_customizer.py \
    --job-desc job_description.txt \
    --cv my_cv.txt \
    --keyword-provider ollama \
    --cv-provider ollama \
    --keyword-model mistral \
    --cv-model llama2 \
    --verbose
```

## Command Line Options

- `--job-desc`: Path to job description text file
- `--cv`: Path to CV text file
- `--keyword-provider`: AI provider for keyword extraction (ollama or openai)
- `--cv-provider`: AI provider for CV modification (ollama or openai)
- `--keyword-model`: Model name for keyword extraction
- `--cv-model`: Model name for CV modification
- `--output`: Output file path (default: customized_cv.txt)
- `--verbose`: Print detailed output
- `--help`: Show help message

