from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import argparse
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Customize CV based on job description using LLMs'
    )
    
    parser.add_argument(
        '--job-desc',
        default='job_description.txt',
        help='Path to job description text file'
    )
    parser.add_argument(
        '--cv',
        default='my_cv.txt',
        help='Path to CV text file'
    )
    
    parser.add_argument(
        '--keyword-provider',
        choices=['ollama', 'openai'],
        default='ollama',
        help='Provider for keyword extraction (default: ollama)'
    )
    parser.add_argument(
        '--cv-provider',
        choices=['ollama', 'openai'],
        default='ollama',
        help='Provider for CV modification (default: ollama)'
    )
    parser.add_argument(
        '--keyword-model',
        default='llama3',
        help='Model name for keyword extraction (default: llama3 for ollama, gpt-3.5-turbo for openai)'
    )
    parser.add_argument(
        '--cv-model',
        default='llama3',
        help='Model name for CV modification (default: llama3 for ollama, gpt-3.5-turbo for openai)'
    )
    
    # Other arguments
    parser.add_argument(
        '--output',
        default='customized_cv.txt',
        help='Output file path (default: customized_cv.txt)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    return parser.parse_args()

def get_llm(provider, model_name):
    if provider == 'ollama':
        return Ollama(model=model_name)
    elif provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Error: OPENAI_API_KEY not found in .env file")
            sys.exit(1)
        return ChatOpenAI(
            api_key=api_key,
            model_name=model_name or "gpt-3.5-turbo",
            temperature=0.7
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

def extract_keywords(job_description, provider, model_name, verbose=False):
    if verbose:
        print(f"Extracting keywords using {provider} ({model_name}) model...")
    
    llm = get_llm(provider, model_name)
    
    keyword_prompt_template = """
    You are a professional CV/resume expert.
    Given the following job description, extract the most important keywords and phrases.
    Include both hard skills and soft skills that appear frequently or seem crucial for the role.
    Present only the list of keywords without any additional explanation.

    Job Description:
    ```
    {job_description}
    ```

    Please provide a bullet-pointed list of key words and phrases.
    """
    
    prompt = PromptTemplate(
        input_variables=["job_description"],
        template=keyword_prompt_template
    )
    
    final_prompt = prompt.format(job_description=job_description)
    keywords = llm.predict(final_prompt)  # Using predict instead of direct call
    return keywords

def modify_cv(keywords, cv, provider, model_name, verbose=False):
    if verbose:
        print(f"Modifying CV using {provider} ({model_name}) model...")
    
    llm = get_llm(provider, model_name)
    
    cv_modification_template = """
        You are a professional CV/resume expert specializing in tailoring CVs to specific job requirements.

        Your task is to optimize the given CV using the extracted keywords/phrases from the job description. Follow these guidelines:

        1. KEYWORD INCORPORATION:
        - Naturally integrate keywords where they align with existing experience
        - Look for synonyms or similar phrases in the CV that could be replaced with the exact keywords
        - When using a keyword/phrase from the list, mark it with '$' prefix

        2. EXPERIENCE EMPHASIS:
        - Reorder bullet points to prioritize most relevant experiences
        - Expand descriptions of experiences that match job requirements
        - Add specific metrics or outcomes where they support required skills
        - Consider merging related bullet points to create stronger, keyword-rich descriptions

        3. SKILLS ALIGNMENT:
        - Identify transferable skills that match keywords but may be described differently
        - Highlight relevant technical skills earlier in descriptions
        - Ensure soft skills are demonstrated through concrete examples

        4. HONESTY RULES:
        - Never invent or exaggerate experiences
        - Only use keywords where they genuinely match existing experience
        - Keep all dates, roles, and qualifications exactly as they appear
        - If you're expanding a description, use only information that can be reasonably inferred from the original

        5. FORMAT REQUIREMENTS:
        - Maintain the original CV structure and sections
        - Keep the professional tone and writing style
        - Ensure all modified content aligns with the original context

        Keywords and phrases from the job description:
        {keywords}

        Original CV:
        {cv}
        
        Please provide the modified CV with the following:
        1. Mark all incorporated keywords/phrases with '$' prefix
        2. Maintain original format but optimize content following the guidelines above
        3. Focus on meaningful integration rather than superficial keyword placement
        4. Ensure every modification preserves the authenticity of the original experience

        Remember: The goal is to make the existing experience more relevant and visible to the job requirements, not to create new experience.
        """
    
    prompt = PromptTemplate(
        input_variables=["keywords", "cv"],
        template=cv_modification_template
    )
    
    final_prompt = prompt.format(keywords=keywords, cv=cv)
    modified_cv = llm.predict(final_prompt)  # Using predict instead of direct call
    return modified_cv

def load_text_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{file_path}': {str(e)}")
        sys.exit(1)

def main():
    args = parse_arguments()
    
    if args.verbose:
        print("Starting CV customization process...")
    
    # Load files
    job_description = load_text_file(args.job_desc)
    cv = load_text_file(args.cv)
    
    # Step 1: Extract keywords
    keywords = extract_keywords(
        job_description, 
        args.keyword_provider,
        args.keyword_model,
        args.verbose
    )
    if args.verbose:
        print("\nExtracted Keywords:")
        print(keywords)
        print("\n" + "="*50 + "\n")
    
    # Step 2: Modify CV using keywords
    modified_cv = modify_cv(
        keywords,
        cv,
        args.cv_provider,
        args.cv_model,
        args.verbose
    )
    
    # Save output
    try:
        with open(f"cvs/{args.output}", "w") as f:
            f.write(modified_cv)
        if args.verbose:
            print(f"\nCustomized CV saved to: {args.output}")
    except Exception as e:
        print(f"Error saving output file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()