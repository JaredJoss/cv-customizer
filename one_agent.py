from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
import os

# Initialize Ollama (assuming you have it installed and running)
llm = Ollama(model="llama3")  # or another model you prefer

# Template for the prompt
prompt_template = """
You are a professional CV/resume expert. 
You will be given a job description and CV, and your task is to customize the CV to better match the job requirements.

This is a two step process:
1. Pull out important words that come up often in the job description and seem important, these are hard skills as well as soft skills. 
2. Use the important words from 1. and find ways to fit them into the CV. You have to maintain honesty and accuracy while focusing on relevant experience and skills.

You do not have to focus on making the existing sentences more concise and proffesional but rather focus on personalizing the CV to the job description. Do not make up experience that is not present in the CV.

Job Description:
```
{job_description}
```

Original CV:
```
{cv}
```

Please provide the following:
- the key words you have pulled out from the job description
- modified CV
"""

def load_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def customize_cv(job_description_path, cv_path):
    # Load the job description and CV
    job_description = load_text_file(job_description_path)
    cv = load_text_file(cv_path)
    
    # Create the prompt
    prompt = PromptTemplate(
        input_variables=["job_description", "cv"],
        template=prompt_template
    )
    
    # Generate the customized CV
    final_prompt = prompt.format(job_description=job_description, cv=cv)
    response = llm(final_prompt)
    
    return response

# Example usage
if __name__ == "__main__":
    job_desc_path = "job_description.txt"
    cv_path = "my_cv.txt"
    
    customized_cv = customize_cv(job_desc_path, cv_path)
    
    # Save the result
    with open("customized_cv.txt", "w") as f:
        f.write(customized_cv)
    