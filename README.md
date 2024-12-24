# Using TAPO demo

## Try now!

We provide an IPYNB file in 'demo' folder for easily using TAPO.

To run the provided code snippets, you need to supply specific access credentials. Here's a breakdown of what you need:

1. **Hugging Face Hub Access Token**:
   - You need to obtain an access token from Hugging Face Hub.
   - Replace `'YOUR_ACCESS_TOKEN'` with your actual token in the `login` function:
     ```python
     from huggingface_hub import login
     login('YOUR_ACCESS_TOKEN', add_to_git_credential=True)
     ```
   - This token is used to authenticate and access the resources available on Hugging Face.

2. **OpenAI API Key**:
   - You need an API key from OpenAI to use their services.
   - Replace `'YOUR_API_KEY'` with your actual API key in the `OpenAI` client setup:
     ```python
     from openai import OpenAI
     client = OpenAI(api_key='YOUR_API_KEY')
     ```

3. **Model ID**:
   - Specify the model ID you intend to use from OpenAI.
   - Replace `'YOUR_MODEL_ID'` with the specific model ID:
     ```python
     model_id = 'YOUR_MODEL_ID'
     ```

Make sure you have these credentials set up in your environment to successfully authenticate and run the code.

**Code Sturcture**

- data
    |- dataset
    |- gsm.py
    |- mutation_prompts.py
    |- thinking_styles.py
    |- types.py
- TAPO_main.ipynb: notebook demonstrating TAPO

**Dataset Format Description**

In this codebase, each record in the dataset is stored in JSON format and contains two main fields: `question` and `answer`. Below is a detailed description of the format:

```python
{
  "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
  "answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3"
}
```

