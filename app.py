import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

st.title('Return-to-Work Job Listings')

# Initialize RAG pipeline
tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/rag-token-nq")
rag_pipeline = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

# Function to fetch job listings (example: scraping from Indeed)
def fetch_jobs(query):
    url = f"https://www.indeed.com/jobs?q={query}&l="
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    jobs = soup.find_all('div', class_='jobsearch-SerpJobCard')
    job_list = []
    for job in jobs:
        title = job.find('a', class_='jobtitle').text.strip()
        company = job.find('span', class_='company').text.strip()
        location = job.find('div', class_='location').text.strip()
        job_list.append(f"{title} at {company} in {location}")
    return job_list

# Function to get detailed descriptions using RAG
def get_job_descriptions(jobs):
    descriptions = []
    for job in jobs:
        input_text = f"Describe the job role: {job}"
        result = rag_pipeline(input_text, max_length=200)
        descriptions.append(result[0]['generated_text'])
    return descriptions

query = st.text_input('Enter job search query:', 'return to work')
if query:
    job_results = fetch_jobs(query)
    if job_results:
        st.write(f"Found {len(job_results)} jobs for '{query}':")
        detailed_descriptions = get_job_descriptions(job_results)
        for job, description in zip(job_results, detailed_descriptions):
            st.write(f"**{job}**\n{description}\n")
    else:
        st.write('No jobs found.')
