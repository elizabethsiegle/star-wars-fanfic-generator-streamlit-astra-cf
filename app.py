from astrapy.db import AstraDB
import base64
from datasets import load_dataset
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
#from langchain_community.vectorstores import AstraDB as astra 
from langchain_astradb import AstraDBVectorStore as astra 
from openai import OpenAI
import os
import requests
import streamlit as st
from PIL import Image
import json



def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')


# Load API secrets
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN= os.environ.get("CF_API_TOKEN")

# Initialization
db = AstraDB(
  token=ASTRA_DB_APPLICATION_TOKEN,
  api_endpoint=ASTRA_DB_API_ENDPOINT)

print(f"Connected to Astra DB: {db.get_collections()}")

url =f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/meta/llama-2-7b-chat-fp16"

def query_llama1(char_sim_search, species_sim_search, vehicle_sim_search, planet_sim_search, starship_sim_search):
    prompt = f"Return only a summarization of the Star Wars data contained in the following JSON objects. A character can only be from one planet: {char_sim_search} {species_sim_search} {vehicle_sim_search} {planet_sim_search} {starship_sim_search}"
    print(f'prompt {prompt}')
    payload = {
        "max_tokens": 300,
        "prompt": prompt,
        "raw": False,
        "stream": False
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    parsed_data = json.loads(response.text)
    result = parsed_data['result']['response']
    print(result)
    return result

def query_llama2(res):
    prompt = f"Return only a story based off of the following: {res}"
    print(f'prompt {prompt}')
    payload = {
        "max_tokens": 9999,
        "prompt": prompt,
        "raw": False,
        "stream": False
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    parsed_data = json.loads(response.text)
    result2 = parsed_data['result']['response']
    print(result2)
    return result2

def query_sdxl(payload):
  api_url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/bytedance/stable-diffusion-xl-lightning"
  response = requests.post(api_url, headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"}, json=payload)
  return response.json()

dataCache = {}

char_docs = []
veh_docs = []
plan_docs = []
ss_docs=[]
species_docs = []
@st.cache_data
def loaddata(dataset):  #lizziepikachu/starwars_characters"
    if dataset in dataCache.keys():
        return dataCache[dataset]
    data = load_dataset(dataset)["train"]
    dataCache[dataset] = data
    return data

def ret_docs():
    sw_char_huggingface_dataset = loaddata("lizziepikachu/starwars_characters")
    sw_vehicle_huggingface_dataset = loaddata("lizziepikachu/starwars_vehicles")
    sw_planet_huggingface_dataset = loaddata("lizziepikachu/starwars_planets")
    sw_starship_huggingface_dataset = loaddata("lizziepikachu/starwars_starships")
    sw_species_huggingface_dataset = loaddata("lizziepikachu/starwars_species")
    print(f"An example entry from Hugging Face dataset: {sw_char_huggingface_dataset[0]}")
    for entry in sw_char_huggingface_dataset:
        metadata = {"name": entry["name"], "species": entry["species"], "gender": entry["gender"], "hair_color": entry["hair_color"], "homeworld": entry["homeworld"]}
    
        # Add a LangChain document with the name and metadata tags
        doc = Document(page_content=entry["name"], metadata=metadata)
        char_docs.append(doc)

    for entry in sw_vehicle_huggingface_dataset:
        metadata = {"name": entry["name"], "manufacturer": entry["manufacturer"], "cost_in_credits": entry["cost_in_credits"] }
        # Add a LangChain document with the name and metadata tags
        doc = Document(page_content=entry["name"], metadata=metadata)
        veh_docs.append(doc)

    for entry in sw_planet_huggingface_dataset:
        metadata = {"name": entry["name"], "climate": entry["climate"], "terrain": entry["terrain"], "population": entry["population"] }
        # Add a LangChain document with the name and metadata tags
        doc = Document(page_content=entry["name"], metadata=metadata)
        plan_docs.append(doc)

    for entry in sw_starship_huggingface_dataset:
        metadata = {"name": entry["name"], "manufacturer": entry["manufacturer"], "cost_in_credits": entry["cost_in_credits"] }
        # Add a LangChain document with the name and metadata tags
        doc = Document(page_content=entry["name"], metadata=metadata)
        ss_docs.append(doc)
    
    for entry in sw_species_huggingface_dataset:
        metadata = {"name": entry["name"], "classification": entry["classification"], "designation": entry["designation"], "language": entry["language"], "homeworld": entry["homeworld"]  }
        # Add a LangChain document with the name and metadata tag
        doc = Document(page_content=entry["name"], metadata=metadata)
        species_docs.append(doc)

print(f'char_docs {char_docs}, veh_docs {veh_docs}, plan_docs {plan_docs}, ss_docs {ss_docs}, species_docs {species_docs}') #


# Set up Streamlit app
def main():
    st.image("leiavader-poe.jpeg", width=400)
    st.write("⭐️🔫📝")
    st.header("Generate Star Wars fanfiction")
    st.write("This Python🐍 web🕸️ app is built👩🏻‍💻 w/ [Streamlit](https://streamlit.io/), [LangChain](https://www.langchain.com/)⛓️, [CloudFlare Workers AI](https://ai.cloudflare.com/), and [Astra DB](https://www.datastax.com/products/datastax-astra)")
    # User input: PDF file upload
    char_inp = st.text_input("Who is your favorite character in Star Wars?")
    planet_inp = st.text_input("Describe a planet you'd want to read about")
    vehicle_inp = st.text_input("Describe a vehicle you'd like to ride in")
    starship_inp = st.text_input("Describe a starship you'd like to ride in")
    species_inp = st.text_input("Describe a species you'd like to read about")
    if char_inp is not None and vehicle_inp is not None and st.button('enter✅'):
        # load dataset once on page load/on server start
        with st.spinner('Processing input📈...'):
            ret_docs()
            embedding_function = OpenAIEmbeddings()
            vstore_char = astra(
                embedding=embedding_function,
                collection_name="sw_char",
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
            )
            char_inserted_ids = vstore_char.add_documents(char_docs) # chars
            print(f"\nInserted Char {len(char_inserted_ids)} documents.")

            vstore_vehicle = astra(
                embedding=embedding_function,
                collection_name="sw_veh",
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
            )
            vehicle_inserted_ids = vstore_vehicle.add_documents(veh_docs) # vehs
            print(f"\nInserted vehicle {len(vehicle_inserted_ids)} documents.")

            vstore_planet = astra(
                embedding=embedding_function,
                collection_name="sw_plan",
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
            )
            planet_inserted_ids = vstore_planet.add_documents(plan_docs) # plans
            print(f"\nInserted planet {len(planet_inserted_ids)} documents.")

            vstore_starship = astra(
                embedding=embedding_function,
                collection_name="sw_ss",
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
            )
            starship_inserted_ids = vstore_starship.add_documents(ss_docs) # ss
            print(f"\nInserted starship {len(starship_inserted_ids)} documents.")

            vstore_species = astra(
                embedding=embedding_function,
                collection_name="sw_species",
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
            )
            species_inserted_ids = vstore_species.add_documents(species_docs) # ss
            print(f"\nInserted species {len(species_inserted_ids)} documents.")

            char_sim_search = vstore_char.similarity_search(char_inp, k = 3) 
            vehicle_sim_search = vstore_vehicle.similarity_search(vehicle_inp, k = 3) 
            planet_sim_search = vstore_planet.similarity_search(planet_inp, k = 3) 
            starship_sim_search = vstore_starship.similarity_search(starship_inp, k = 3) 
            species_sim_search = vstore_species.similarity_search(species_inp, k = 3) 
            
            print(f'char_sim_search {char_sim_search}, vehicle_sim_search {vehicle_sim_search}, planet_sim_search {planet_sim_search}, starship_sim_search {starship_sim_search}, species_sim_search {species_sim_search}') #  

            # story = query_hermes({
            #     "prompt": f"You are a world-renowned Star Wars fanfiction writer. Generate a Star Wars fanfiction story about a character like {char_sim_search} who meets a character of the {species_sim_search} species who drives a vehicle like {vehicle_sim_search}. To leave the planet, they encounter characters on a planet like {planet_sim_search} who leave on a starship like {starship_sim_search}", # on a planet like {planet_sim_search} drive a starship like a {starship_sim_search}

            # })
            # story = story['result']["response"]
            # print(story)
            sum_query = query_llama1(char_sim_search, species_sim_search, vehicle_sim_search, planet_sim_search, starship_sim_search)
            print(sum_query)
            story = query_llama2(sum_query)
            print(story)

            # img = query_sdxl({
            #     "prompt": f"You are a world-renowned painter of Star Wars-related art. Generate an image about a character like {char_sim_search} in front of a vehicle like {vehicle_sim_search}" # or a starship like {starship_sim_search} 

            # # })
            # print(img)
            # # img = Image.open(img)
            # st.image(img)
            
            html_str = f"""
            <p style="font-family:Arial; color:Pink; font-size: 16px;">Story: {story}</p>
            """
            st.markdown(html_str, unsafe_allow_html=True)
    st.write("Made w/ ❤️ in Hawaii 🏝️🌺")
    st.write("✅ out the [code on GitHub](https://github.com/elizabethsiegle/star-wars-fanfic-generator-streamlit-astra-cf)")


if __name__ == "__main__":
    main()