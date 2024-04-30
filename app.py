from astrapy.db import AstraDB
from datasets import load_dataset
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore as astra 
import os
import requests
import streamlit as st
import json

# Load API secrets
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN= os.environ.get("CF_API_TOKEN")

# init astra
db = AstraDB(
  token=ASTRA_DB_APPLICATION_TOKEN,
  api_endpoint=ASTRA_DB_API_ENDPOINT)

print(f"Connected to Astra DB: {db.get_collections()}")

# takes in many objects from sim search of different datasets, returns most similar according to the text input from streamlit web app
def generate_text_from_sim_search(char_sim_search, species_sim_search, vehicle_sim_search, planet_sim_search, starship_sim_search, url):
    prompt = f"Summarize the Star Wars data contained in the following data and return nothing else. {char_sim_search} {species_sim_search} {vehicle_sim_search} {planet_sim_search} {starship_sim_search}"
    payload = {
        "max_tokens": 400, # max of max tokens?
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

# generate story based on string from generate_text_from_sim_search function containing similarity search results + movie rating
def gen_story_from_sim_search_and_movie_rating(res, movie_rating, url):
    prompt = f"Return only a humorous {movie_rating}-rated story based off of the following and nothing else. Do not mention a homeworld. The story should have an introduction paragraph, a villain and conflict, and a conclusion paragraph: {res}"
    print(f'prompt {prompt}')
    payload = {
        "max_tokens": 2000,
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
    print(f'parsed_data {parsed_data}')
    result2 = parsed_data['result']['response']
    print(f'result2 {result2}')
    return parsed_data

dataCache = {}

# init docs arrs for each dataset in HF
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

# return docs after loading from HF w/ langchain func loaddata, loop through datasets and pull specific data columns, add a lc doc with the name and metadata tags
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
    st.markdown("""
        <style>
            .big-font {
                font-size:40px !important;
                color:green;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font"<p>Star Wars fanfic generator⭐️🔫📝🤖</p>', unsafe_allow_html=True)
    st.markdown("![jar jar oh no gif](https://media3.giphy.com/media/3owzWj2ViX6FJj5xMQ/giphy.gif)")
    st.write(":blue[This Python🐍 web🕸️ app is built👩🏻‍💻 w/ [Streamlit](https://streamlit.io/), [LangChain](https://www.langchain.com/)⛓️, [Cloudflare Workers AI](https://ai.cloudflare.com/), and [Astra DB](https://www.datastax.com/products/datastax-astra)]")
    char_inp = st.text_input(":red[Who is your favorite character in Star Wars⭐️🔫?]")
    planet_inp = st.multiselect(
        ':green[Your ideal vacation spot is]',
        ['grasslands🦁', 'mountains⛰️', 'tropical🏝️', 'jungle🌲', 'rainforests💧', 'cityscape🌆', 'caves', 'lakes🚤', 'ice canyons', 'urban🏢', 'swamp👹', 'reefs🐟', 'plains', 'volcanoes🌋', 'arid🌵', 'tundra🥶'],
        ['ice canyons', 'swamp👹', 'cityscape🌆'])
    planet_inp = str(planet_inp)
    st.markdown("![do it meme](https://media1.giphy.com/media/3o84sw9CmwYpAnRRni/giphy.gif)")
    char_labels = ["Jabba the Hutt", "Admiral Ackbar", "Jar Jar Binks", "Count Dooku", "Obi-wan Kenobi", "Poe Dameron"]
    vehicle_inp = st.select_slider(':orange[On a scale from Jabba the Hutt to Poe Dameron, how 🔥 is the villain✈️ of your fanfic?]', options=char_labels)
    

    starship_inp = st.text_input(":yellow[Describe a starship⭐️🚀 to drive🚗]")
    species_inp = st.text_input(":pink[Describe u✌️🥰]")
    st.markdown("![So uncivilized gif](https://media0.giphy.com/media/xTiIzkLOknx8ELm4Ok/200.gif)")
    sal_labels = ["R2D2--like the innocent astromech droid, suitable for all", "Chewbacca moaning--could go over young heads", "so uncivilized! May be inappropriate for species < 13", "Jar Jar || Jabba sans-clothes--may contain content !suitable for < 17 w/o parental consent"]
    salaciousness_met = st.select_slider(":orange[On a scale from R2-D2😇 to Jar Jar sans-clothes🔥, how salacious🥵 should the fanfic be😘?]", options= sal_labels)

    st.markdown("![screaming r2d2 gif](https://assets.teenvogue.com/photos/572a3302321c4faf6ae8a317/16:9/w_2580,c_limit/R2SCREAM.gif)") #screaming r2

    # All models at https://developers.cloudflare.com/workers-ai/models/
    img_model = st.selectbox(
    "Choose your character (Text-To-Image model):",
        options=(
            "@cf/lykon/dreamshaper-8-lcm",
            "@cf/bytedance/stable-diffusion-xl-lightning",
            "@cf/stabilityai/stable-diffusion-xl-base-1.0",
        ),
    )
    text_model = st.selectbox(
        "Choose your weapon (Text generation model):",
        options= (
            "@cf/mistral/mistral-7b-instruct-v0.1",
            "@cf/meta/llama-2-7b-chat-fp16",
            "@cf/meta/llama-2-7b-chat-int8"
        )
    )
    url =f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{text_model}"
    if char_inp is not None and vehicle_inp is not None and species_inp is not None and st.button('Generate🤖'):
        # load dataset once on page load/on server start
        with st.spinner('Processing📈...'):
            st.markdown("![i will finish what you started gif](https://y.yarn.co/2f4fabe4-6046-4bbd-96bd-3a3ccf9853c9_text.gif)")
            ret_docs() # load data, add a LangChain document with the name and metadata tags
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
            planet_inserted_ids = vstore_planet.add_documents(plan_docs) # planets
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
            species_inserted_ids = vstore_species.add_documents(species_docs) # species
            print(f"\nInserted species {len(species_inserted_ids)} documents.")

            # k = # nearest neighbors/most similar vectors to retrieve/query in a vector db
            char_sim_search = vstore_char.similarity_search(char_inp, k = 3) 
            vehicle_sim_search = vstore_vehicle.similarity_search(vehicle_inp, k = 3) 
            planet_sim_search = vstore_planet.similarity_search(planet_inp, k = 3) 
            starship_sim_search = vstore_starship.similarity_search(starship_inp, k = 3) 
            species_sim_search = vstore_species.similarity_search(species_inp, k = 3) 
            
            print(f'char_sim_search {char_sim_search}, vehicle_sim_search {vehicle_sim_search}, planet_sim_search {planet_sim_search}, starship_sim_search {starship_sim_search}, species_sim_search {species_sim_search}') #  sim search results

            # map salacious rating strings to actual movie ratings like g, pg, pg13, r
            sal_dict = dict(g= sal_labels[0], pg = sal_labels[1], pg13 = sal_labels[2], r = sal_labels[3])
            movie_rating = list(sal_dict.keys())[list(sal_dict.values()).index(salaciousness_met)]

            sum_query = generate_text_from_sim_search(char_sim_search, species_sim_search, vehicle_sim_search, planet_sim_search, starship_sim_search, url)
            print(sum_query)

            story = gen_story_from_sim_search_and_movie_rating(sum_query, movie_rating, url)['result']['response']
            
            img_prompt = f"You are a world-renowned painter of lighthearted Star Wars-related art. Generate a humorous {movie_rating}-rated image relating to {sum_query}"
            #img_prompt = f"Generate a seductive image of Jar Jar Binks"
            img_url =f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{img_model}"
            headers = {
                "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
            }
            resp = requests.post(
                img_url,
                headers=headers,
                json={"prompt": img_prompt},
            )
            st.image(resp.content, caption=f"AI-generated image from {img_model}") #bytes lmao
            
            html_str = f"""
            <p style="font-family:Comic Sans; color:Pink; font-size: 18px;">{story}</p>
            """
            st.markdown(html_str, unsafe_allow_html=True)
    st.write("Made w/ ❤️ in Hawaii 🏝️🌺 && Portland ☔️🌳")
    st.write("✅ out the [code on GitHub](https://github.com/elizabethsiegle/star-wars-fanfic-generator-streamlit-astra-cf)")


if __name__ == "__main__":
    main()