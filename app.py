import os

import replicate
import streamlit as st
from dotenv import load_dotenv
from elevenlabs import generate
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")

llm = OpenAI(temperature=0.9)

def generate_recipe(food, calories):
    prompt = PromptTemplate(
        input_variables=["food", "calories"],
        template=""" 
         You are an experienced chef, create a recipe for the following food {food} that is under {calories} calories.
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({
    'food': food,
    'calories': calories
    })


def generate_audio(text, voice):
    audio = generate(text=text, voice=voice, api_key=eleven_api_key)
    return audio


def generate_images(food):
    output = replicate.run(
        "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
        input={"prompt": food}
    )
    return output

def app():
    st.set_page_config(page_title="Recipe Pal")
    st.title("Recipe Pal")
    st.header("Recipe Generation")

    with st.sidebar:
        options = ["Bella", "Antoni", "Arnold", "Adam", "Domi", "Elli", "Josh", "Rachel", "Sam"]
        voice = st.selectbox("Select a voice", options)

    with st.form(key='my_form'):
        food = st.text_input(label="Enter what you want to cook.", placeholder="Enter a food you want to cook")
        calories = st.number_input(label="Enter the calorie limit.", min_value=1, max_value=3000, value=200)
        submitted = st.form_submit_button("Generate Recipe")


    if submitted:
        recipe = generate_recipe(food, calories) 
        audio = generate_audio(recipe, voice)

        st.subheader("Recipe")
        st.markdown(recipe)
        st.subheader("Audio")
        st.audio(audio, format='audio/mp3')
        st.subheader("Images")
        images = generate_images(food)
        for image in images:
            st.image(image, use_column_width=True)
        


if __name__ == '__main__':
    app()