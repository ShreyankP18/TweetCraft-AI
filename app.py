import streamlit as st
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated, Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage
import operator
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import re


load_dotenv()


st.set_page_config(
    page_title="TweetCraft AI",
    page_icon="✎ᝰ.",
    layout="centered"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap'); /* Fancy font for TweetCraft */

    body {
        font-family: 'Inter', sans-serif;
        background-color: #0E1117; /* Streamlit's default dark */
        color: #E0E0E0;
    }

    .main .block-container {
        padding: 1.5rem 1rem;
        max-width: 720px;
    }

    /* Header Styling */
    .header {
        text-align: center;
        margin-bottom: 2.5rem;
    }
    .header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .header .fancy-title {
        font-family: 'Pacifico', cursive; /* Applying fancy font */
        color: #FFFFFF; /* White color */
        font-weight: 400;
    }
    .header .ai-title {
        font-family: 'Inter', sans-serif; /* Normal font */
        color: #00A3FF; /* Blue color */
        font-weight: 700;
    }
    .header p {
        font-size: 1.1rem;
        color: #B0B0B0;
        max-width: 500px;
        margin: auto;
    }

    /* Input Widgets Styling */
    .stTextInput label {
        font-weight: 600;
        color: #E0E0E0;
        margin-bottom: 0.25rem;
    }
    .stTextInput input {
        background-color: #262730;
        color: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #333333;
    }
    div[data-testid="stForm"] {
        border: none;
        padding: 0;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(45deg, #007BFF, #00A3FF);
        color: white;
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
        border: none;
        margin-top: 1rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3);
        transform: translateY(-2px);
    }

    /* Tweet Output Card */
    .tweet-card {
        background-color: #1E1E1E;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 2rem;
        border: 1px solid #333333;
    }
    .tweet-header { display: flex; align-items: center; margin-bottom: 1rem; }
    .tweet-header img { width: 48px; height: 48px; border-radius: 50%; margin-right: 12px; }
    .tweet-header .author { font-weight: 700; color: #E0E0E0; }
    .tweet-header .handle { color: #A0A0A0; }
    .tweet-body { font-size: 1.1rem; line-height: 1.6; color: #FFFFFF; }
    .tweet-body .hashtag { color: #1DA1F2; font-weight: 500; }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def get_llms():
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"): return None
    try:
        generator_llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='meta-llama/Meta-Llama-3-8B-Instruct', task='text-generation'))
        evaluator_llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.3', task='text-generation'))
        optimizer_llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.3', task='text-generation'))
        return generator_llm, evaluator_llm, optimizer_llm
    except Exception as e:
        st.error(f"Models could not be loaded: {e}")
        return None

llms = get_llms()
if llms:
    generator_llm, evaluator_llm, optimizer_llm = llms

class TweetEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(...)
    feedback: str = Field(...)
parser_optimizer = PydanticOutputParser(pydantic_object=TweetEvaluation)

class TweetState(TypedDict):
    topic: str; mood: Optional[str]; style_account: Optional[str]
    tweet: str; evaluation: str; feedback: str; iteration: int
    max_iteration: int; tweet_history: Annotated[list[str], operator.add]

def generate_tweet(state: TweetState):
    prompt = f'Write an engaging, viral-worthy tweet about: "{state["topic"]}".'
    if state.get("mood"): prompt += f' The tweet must have a **{state["mood"]}** tone.'
    if state.get("style_account"): prompt += f' Write it in the style of Twitter user @{state["style_account"]}.'
    messages = [SystemMessage(content="You are a world-class social media expert known for crafting viral tweets."), HumanMessage(content=prompt)]
    response = generator_llm.invoke(messages).content
    return {'tweet': response, 'tweet_history': [response], 'iteration': 1}

def evaluate_tweet(state: TweetState):
    messages = [SystemMessage(content="You are a harsh but fair social media critic."), HumanMessage(content=f'Critique this tweet: "{state["tweet"]}" for the topic "{state["topic"]}". Tone should be "{state.get("mood", "any")}". Strictly return JSON:\n{parser_optimizer.get_format_instructions()}')]
    raw_output = evaluator_llm.invoke(messages).content
    try:
        response = parser_optimizer.parse(raw_output)
        return {'evaluation': response.evaluation, 'feedback': response.feedback}
    except Exception: return {'evaluation': 'needs_improvement', 'feedback': 'Critic failed to provide valid feedback.'}

def optimize_tweet(state: TweetState):
    prompt = f'Based on the feedback "{state["feedback"]}", rewrite this tweet to make it better.\nOriginal Tweet: "{state["tweet"]}"\nTopic: "{state["topic"]}"'
    if state.get("mood"): prompt += f'\nEnsure the new tweet maintains a **{state["mood"]}** tone.'
    messages = [SystemMessage(content="You are a master tweet editor who improves content based on feedback."), HumanMessage(content=prompt)]
    response = optimizer_llm.invoke(messages).content
    iteration = state['iteration'] + 1
    return {'tweet': response, 'iteration': iteration, 'tweet_history': [response]}

def route_evaluation(state: TweetState):
    return END if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration'] else 'optimize'

@st.cache_resource
def compile_graph():
    graph = StateGraph(TweetState)
    graph.add_node('generate', generate_tweet); graph.add_node('evaluate', evaluate_tweet); graph.add_node('optimize', optimize_tweet)
    graph.add_edge(START, 'generate'); graph.add_edge('generate', 'evaluate')
    graph.add_conditional_edges('evaluate', route_evaluation, {'optimize': 'optimize', END: END})
    graph.add_edge('optimize', 'evaluate')
    return graph.compile()
workflow = compile_graph()



st.markdown("""
<div class="header">
    <h1><span class="fancy-title">TweetCraft</span> <span class="ai-title">AI</span> ✎ᝰ.</h1>
    <p>Turn your ideas into viral tweets in seconds. Just provide a topic and let the AI handle the rest.</p>
</div>
""", unsafe_allow_html=True)


with st.form("tweet_form"):
    topic = st.text_input("What's the topic of your tweet?", placeholder="e.g., The future of Artificial Intelligence")
    mood = st.text_input("What's the mood? (Optional)", placeholder="e.g., Funny, inspirational, serious, sarcastic")
    style_account = st.text_input("Copy the style of a Twitter user? (Optional)", placeholder="e.g., naval (without @)")
    
    
    submitted = st.form_submit_button("🚀 Craft My Tweet")

if submitted:
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"): st.error("Hugging Face API Token not found in your .env file!")
    elif not topic: st.warning("Please enter a topic to craft your tweet!", icon="⚠️")
    elif not llms: st.error("Models failed to load. Please check your API key.")
    else:
        initial_state = {"topic": topic, "mood": mood, "style_account": style_account, "max_iteration": 3}
        with st.spinner("Analyzing topic... Crafting initial draft... Reviewing and optimizing..."):
            final_state = workflow.invoke(initial_state)
            final_tweet = final_state.get('tweet', "An error occurred. Please try again.")

            
            highlighted_tweet = re.sub(r'(#\w+)', r'<span class="hashtag">\1</span>', final_tweet)

        
        st.markdown(f"""
        <div class="tweet-card">
            <div class="tweet-header">
                <img src="https://pbs.twimg.com/profile_images/1780044485541699584/p78MCn3B_400x400.jpg" alt="Avatar">
                <div>
                    <span class="author">AI Influencer</span><br>
                    <span class="handle">@aicrafted_</span>
                </div>
            </div>
            <div class="tweet-body">
                {highlighted_tweet}
            </div>
        </div>
        """, unsafe_allow_html=True)