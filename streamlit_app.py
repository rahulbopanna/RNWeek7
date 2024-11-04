import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

# Initialize ChatOpenAI with the API key
api_key = st.secrets["OpenAIKey"]
model = "gpt-4o-mini"
chatbot = ChatOpenAI(openai_api_key=api_key, model=model)

# Streamlit App Title
st.title("Airline Feedback Form")

# Input field for user feedback
user_feedback = st.text_input("Share your recent travel experience:", "")

# Feedback classification template
classification_template = """
Please classify the feedback into one of these categories:
1. "service_issue" - Negative feedback about the airline's services (e.g., lost luggage, poor service).
2. "external_factor" - Negative feedback due to reasons outside the airline's control (e.g., weather, airport issues).
3. "positive_experience" - Positive feedback about the airline.

Respond with only: "service_issue", "external_factor", or "positive_experience".

Feedback:
{feedback}
"""

# Create a prompt template for classification
classification_prompt = PromptTemplate(input_variables=["feedback"], template=classification_template)
classification_chain = LLMChain(llm=chatbot, prompt=classification_prompt)

# Predefined responses for each feedback type
responses = {
    "service_issue": "We are sorry to hear about your experience. Our customer service will contact you shortly.",
    "external_factor": "We appreciate your understanding regarding issues beyond our control. Thank you!",
    "positive_experience": "Thank you for your kind words! We are thrilled to hear you enjoyed your trip."
}

# Process feedback if provided
if user_feedback:
    try:
        # Classify the feedback
        feedback_category = classification_chain.run({"feedback": user_feedback})

        # Get the appropriate response based on classification
        response_text = responses.get(feedback_category, "We encountered an unexpected error in processing your feedback.")
        st.write(response_text)
        
    except Exception as e:
        st.error(f"An error occurred while processing your feedback: {e}")
