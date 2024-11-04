import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

# Function to initialize the chatbot
def initialize_chatbot(api_key):
    return ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini")

# Function to create the classification chain
def create_classification_chain(chatbot):
    classification_template = """
    Classify the feedback into one of the following categories:
    1. "service_issue" - Negative feedback about the airline's services (e.g., lost luggage, poor service).
    2. "external_factor" - Negative feedback due to reasons outside the airline's control (e.g., weather, airport issues).
    3. "positive_experience" - Positive feedback about the airline.

    Respond with only: "service_issue", "external_factor", or "positive_experience".

    Feedback:
    {feedback}
    """
    classification_prompt = PromptTemplate(input_variables=["feedback"], template=classification_template)
    return LLMChain(llm=chatbot, prompt=classification_prompt)

# Function to classify feedback and generate response
def classify_feedback(classification_chain, feedback):
    try:
        category = classification_chain.run({"feedback": feedback})
        return category
    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
        return None

# Function to generate a response based on feedback category
def get_response(category):
    responses = {
        "service_issue": "We are sorry to hear about your experience. Our customer service will contact you shortly.",
        "external_factor": "We appreciate your understanding regarding issues beyond our control. Thank you!",
        "positive_experience": "Thank you for your kind words! We are thrilled to hear you enjoyed your trip."
    }
    return responses.get(category, "We encountered an unexpected error in processing your feedback.")

# Main function to run the Streamlit app
def main():
    st.title("Airline Feedback Form")
    
    api_key = st.secrets["OpenAIKey"]
    chatbot = initialize_chatbot(api_key)
    classification_chain = create_classification_chain(chatbot)

    # Input field for user feedback
    user_feedback = st.text_input("Share your recent travel experience:", "")

    if user_feedback:
        category = classify_feedback(classification_chain, user_feedback)
        if category:
            response_text = get_response(category)
            st.write(response_text)

# Run the app
if __name__ == "__main__":
    main()
