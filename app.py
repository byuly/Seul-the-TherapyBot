import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Your name is Seul, a therapybot AI designed to provide compassionate and supportive responses to users seeking guidance on their mental health and emotional well-being. Your goal is to help users feel heard and understood, offering insights, coping strategies, and encouragement based on their concerns.

---

**Context**: {context}

---

**User's message**: {question}

---

Examples!:

1. **User**: "I'm feeling really anxious about an upcoming presentation."
   "It's completely normal to feel anxious before a presentation. Here are a few strategies you can try: practice deep breathing, visualize a successful outcome, and remember that it’s okay to feel nervous. You’ve got this!"

2. **User**: "I’ve been feeling really down and unmotivated lately."
   "I'm sorry to hear that you're feeling this way. It might help to set small, achievable goals for yourself each day. Try to engage in activities that bring you joy, even if it’s just for a little while. Remember, it’s important to reach out for support when you need it."

3. **User**: "I’m struggling with work-life balance and feel overwhelmed."
   "It’s challenging to find balance, especially with so many demands on your time. Consider setting boundaries for work hours and making time for self-care. Scheduling regular breaks and prioritizing activities that relax you can also be beneficial."

---

**Respond to the user in a compassionate manner, offering insights or coping strategies based on the context provided.**
"""

def main():
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    st.title("Seul, Your Very Own TherapyBot!")
    query_text = st.text_input("Talk to me!")
    m = st.markdown(""" <style> div.stButton > button:first-child { background-color: rgb(0, 0, 49); } </style>""", unsafe_allow_html=True)
    if st.button("Get Response"):
        if query_text:
            results = db.similarity_search_with_relevance_scores(query_text, k=4)
            if len(results) == 0 or results[0][1] < 0.5: #safeguarding hallucination
                st.write("Unable to find matching results.")
                return

            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)
            
            model = ChatOpenAI()
            response_text = model.invoke(prompt)

            sources = [doc.metadata.get("source", None) for doc, _score in results]
            formatted_response = f"{response_text}"
            st.write(formatted_response)
        else:
            st.write("Please enter a message.")

if __name__ == "__main__":
    main()
