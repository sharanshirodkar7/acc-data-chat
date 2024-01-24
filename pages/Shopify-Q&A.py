import time

import pandas as pd
import streamlit as st
import lancedb
from lancedb.embeddings import with_embeddings
from langchain import PromptTemplate
import predictionguard as pg
import streamlit as st

# Assuming check_password.py is fixed as previously described
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

#---------------------#
# Lance DB Setup      #
#---------------------#

# local path of the vector db
uri = "acc-demo.lancedb"
db = lancedb.connect(uri)

def embed(query, embModel):
    return embModel.encode(query)

def batch_embed_func(batch):
    return [st.session_state['en_emb'].encode(sentence) for sentence in batch]

#---------------------#
# Streamlit config    #
#---------------------#

if "login" not in st.session_state:
    st.session_state["login"] = False

# Hide the hamburger menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


#---------------------#
# Prompt Template     #
#---------------------#

thread_template = """### User:
{user}

### Assistant:
{assistant}"""

thread_prompt = PromptTemplate(template=thread_template,
    input_variables=["user", "assistant"],
)

qa_template = """### System:
Respond with a concise (1-2 sentence) answer to the user questions using the provided context and message history. Only answer based on the context. Do not provide any additional explanations.

{thread}

### User:
Question: "{question}"

Context: "{context}"

### Assistant:
"""

qa_prompt = PromptTemplate(template=qa_template,
    input_variables=["thread", "question", "context"],
)

#--------------------------#
# Streamlit Sidebar        #
#--------------------------#

logo_path = "logo.jpg"  
st.sidebar.image(logo_path, width=300, use_column_width=True)


#--------------------------#
#  Caching Functions       #
#--------------------------#

def check_cache(new_input):
    if "acc-chat" in db.table_names():
        table = db.open_table("acc-chat")
        results = table.search(embed(new_input, st.session_state['en_emb'])).limit(1).to_pandas()
        results = results[results['_distance'] < 0.2]
        if len(results) == 0:
            return False, {}
        else:
            results.sort_values(by=['_distance'], inplace=True, ascending=True)
            return True, results['answer'].values[0]
    else:
        return False, {}
        
def add_to_cache(new_input, answer):
    pre_process_data = []
    pre_process_data.append([
        new_input,
        answer
    ])
    ppdf = pd.DataFrame(pre_process_data, columns=[
        'text', 
        'answer'
    ])
    vecData = with_embeddings(batch_embed_func, ppdf)
    if "acc-chat" not in db.table_names():
        db.create_table("acc-chat", data=vecData)
    else:
        table = db.open_table("acc-chat")
        table.add(data=vecData)


#--------------------------#
# CHAT FUNCTION
#--------------------------#

def get_answer(question, thread):

    # load the collections table
    table = db.open_table("acc")
    results = table.search(embed(question, st.session_state['en_emb'])).limit(2).to_pandas()
    results = results[results['_distance'] < 0.8]

    if len(results) != 0:

        # Check the returned results
        results.sort_values(by=['_distance'], inplace=True, ascending=True)

        doc_use = ""
        for _, row in results.iterrows():
            if len(row['desc'].split(' ')) < 10:
                continue
            else:
                doc_use = row['desc']
                break
        
        # Handle no relevant docs
        if doc_use == "":
            completion = "Sorry, I can't find any relevant information to answer"
            return completion
        else:
            prompt_filled = qa_prompt.format(thread=thread, question=question, context=doc_use)

            # Respond to the user
            output = pg.Completion.create(
                model="Neural-Chat-7B",
                prompt=prompt_filled,
                max_tokens=100,
                temperature=0.1
            )
            completion = output['choices'][0]['text']

            # Check for factual consistency
            fact_score = pg.Factuality.check(
                reference=doc_use,
                text=completion
            )['checks'][0]['score']*100.0

            if fact_score > 75.0:
                completion += "\n\n**✅ Probability of being factual:** " + str("%0.1f" % fact_score) + "%"
            elif fact_score and fact_score <= 75.0:
                completion += "\n\n**⚠️ Probability of being factual:** " + str("%0.1f" % fact_score) + "%"

            return completion
        
    elif len(results) == 0 and thread != "":
        prompt_filled = qa_prompt.format(thread=thread, question=question, context="see the above message thread")
        # Respond to the user
        output = pg.Completion.create(
            model="Neural-Chat-7B",
            prompt=prompt_filled,
            max_tokens=100,
            temperature=0.1
        )
        completion = output['choices'][0]['text']
        return completion
    
    else:
        completion = "Sorry, I can't find any relevant information to answer."
        return completion


#--------------------------#
# Streamlit app            #
#--------------------------#
    
if st.session_state['login'] == False:
    if not check_password():
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # contruct prompt thread
        examples = []
        turn = "user"
        example = {}
        for m in st.session_state.messages:
            latest_message = m["content"]
            example[turn] = m["content"]
            if turn == "user":
                turn = "assistant"
            else:
                turn = "user"
                examples.append(example)
                example = {}
        if len(example) > 2:
            examples = examples[-2:]
        if len(examples) > 0:
            thread = "\n\n".join([thread_prompt.format(
                user=e["user"],
                assistant=e["assistant"]
            ) for e in examples])
        else:
            thread = ""

        # Check for PII
        with st.spinner("Checking for PII..."):
            pii_result = pg.PII.check(
                prompt=latest_message,
                replace=False,
                replace_method="fake"
            )

        # Check for injection
        with st.spinner("Checking for security vulnerabilities..."):
            injection_result = pg.Injection.check(
                prompt=latest_message,
                detect=True
            )

        # Check the cache
        incache, cached_response = check_cache(latest_message)
        if incache:
            with st.spinner("Generating an answer..."):
                time.sleep(1)
                completion = cached_response

            # display response
            for token in completion.split(" "):
                full_response += " " + token
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.075)
            message_placeholder.markdown(full_response)

        # Handle insecure states
        elif "[" in pii_result['checks'][0]['pii_types_and_positions']:
            st.warning('Warning! PII detected. Please avoid using personal information.')
            full_response = "Warning! PII detected. Please avoid using personal information."
        elif injection_result['checks'][0]['probability'] > 0.5:
            st.warning('Warning! Injection detected. Your input might result in a security breach.')
            full_response = "Warning! Injection detected. Your input might result in a security breach."

        # generate response
        else:
            with st.spinner("Generating an answer..."):
                completion = get_answer(latest_message, thread)

                # display response
                for token in completion.split(" "):
                    full_response += " " + token
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.075)
                message_placeholder.markdown(full_response)

        if not incache:
            add_to_cache(latest_message, full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
