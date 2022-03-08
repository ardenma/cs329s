from re import S
import requests
import logging
from time import perf_counter

import streamlit as st

from src.utils.datatypes import Query, Feedback

class StreamlitFrontend():
    def __init__(self):
        self.server_endpoint = "127.0.0.1:8000"

        if 'current_query_id' not in st.session_state:
            st.session_state['current_query_id'] = 0
        
        if 'current_response' not in st.session_state:
            st.session_state['current_response'] = None
        
        self.render()

    def render(self):
        st.title('Misinformation Detection App')
        st.text_input("Try a query below:", key="query", placeholder='Type query here!', on_change=self.send_query)

        if st.session_state.current_response is not None:
            self.render_result()

        self.render_blurb()
        self.render_feedback()
        
        st.markdown("Check out our github at https://github.com/ardenma/cs329s :)")

    def render_blurb(self):
        st.markdown("Hi and welcome to our Misinformation Detection App! Try submitting a query above, or click on the below box for more information about our app and model!")
        with st.expander(label= "More Information / Technical Details:", expanded=False):
            st.markdown("Our approach consists of 3 main steps:")
            st.markdown("1. Train a transformer model to generate embededings from query strings using data from the LIAR dataset (https://huggingface.co/datasets/liar)")
            st.markdown("2. Use this our model to generate an embedding space of the LIAR dataset examples")
            st.markdown("3. Then, given a query we can generate and embedding, match it to the K-closest embeddings in our embedding space, \
                        and the utilize a voting based approach to generate inferences (e.g. if 2/3 votes are 'true' we will return the label true!)")
            # st.subheader("Model information:")
            # st.subheader("Dataset information:")

    def render_feedback(self):
        with st.container():
            st.header("Feedback:")
            st.text_input("Please leave some feedback!", key="text_feedback", placeholder='Type feedback here!', on_change=self.send_feedback)

    def render_result(self):
        resp = st.session_state['current_response']
        st.header(f"Results:")
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Predicted Label:", resp['predicted_class'])
            col2.metric("End-to-end Latency (ms)", f"{st.session_state.last_latency_ms:.3f}", f"{st.session_state.last_latency_delta_ms:.3f}") 
            col3.metric("Inference Latency (ms)", f"{st.session_state.last_server_latency_ms:.3f}", f"{st.session_state.last_server_latency_delta_ms:.3f}")
            col4.metric("Network Latency (ms)", f"{st.session_state.last_network_latency_ms:.3f}", f"{st.session_state.last_network_latency_delta_ms:.3f}" )
            st.write("Potential class labels: (true, false, unsure)")

        st.header(f"Most similar statments:")
        st.write(resp['most_similar_examples'])

        st.header("Statement class labels and similarities:")
        for i in range(3):
            with st.container():
                col1, col2, col3 = st.columns(3)
                col1.write(resp['most_similar_examples'][i])
                col2.metric("Label", resp["example_classes"][i])
                col3.metric("Cosine Similarity", f"{resp['example_similarities'][i]:.3f}")
        
        st.header("Full API response:")
        with st.expander(label= "Full response:", expanded=False):
            st.write(resp)

    def send_query(self):
        query = Query(id=st.session_state.current_query_id, data=st.session_state.query).json()
        st.session_state.current_query_id += 1
        
        # Send query to server
        time_start = perf_counter()
        with st.spinner("Sending query to server. Please wait..."):
            st.session_state.current_response = requests.post(f"http://{self.server_endpoint}/app/predict", data=query).json()
        time_end = perf_counter()

        current_latency_ms = (time_end - time_start) * 1000
        current_server_latency_ms = st.session_state.current_response['diagnostics']['server_side_latency_ms']
        current_network_latency_ms = current_latency_ms - current_server_latency_ms


        if 'last_latency_ms' in st.session_state:
            st.session_state['last_latency_delta_ms'] = st.session_state.last_latency_ms - current_latency_ms
        else:
            st.session_state['last_latency_delta_ms'] = 0

        if 'last_server_latency_ms' in st.session_state:
            st.session_state['last_server_latency_delta_ms'] = st.session_state.last_server_latency_ms - current_server_latency_ms
        else:
            st.session_state['last_server_latency_delta_ms'] = 0

        if 'last_network_latency_ms' in st.session_state:
            st.session_state['last_network_latency_delta_ms'] = st.session_state.last_network_latency_ms - current_network_latency_ms
        else:
            st.session_state['last_network_latency_delta_ms'] = 0

        st.session_state.last_latency_ms = current_latency_ms
        st.session_state.last_server_latency_ms = current_server_latency_ms
        st.session_state.last_network_latency_ms = current_network_latency_ms

        logging.info(st.session_state.current_response)

    def send_feedback(self):
        feedback = Feedback(text_feedback=st.session_state.text_feedback).json()
        with st.spinner("Thanks for the feedback! Sending feedback to server. Please wait..."):
            requests.post(f"http://{self.server_endpoint}/app/feedback", data=feedback).json()
        st.success('Thanks! Your feedback was received :)')

if __name__=="__main__":
    StreamlitFrontend()