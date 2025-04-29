import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from langchain_core.language_models import LLM  
from pydantic import Field

# Paths
FINETUNED_MODEL_PATH = "finetuned-llama3-products"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DB_FAISS_PATH = "vectorstore/db_faiss"

HF_TOKEN = os.environ.get("HF_TOKEN")

# Custom LLM wrapper
class CustomLLM(LLM):
    model: object = Field(exclude=True)
    tokenizer: object = Field(exclude=True)

    @property
    def _llm_type(self) -> str:
        return "custom-llm"

    def _call(self, prompt: str, **kwargs: any) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def _identifying_params(self):
        return {}

# Load model
@st.cache_resource
def load_llm():
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(
        base_model,
        FINETUNED_MODEL_PATH,
        device_map="auto",
        offload_folder="./offload",
        offload_state_dict=True,
    )

    return CustomLLM(model=model, tokenizer=tokenizer)

# Load vectorstore
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Set custom prompt
def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

CUSTOM_PROMPT_TEMPLATE = """
🎯 You are a top-tier creative copywriter, specializing in writing engaging and persuasive product ads.

Your mission is to **generate a captivating product ad** using the given information below.
Be creative, highlight unique selling points, and make it emotionally appealing to the audience. 🌟

---

🏢 Company Context:
{context}

🛍️ Product Description:
{question}

🎨 Motif (Tone/Emotion to Use):
Excitement

---

✨ Rules:
- Start with a catchy headline
- Highlight key features naturally
- Make it feel persuasive but NOT robotic
- Use the motif/emotion strongly
- Keep it concise (max 120 words)
- Sprinkle emojis naturally
- Avoid any unrelated content

---

Now, generate the product ad below:
"""

# Streamlit app
def main():
    st.title("🚀 Enterprise Product Ad Generator")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Describe your new product idea ✍️")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        try:
            vectorstore = get_vectorstore()
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]
            result_to_show = result + "\n\n📚 Source Docs:\n" + str(source_documents)

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()