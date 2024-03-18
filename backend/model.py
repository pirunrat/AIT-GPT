import torch
import os
from langchain import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ChatMessageHistory
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from langchain import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain

class Model:
    def __init__(self):
        try:
            self.device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

            # Prompt Section
            prompt_template = """
                Founded in 1959 as SEATO Graduate School of Engineering, it receives funding from organizations and governments around the world.
                In 1967, The Constituent Assembly of Thailand approved legislation for the Charter of the newly named Asian Institute of Technology in October.
                The Asian Institute of Technology Enabling Act was published in the Royal Thai Government Gazette in November the same year.
                AIT became independent of SEATO as an institution of higher learning empowered to grant degrees.
                {context}
                Question: {question}
                Answer:
                """.strip()

            self.PROMPT = PromptTemplate.from_template(template=prompt_template)

            # Load PDF
            nlp_docs = './SET-ICT-DSAI.pdf'
            loader = PyMuPDFLoader(nlp_docs)
            documents = loader.load()

            # Document Transformer
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=70, chunk_overlap=10)
            doc = text_splitter.split_documents(documents)

            # Text Embedding Model
            model_name = 'hkunlp/instructor-base'
            embedding_model = HuggingFaceInstructEmbeddings(model_name=model_name, model_kwargs={"device": self.device})

            # Vector Store
            vector_path = './vector-store'
            db_file_name = 'nlp_stanford'
            vectordb = FAISS.load_local(folder_path=os.path.join(vector_path, db_file_name), embeddings=embedding_model,
                                        index_name='nlp')

            self.retriever = vectordb.as_retriever()

            # Memory
            self.history = ChatMessageHistory()

            # Chain
            model_id = 'fastchat-t5-3b-v1.0'
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

            pipe = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256,
                            model_kwargs={"temperature": 0, "repetition_penalty": 1.5})

            llm = HuggingFacePipeline(pipeline=pipe)
            question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)

            query = 'Comparing both of them'
            chat_history = "Human:What is Machine Learning\nAI:\nHuman:What is Deep Learning\nAI:"

            question_generator({'chat_history': chat_history, "question": query})

            doc_chain = load_qa_chain(llm=llm, chain_type='stuff', prompt=self.PROMPT, verbose=True)
            query = "What is AIT"
            input_document = self.retriever.get_relevant_documents(query)

            doc_chain({'input_documents': input_document, 'question': query})

            memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True,
                                                    output_key='answer')

            chain = ConversationalRetrievalChain(retriever=self.retriever, question_generator=question_generator,
                                                combine_docs_chain=doc_chain, return_source_documents=True,
                                                memory=memory, verbose=True, get_chat_history=lambda h: h)

            self.chain = chain

        except KeyError as e:
            print(f'Error from model: {e}')

    def answer_question(self, question):
        try:
            answer = self.chain({"question": question})
            return answer
        except Exception as e:
            print(f'Error answering question: {e}')
            return None
