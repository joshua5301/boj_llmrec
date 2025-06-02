import os
import pandas as pd

from .recommender import Recommender
from .llm import LLM

class Session:
    def __init__(self, llm: LLM, user_handle: str, profile: dict) -> None:
        self.user_handle = user_handle
        self.llm = llm
        self.profile = profile
        self.title = None
        self.prev_msgs = []

    def chat(self, message: str) -> str:
        text_response, speech_response, prev_msgs, keywords = self.llm.chat(message, self.prev_msgs, self.user_handle, self.profile)
        self.prev_msgs = prev_msgs
        if not self.title:
            self.title = self.llm.get_session_title(message, speech_response)
        return text_response, speech_response, keywords

class LLMRec:
    def __init__(self, api_key: str) -> None:
        self.TOP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_PATH = os.path.join(self.TOP_PATH, 'data')
        self.MODEL_PATH = os.path.join(self.TOP_PATH, 'saved')
        self.recommender = Recommender(self.DATA_PATH)
        self.llm = LLM(api_key=api_key, recommender=self.recommender)
        self._load_model()

    def _load_model(self) -> None:
        lightgcn_model_path = os.path.join(self.MODEL_PATH, 'LightGCN_model.pth')
        multivae_model_path = os.path.join(self.MODEL_PATH, 'MultiVAE_model.pth')
        if not os.path.exists(lightgcn_model_path):
            raise FileNotFoundError(f"Model file not found at {lightgcn_model_path}. Please train the model first.")
        if not os.path.exists(multivae_model_path):
            raise FileNotFoundError(f"Model file not found at {multivae_model_path}. Please train the model first.")
        self.recommender.load_model(lightgcn_model_path, model_type='LightGCN')
        self.recommender.load_model(multivae_model_path, model_type='MultiVAE')
        self.llm = LLM(api_key=self.llm.api_key, recommender=self.recommender)

    def get_new_session(self, user_handle: str, profile: dict) -> Session:
        session = Session(self.llm, user_handle, profile)
        return session
