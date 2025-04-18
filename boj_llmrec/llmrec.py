import os
import pandas as pd

from .recommender import Recommender
from .llm import LLM

class Session:
    def __init__(self, user_handle: str, recommendations: list[int], llm: LLM) -> None:
        self.user_handle = user_handle
        self.llm = llm
        self.recommendations = recommendations
        self.prev_msgs = []

    def chat(self, message: str) -> str:
        response, prev_msgs = self.llm.chat(message, self.prev_msgs, self.recommendations)
        self.prev_msgs = prev_msgs
        return response

class LLMRec:
    def __init__(self, api_key: str) -> None:
        self.TOP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_PATH = os.path.join(self.TOP_PATH, 'data')
        self.MODEL_PATH = os.path.join(self.TOP_PATH, 'saved')
        self.recommender = Recommender(self.DATA_PATH)
        self.llm = LLM(api_key=api_key)
        self.problem_info = pd.read_csv(os.path.join(self.DATA_PATH, 'problem_info.csv'))
        self._load_model()

    def _load_model(self) -> None:
        model_path = os.path.join(self.MODEL_PATH, 'model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        self.recommender.load_model(model_path)

    def get_new_session(self, user_handle: str) -> Session:
        rec_ids = self.recommender.recommend(user_handle)
        rec_df = self.problem_info.set_index('problemId', drop=True).loc[rec_ids].reset_index()
        session = Session(user_handle, rec_df, self.llm)
        return session 
