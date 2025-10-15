from abc import ABC, abstractmethod
import pandas as pd
class Decomposer(ABC):
    def __init__(self, cfg: dict): self.cfg = cfg
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """戻り値は overlaylib.io.OutputBundle"""