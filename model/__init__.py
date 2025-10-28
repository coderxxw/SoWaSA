from model.bsarec import BSARecModel
from model.sasrec import SASRecModel
from model.caser import CaserModel
from model.gru4rec import GRU4RecModel
from model.bert4rec import BERT4RecModel
from model.fmlprec import FMLPRecModel
from model.fearec import FEARecModel

from model.sowasa  import SoWaSARecModel

MODEL_DICT = {
    "bsarec": BSARecModel,
    "sasrec": SASRecModel,
    "caser": CaserModel,
    "gru4rec": GRU4RecModel,
    "bert4rec": BERT4RecModel,
    "fmlprec": FMLPRecModel,
    "fearec": FEARecModel,
    "sowasa": SoWaSARecModel,
    }