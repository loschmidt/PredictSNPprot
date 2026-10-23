from enum import StrEnum


class Predictor(StrEnum):
    THREE_CNET = "3Cnet"
    ESM1B = "ESM1b"
    PROVEAN = "PROVEAN"
    VESPAL = "VESPAl"
    VESPAG = "VespaG"
    VARIPRED = "VariPred"
    PHACT = "PHACT"
    PHACTBOOST = "PHACTboost"
    GEMME = "GEMME"


FEATURES: list[str] = [p.value for p in Predictor]
