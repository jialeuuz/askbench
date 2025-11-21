from ask_eval.evaluators.common.math500 import Math500Evaluator
from ask_eval.evaluators.common.medqa import MedQAEvaluator
# from ask_eval.evaluators.common.bbh import BBHEvaluator
from ask_eval.evaluators.common.gpqa import GpqaEvaluator
from ask_eval.evaluators.degrade.medqa_de import MedQADeEvaluator
from ask_eval.evaluators.degrade.math500_de import Math500DeEvaluator
from ask_eval.evaluators.ask import AskEvaluator
from ask_eval.evaluators.ask_lone import AskLoneEvaluator
from ask_eval.evaluators.fata import FataEvaluator



# 评估器映射
EVALUATOR_MAP = {
    "math500": Math500Evaluator,
    # "bbh": BBHEvaluator,
    # "bbh_de": BBHDeEvaluator,
    "gpqa": GpqaEvaluator,
    "math500_de":Math500DeEvaluator,
    "medqa":MedQAEvaluator,
    "medqa_de":MedQADeEvaluator,
    "ask_overconfidence":AskEvaluator,
    "ask_overconfidence_math500":AskEvaluator,
    "ask_overconfidence_medqa":AskEvaluator,
    "ask_mind":AskEvaluator,
    "ask_mind_math500de":AskEvaluator,
    "ask_mind_medqade":AskEvaluator,
    "ask_mind_bbhde":AskEvaluator,
    "ask_mind_gpqade":AskEvaluator,
    "quest_bench":AskEvaluator,
    "ask_lone":AskLoneEvaluator,
    "ask_lone_bbhde":AskLoneEvaluator,
    "ask_lone_gpqade":AskLoneEvaluator,
    "ask_lone_math500de":AskLoneEvaluator,
    "ask_lone_medqade":AskLoneEvaluator,
    "fata_math500":FataEvaluator,
    "fata_medqa":FataEvaluator,
}
