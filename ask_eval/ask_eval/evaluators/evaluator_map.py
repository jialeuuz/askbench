from ask_eval.evaluators.common.math500 import Math500Evaluator
from ask_eval.evaluators.common.aime import AimeEvaluator
from ask_eval.evaluators.common.medqa import MedQAEvaluator
from ask_eval.evaluators.common.bbh import BBHEvaluator
from ask_eval.evaluators.degrade.medqa_de import MedQADeEvaluator
from ask_eval.evaluators.degrade.math500_de import Math500DeEvaluator
from ask_eval.evaluators.ask import AskEvaluator



# 评估器映射
EVALUATOR_MAP = {
    "math500": Math500Evaluator,
    "aime2025": AimeEvaluator,
    "bbh": BBHEvaluator,
    "bbh_de": BBHEvaluator,
    "aime2025_de": AimeEvaluator,
    "math500_de":Math500DeEvaluator,
    "medqa":MedQAEvaluator,
    "medqa_de":MedQADeEvaluator,
    "ask_yes":AskEvaluator,
    "ask_mind":AskEvaluator,
    "ask_mind_math500de":AskEvaluator,
    "ask_mind_medqade":AskEvaluator,
    "ask_mind_aime2025de":AskEvaluator,
    "ask_mind_bbhde":AskEvaluator,
    "quest_bench":AskEvaluator,
    "ask_lone":AskEvaluator,
}