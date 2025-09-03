from ask_eval.evaluators.common.math500 import Math500Evaluator
from ask_eval.evaluators.common.medqa import MedQAEvaluator
from ask_eval.evaluators.degrade.medqa_de import MedQADeEvaluator
from ask_eval.evaluators.degrade.math500_de import Math500DeEvaluator
from ask_eval.evaluators.ask import AskEvaluator



# 评估器映射
EVALUATOR_MAP = {
    "math500": Math500Evaluator,
    "math500_de":Math500DeEvaluator,
    "medqa":MedQAEvaluator,
    "medqa_de":MedQADeEvaluator,
    "ask_yes":AskEvaluator,
    "ask_mind":AskEvaluator,
    "ask_lone":AskEvaluator,
}