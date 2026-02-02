from ask_eval.evaluators.common.math500 import Math500Evaluator
from ask_eval.evaluators.common.medqa import MedQAEvaluator
from ask_eval.evaluators.common.bbh import BBHEvaluator
from ask_eval.evaluators.common.gpqa import GpqaEvaluator
from ask_eval.evaluators.ask import AskEvaluator
from ask_eval.evaluators.in3_interaction import In3InteractionEvaluator
from ask_eval.evaluators.healthbench import HealthBenchEvaluator

# Evaluator registry
EVALUATOR_MAP = {
    "math500": Math500Evaluator,
    "bbh": BBHEvaluator,
    "gpqa": GpqaEvaluator,
    "medqa":MedQAEvaluator,
    "ask_overconfidence":AskEvaluator,
    "ask_overconfidence_math500":AskEvaluator,
    "ask_overconfidence_medqa":AskEvaluator,
    "ask_overconfidence_gpqa":AskEvaluator,
    "ask_overconfidence_bbh":AskEvaluator,
    "ask_mind":AskEvaluator,
    "ask_mind_math500de":AskEvaluator,
    "ask_mind_medqade":AskEvaluator,
    "ask_mind_bbhde":AskEvaluator,
    "ask_mind_gpqade":AskEvaluator,
    "quest_bench":AskEvaluator,
    "in3_interaction":In3InteractionEvaluator,
    "fata_math500":AskEvaluator,
    "fata_medqa":AskEvaluator,
    "healthbench":HealthBenchEvaluator,
}
