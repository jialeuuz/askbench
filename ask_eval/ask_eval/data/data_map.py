from ask_eval.data.jsonl import JsonlLoader

# Dataset loader registry
LOADER_MAP = {
    "math500": JsonlLoader,
    "medqa":JsonlLoader,
    "ask_overconfidence":JsonlLoader,
    "ask_overconfidence_math500":JsonlLoader,
    "ask_overconfidence_medqa":JsonlLoader,
    "ask_overconfidence_gpqa":JsonlLoader,
    "ask_overconfidence_bbh":JsonlLoader,
    "ask_mind":JsonlLoader,
    "ask_mind_math500de":JsonlLoader,
    "ask_mind_medqade":JsonlLoader,
    "ask_mind_bbhde":JsonlLoader,
    "ask_mind_gpqade":JsonlLoader,
    "quest_bench":JsonlLoader,
    "in3_interaction":JsonlLoader,
    "bbh":JsonlLoader,
    "gpqa":JsonlLoader,
    "fata_math500":JsonlLoader,
    "fata_medqa":JsonlLoader,
    "healthbench":JsonlLoader,
}
