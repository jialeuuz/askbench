from ask_eval.data.jsonl import JsonlLoader

# 数据加载器映射
LOADER_MAP = {
    "math500": JsonlLoader,
    "math500_de":JsonlLoader,
    "medqa":JsonlLoader,
    "medqa_de":JsonlLoader,
    "ask_yes":JsonlLoader,
    "ask_mind":JsonlLoader,
    "ask_lone":JsonlLoader,
    "ask_lone_bbhde":JsonlLoader,
    "ask_lone_gpqade":JsonlLoader,
    "ask_lone_math500de":JsonlLoader,
    "ask_lone_medqade":JsonlLoader,
    "ask_mind_math500de":JsonlLoader,
    "ask_mind_medqade":JsonlLoader,
    "ask_mind_bbhde":JsonlLoader,
    "ask_mind_gpqade":JsonlLoader,
    "quest_bench":JsonlLoader,
    "bbh":JsonlLoader,
}
