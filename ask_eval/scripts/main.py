import os
import asyncio
import argparse
from ask_eval.utils.config import load_config, load_merged_config, write_final_result_file, write_final_evalscope_result_file


async def run_tasks(base_config_path: str):
    """Run multiple evaluation tasks defined in a base config."""
    # Load base config
    base_config = load_config(base_config_path)
    tasks = base_config.get("tasks", "enabled").split(",")
    tasks_config_dir = base_config.get("tasks", "tasks_config_path")

    if 'EvalScope' in tasks_config_dir:
        for task in tasks:
            from scripts.run_evalscope_origin import run_evaluation_evalscope
            await run_evaluation_evalscope(base_config,task)

            save_dir = base_config.get("path", "save_dir")
            write_final_evalscope_result_file(save_dir, task, task, base_config)
    else: 
        for task in tasks:
            task_config_path = os.path.join(tasks_config_dir, f"{task}.ini")
            print(f"Task config path: {task_config_path}")
            if not os.path.exists(task_config_path):
                print(f"Warning: task config does not exist: {task_config_path}")
                continue
                
            # Load merged config
            config = load_merged_config(base_config_path, task_config_path)
            # Run task
            if "ask" in task or "quest_bench" in task or "fata" in task or "in3_interaction" in task:
                print("Starting AskBench-style evaluation...")
                from scripts.run_ask import run_ask_evaluation
                save_dir = await run_ask_evaluation(config)
            else:
                from scripts.run import run_evaluation
                await run_evaluation(config)
            
            # Aggregate per-task results into a final file
            save_dir = config.get("path", "save_dir")
            task_name = config.get("model", "task_name")
            write_final_result_file(save_dir, task, task_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the base config file (.ini)")
    args = parser.parse_args()
    
    asyncio.run(run_tasks(args.config))
