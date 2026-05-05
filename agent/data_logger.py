"""
Data Logger - 记录可微调数据
"""

import json
import uuid
from datetime import datetime
import os


class DataLogger:
    """记录Agent决策数据，用于后续微调"""

    def __init__(self, output_path="agent_training_data.jsonl"):
        self.output_path = output_path
        self.session_id = str(uuid.uuid4())

    def log(self, iteration, input_state, bev_quality, agent_output, result):
        """
        记录一条数据

        Args:
            iteration: 当前迭代次数
            input_state: 输入状态 dict
            bev_quality: BEV质量评估 dict
            agent_output: Agent输出 dict (包含thought和action)
            result: 结果 dict (包含new_bev_quality和improved)
        """
        record = {
            "session_id": self.session_id,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "input_state": input_state,
            "bev_quality": bev_quality,
            "agent_output": agent_output,
            "result": result
        }

        # 追加写入JSONL文件
        with open(self.output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load(self):
        """加载所有记录"""
        records = []
        if os.path.exists(self.output_path):
            with open(self.output_path, "r") as f:
                for line in f:
                    records.append(json.loads(line))
        return records

    def analyze(self):
        """分析记录，统计最有效的action"""
        records = self.load()
        if not records:
            return {}

        # 统计action频率和效果
        action_stats = {}
        for record in records:
            action_name = record["agent_output"].get("action", {}).get("name", "unknown")
            improved = record["result"].get("improved", False)

            if action_name not in action_stats:
                action_stats[action_name] = {"count": 0, "improved_count": 0}

            action_stats[action_name]["count"] += 1
            if improved:
                action_stats[action_name]["improved_count"] += 1

        # 计算成功率
        for action_name, stats in action_stats.items():
            stats["success_rate"] = stats["improved_count"] / stats["count"] if stats["count"] > 0 else 0

        return action_stats
