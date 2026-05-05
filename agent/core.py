"""
ReAct Agent Core - Agent主控逻辑
"""

import json
import uuid
import requests
import cv2
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, FEW_SHOT_EXAMPLES
from .functions import AVAILABLE_TOOLS
from .bev_evaluator import BEVEvaluator
from .refiner import ImageRefiner
from .vision_llm import VisionLLM


class AgentCore:
    """Agent核心，ReAct循环引擎"""

    def __init__(self, llm_url="http://localhost:11434", max_iterations=3, fast_mode=False):
        """
        Args:
            llm_url: Ollama服务地址
            max_iterations: 最大迭代次数
            fast_mode: 跳过VisionLLM，用纯规则决策（快速模式）
        """
        self.llm_url = llm_url
        self.max_iterations = max_iterations
        self.fast_mode = fast_mode
        self.evaluator = BEVEvaluator()
        self.refiner = ImageRefiner()
        self.vision_llm = VisionLLM(llm_url=llm_url) if not fast_mode else None
        self.session_id = str(uuid.uuid4())

    def run(self, model, images, intrinsics, extrinsics, lidar_points, lidar_mask, bev_cfg=None):
        """
        运行Agent循环

        Args:
            model: BEVFusion模型
            images: (B, N_cams, 3, H, W)
            intrinsics: (B, N_cams, 3, 3)
            extrinsics: (B, N_cams, 4, 4)
            lidar_points: (B, N_pts, 5)
            lidar_mask: (B, N_pts)
            bev_cfg: BEV配置字典

        Returns:
            dict: 最终结果和决策历史
        """
        history = []
        bev_cfg = bev_cfg or {}

        # 首次生成BEV
        logits, bev_seg = model(images, intrinsics, extrinsics, lidar_points, lidar_mask)
        cam_bev = bev_seg[0] if bev_seg.dim() > 2 else bev_seg

        # 评估
        eval_result = self.evaluator.evaluate(cam_bev)
        history.append({"iteration": 0, "eval": eval_result, "action": None})

        # Agent循环
        for i in range(self.max_iterations):
            iteration = i + 1

            # 检查是否需要优化
            if not eval_result["needs_optimization"]:
                return {
                    "final_bev": bev_seg,
                    "history": history,
                    "finalized": True
                }

            # 生成问题区域到相机的映射
            problem_camera_mapping = self._get_problem_camera_mapping(
                eval_result["problem_coords"], extrinsics, intrinsics, bev_cfg
            )

            # 获取需要分析的相机ID
            camera_ids_to_analyze = self._get_unique_camera_ids(problem_camera_mapping)

            # 使用视觉LLM分析这些相机的图像（fast_mode跳过）
            if self.fast_mode:
                vision_analysis = []
            else:
                vision_analysis = self._analyze_images_with_vision_llm(
                    images, camera_ids_to_analyze
                )

            # 生成描述
            problem_areas = self._format_problem_areas(
                eval_result["problem_coords"],
                problem_camera_mapping,
                vision_analysis
            )

            # 结合BEV评估和视觉LLM分析做决策
            decision = self._make_decision(
                eval_result,
                vision_analysis,
                problem_areas,
                history=history
            )

            if decision is None:
                decision = {
                    "thought": "无法决定，使用finalize",
                    "action": {"name": "finalize", "parameters": {}}
                }

            history.append({"iteration": iteration, "decision": decision, "vision_analysis": vision_analysis})

            # 检查是否是finalize
            if decision["action"]["name"] == "finalize":
                return {
                    "final_bev": bev_seg,
                    "history": history,
                    "finalized": True
                }

            # 执行action
            images = self._execute_action(decision["action"], images)

            # 重新生成BEV
            logits, bev_seg = model(images, intrinsics, extrinsics, lidar_points, lidar_mask)
            cam_bev = bev_seg[0] if bev_seg.dim() > 2 else bev_seg

            # 评估
            new_eval = self.evaluator.evaluate(cam_bev)
            history.append({"iteration": iteration, "eval": new_eval})

            eval_result = new_eval

        # 达到最大迭代次数
        return {
            "final_bev": bev_seg,
            "history": history,
            "finalized": False,
            "reason": "达到最大迭代次数"
        }

    def _get_problem_camera_mapping(self, problem_coords, extrinsics, intrinsics, bev_cfg):
        """获取问题区域对应的相机"""
        if not problem_coords:
            return []

        try:
            mapping = self.evaluator.bev_to_camera_mapping(
                problem_coords, extrinsics, intrinsics, bev_cfg
            )
            return mapping
        except Exception as e:
            import traceback
            print(f"映射失败: {e}")
            traceback.print_exc()
            return []

    def _get_unique_camera_ids(self, problem_camera_mapping):
        """从映射中获取需要分析的相机ID"""
        camera_ids = set()
        for m in problem_camera_mapping:
            camera_ids.update(m.get("camera_ids", []))
        return list(camera_ids) if camera_ids else [0, 1, 2, 3, 4, 5]

    def _analyze_images_with_vision_llm(self, images, camera_ids):
        """使用视觉LLM分析图像"""
        try:
            analyses = self.vision_llm.analyze_images(images, camera_ids)
            return analyses
        except Exception as e:
            print(f"视觉LLM分析失败: {e}")
            return []

    def _make_decision(self, eval_result, vision_analysis, problem_areas, history=None):
        """根据BEV评估和视觉LLM分析做决策"""

        # fast_mode: 纯规则决策，不依赖VisionLLM
        # 只做一次增强，之后直接finalize
        if self.fast_mode:
            integrity = eval_result.get("integrity", 1.0)
            already_enhanced = any(
                h.get("decision", {}).get("action", {}).get("name") == "enhance_image"
                for h in (history or []) if isinstance(h, dict)
            )
            if integrity < 0.95 and not already_enhanced:
                return {
                    "thought": f"[FastMode] integrity={integrity:.3f}，做一次对比度增强",
                    "action": {
                        "name": "enhance_image",
                        "parameters": {"camera_ids": [0, 1, 2, 3, 4, 5], "enhancement_type": "contrast", "factor": 1.3}
                    }
                }
            else:
                return {
                    "thought": f"[FastMode] integrity={integrity:.3f}，完成",
                    "action": {"name": "finalize", "parameters": {}}
                }

        # 如果有视觉LLM的分析结果，优先使用
        if vision_analysis:
            # 直接从analysis中提取conditions来决定工具
            for analysis in vision_analysis:
                cam_id = analysis.get("camera_id", 0)
                conditions = analysis.get("conditions", [])

                # 根据conditions决定工具（更准确的匹配）
                if "rain" in conditions:
                    return {
                        "thought": f"检测到{analysis.get('camera_name', cam_id)}相机图像有雨，建议去雨处理",
                        "action": {
                            "name": "remove_rain",
                            "parameters": {"camera_ids": [cam_id], "regions": None}
                        }
                    }
                elif "fog" in conditions or "haze" in conditions:
                    return {
                        "thought": f"检测到{analysis.get('camera_name', cam_id)}相机图像有雾/霾，建议去雾处理",
                        "action": {
                            "name": "dehaze",
                            "parameters": {"camera_ids": [cam_id], "regions": None}
                        }
                    }
                elif "glare" in conditions or "low_light" in conditions:
                    return {
                        "thought": f"检测到{analysis.get('camera_name', cam_id)}相机图像光照不佳(眩光/弱光)，建议增强",
                        "action": {
                            "name": "enhance_image",
                            "parameters": {"camera_ids": [cam_id], "enhancement_type": "contrast", "factor": 1.5}
                        }
                    }

            # Fallback: 如果没有匹配到conditions，使用merge_analyses的suggested_tools
            tool_plan = self.vision_llm.merge_analyses(vision_analysis)

            # 按优先级选择工具
            if tool_plan["remove_rain"]["camera_ids"]:
                cam_ids = tool_plan["remove_rain"]["camera_ids"]
                regions = tool_plan["remove_rain"]["regions"]
                return {
                    "thought": f"检测到{cam_ids}相机图像有雨，建议去雨处理",
                    "action": {
                        "name": "remove_rain",
                        "parameters": {
                            "camera_ids": cam_ids,
                            "regions": regions if regions else None
                        }
                    }
                }

            if tool_plan["dehaze"]["camera_ids"]:
                cam_ids = tool_plan["dehaze"]["camera_ids"]
                regions = tool_plan["dehaze"]["regions"]
                return {
                    "thought": f"检测到{cam_ids}相机图像有雾/霾，建议去雾处理",
                    "action": {
                        "name": "dehaze",
                        "parameters": {
                            "camera_ids": cam_ids,
                            "regions": regions if regions else None
                        }
                    }
                }

            if tool_plan["enhance_image"]["camera_ids"]:
                cam_ids = tool_plan["enhance_image"]["camera_ids"]
                return {
                    "thought": f"检测到{cam_ids}相机图像需要增强",
                    "action": {
                        "name": "enhance_image",
                        "parameters": {
                            "camera_ids": cam_ids,
                            "enhancement_type": "contrast",
                            "factor": 1.5
                        }
                    }
                }

        # 回退到默认决策
        if eval_result["edge_density"] < 0.2:
            return {
                "thought": "BEV边缘非常模糊",
                "action": {
                    "name": "enhance_image",
                    "parameters": {
                        "camera_ids": [0, 1, 2],
                        "enhancement_type": "contrast",
                        "factor": 1.8
                    }
                }
            }
        else:
            return {
                "thought": "BEV质量一般，建议finalize",
                "action": {
                    "name": "finalize",
                    "parameters": {}
                }
            }

    def _format_problem_areas(self, problem_coords, problem_camera_mapping, vision_analysis=None):
        """格式化问题区域描述"""
        if not problem_coords:
            return None

        mapping_dict = {}
        for m in problem_camera_mapping:
            bev_center = tuple(m.get("bev_center", [0, 0]))
            mapping_dict[bev_center] = m.get("camera_ids", [])

        areas = []
        for idx, region in enumerate(problem_coords[:3]):
            bbox = region["bbox"]
            center = region["center"]
            camera_ids = mapping_dict.get(tuple(center), [])

            # 如果有视觉LLM分析，添加更多信息
            vision_info = ""
            if vision_analysis:
                for analysis in vision_analysis:
                    if analysis.get("camera_id") in camera_ids:
                        conditions = analysis.get("conditions", [])
                        if conditions:
                            vision_info = f" [视觉检测: {','.join(conditions)}]"

            if camera_ids:
                camera_names = self._get_camera_names(camera_ids)
                areas.append(
                    f"BEV区域({center[0]},{center[1]})"
                    f"，对应{camera_names}(ID:{camera_ids})"
                    f"，bbox:[{bbox[0]},{bbox[1]}-{bbox[2]},{bbox[3]}]"
                    f"{vision_info}"
                )

        return ", ".join(areas) if areas else None

    def _get_camera_names(self, camera_ids):
        """相机ID转名称"""
        camera_names = {
            0: "CAM_FRONT",
            1: "CAM_FRONT_RIGHT",
            2: "CAM_FRONT_LEFT",
            3: "CAM_BACK",
            4: "CAM_BACK_RIGHT",
            5: "CAM_BACK_LEFT"
        }
        return [camera_names.get(i, f"Camera{i}") for i in camera_ids]

    def _execute_action(self, action, images):
        """执行action"""
        name = action["name"]
        params = action.get("parameters", {})
        regions = params.get("regions")

        if name == "enhance_image":
            camera_ids = params.get("camera_ids", [0, 1, 2, 3, 4, 5])
            enhancement_type = params.get("enhancement_type", "contrast")
            factor = params.get("factor", 1.5)
            return self.refiner.enhance_image(images, camera_ids, enhancement_type, factor)

        elif name == "remove_rain":
            camera_ids = params.get("camera_ids", [0, 1, 2, 3, 4, 5])
            method = params.get("method", "CLAHE")
            return self.refiner.remove_rain(images, camera_ids, method, regions)

        elif name == "dehaze":
            camera_ids = params.get("camera_ids", [0, 1, 2, 3, 4, 5])
            method = params.get("method", "CLAHE")
            return self.refiner.dehaze(images, camera_ids, method, regions)

        elif name == "crop_and_zoom":
            camera_ids = params.get("camera_ids", [0, 1, 2, 3, 4, 5])
            bbox = params.get("bbox", [0.3, 0.3, 0.7, 0.7])
            zoom_factor = params.get("zoom_factor", 2.0)
            return self.refiner.crop_and_zoom(images, camera_ids, bbox, zoom_factor)

        return images
