"""
Qwen2.5-VL with Scene Graph Chain-of-Thought for lmms-eval

This model implements two inference modes:
1. Two-step mode: Generate scene graph first, then answer with scene graph context
2. Unified mode: Generate scene graph and answer in a single pass (faster)

Mimics the logic from VLMEvalKit's SceneGraphChainOfThought implementation.
"""

import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)

from .qwen2_5_vl import Qwen2_5_VL

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning(
        "Failed to import qwen_vl_utils; "
        "Please install it via `pip install qwen-vl-utils`"
    )
    process_vision_info = None


@register_model("qwen2_5_vl_scene_graph")
class Qwen2_5_VL_SceneGraph(Qwen2_5_VL):
    """
    Qwen2.5-VL with Scene Graph Chain-of-Thought

    Extends Qwen2.5-VL to provide scene graph-based reasoning:
    - Two-step mode: Explicit scene graph generation followed by answer generation
    - Unified mode: Single-pass generation with scene graph instruction (default)
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        # Scene graph specific parameters
        use_unified_inference: bool = True,
        use_detailed_scene_graph: bool = True,
        scene_graph_prompt: Optional[str] = None,
        verbose_scene_graph: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            use_cache=use_cache,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_num_frames=max_num_frames,
            use_custom_video_loader=use_custom_video_loader,
            fps=fps,
            max_image_size=max_image_size,
            system_prompt=system_prompt,
            interleave_visuals=interleave_visuals,
            reasoning_prompt=reasoning_prompt,
            **kwargs,
        )

        self.use_unified_inference = use_unified_inference
        self.use_detailed_scene_graph = use_detailed_scene_graph
        self.verbose_scene_graph = verbose_scene_graph
        self.scene_graph_prompt = (
            scene_graph_prompt or self._get_default_scene_graph_prompt()
        )

        if self.rank == 0:
            mode = "unified (single-pass)" if use_unified_inference else "two-step"
            eval_logger.info(
                f"Initialized Scene Graph Qwen2.5-VL with {mode} inference mode"
            )
            if verbose_scene_graph:
                eval_logger.info(
                    f"Scene graph prompt: {self.scene_graph_prompt[:100]}..."
                )

    def _get_default_scene_graph_prompt(self) -> str:
        """Get the default scene graph generation prompt."""
        if self.use_detailed_scene_graph:
            return (
                "Describe objects and their relationships in this image. "
                "Format: Objects: [list]. Relationships: [how they relate]. "
                "No coordinates."
            )
        return "List objects and relationships. No coordinates."

    def _clean_scene_graph_output(self, scene_graph: str) -> str:
        """
        Clean scene graph output to remove detection-style formatting.

        Removes bounding boxes, coordinates, and JSON artifacts to focus on
        semantic content.
        """
        if not scene_graph or not scene_graph.strip():
            return "No scene graph generated."

        # Remove bbox patterns
        scene_graph = re.sub(r'"bbox_2d"\s*:\s*\[[^\]]+\]\s*,?', "", scene_graph)
        scene_graph = re.sub(
            r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]", "", scene_graph
        )
        scene_graph = re.sub(r"\b\d+,\s*\d+,\s*\d+,\s*\d+\b", "", scene_graph)

        # Extract labels from detection format if present
        if scene_graph.strip().startswith("[") and "bbox" in scene_graph:
            labels = re.findall(r'"label"\s*:\s*"([^"]+)"', scene_graph)
            if labels:
                return f"Objects: {', '.join(labels)}"

        # Clean up whitespace and formatting
        scene_graph = re.sub(r"\s+", " ", scene_graph)
        scene_graph = re.sub(r",\s*}", "}", scene_graph)
        scene_graph = re.sub(r",\s*]", "]", scene_graph)

        return scene_graph.strip()

    def _get_unified_prompt(self, original_text: str) -> str:
        """Create unified prompt combining scene graph generation and QA."""
        scene_instruction = (
            "First, identify objects and relationships (no coordinates)."
        )
        return f"{scene_instruction} Then answer: {original_text.strip()}"

    def _parse_unified_response(self, response: str) -> str:
        """
        Parse unified response to extract the final answer.

        Looks for answer indicators and removes scene graph content.
        """
        # Primary indicators for answer section
        answer_indicators = ["Then answer:", "Answer:", "Then:", "\nAnswer"]

        for indicator in answer_indicators:
            if indicator in response:
                parts = response.split(indicator, 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
                    # Clean up if scene graph content leaked
                    for start_pattern in [
                        "Therefore",
                        "So",
                        "The answer",
                        "Based on",
                    ]:
                        if start_pattern in answer[:100]:
                            idx = answer.find(start_pattern)
                            return answer[idx:].strip()
                    return answer

        # Fallback: look for common answer start patterns
        fallback_indicators = ["Therefore", "So,", "Based on", "The answer is"]
        for indicator in fallback_indicators:
            if indicator in response:
                parts = response.split(indicator, 1)
                if len(parts) > 1:
                    return (indicator + " " + parts[1]).strip()

        return response.strip()

    def _process_visuals(
        self, visual_list: List[List]
    ) -> List[List[dict]]:
        """
        Process visual inputs (images/videos) into format expected by processor.

        Args:
            visual_list: List of visual inputs per instance

        Returns:
            List of processed visual dictionaries
        """
        processed_visuals_list = []

        for visuals in visual_list:
            processed_visuals = []

            if visuals is not None:
                for visual in visuals:
                    if isinstance(visual, str) and visual.endswith(
                        (".mp4", ".avi", ".mov")
                    ):
                        # Video file
                        processed_visuals.append(
                            {
                                "type": "video",
                                "video": visual,
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )
                    elif isinstance(visual, Image.Image):
                        # Image file - convert to base64
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        processed_visuals.append(
                            {
                                "type": "image",
                                "image": f"data:image/jpeg;base64,{base64_string}",
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )

            processed_visuals_list.append(processed_visuals)

        return processed_visuals_list

    def _build_messages(
        self, contexts: List[str], processed_visuals_list: List[List[dict]]
    ) -> List[List[dict]]:
        """
        Build batched messages from contexts and processed visuals.

        Args:
            contexts: List of text contexts
            processed_visuals_list: List of processed visual inputs

        Returns:
            List of message dictionaries in chat format
        """
        batched_messages = []
        for context, processed_visuals in zip(contexts, processed_visuals_list):
            message = [{"role": "system", "content": self.system_prompt}]
            message.append(
                {
                    "role": "user",
                    "content": processed_visuals + [{"type": "text", "text": context}],
                }
            )
            batched_messages.append(message)
        return batched_messages

    def _process_video_frames(self, video_inputs):
        """Sample frames from video inputs using max_num_frames."""
        if video_inputs is None:
            return None

        total_frames = video_inputs[0].shape[0]
        indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
        indices = np.unique(indices)

        # Ensure last frame is included
        if total_frames - 1 not in indices:
            indices = np.append(indices, total_frames - 1)
            indices = np.unique(indices)

        video_inputs[0] = video_inputs[0][indices]
        return video_inputs

    def _batch_generate(
        self, batched_messages: List[List[dict]], gen_kwargs: dict
    ) -> List[str]:
        """
        Generate responses for a batch of messages.

        Args:
            batched_messages: List of message dictionaries
            gen_kwargs: Generation keyword arguments

        Returns:
            List of generated text responses
        """
        # Apply chat template and process vision info
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in batched_messages
        ]
        image_inputs, video_inputs = process_vision_info(batched_messages)
        video_inputs = self._process_video_frames(video_inputs)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if self.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(self.device)

        # Setup generation parameters
        default_gen_kwargs = {
            "max_new_tokens": 32768,
            "temperature": 0.0,
            "top_p": None,
            "num_beams": 1,
        }
        current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}

        if current_gen_kwargs["temperature"] > 0:
            current_gen_kwargs["do_sample"] = True
        else:
            current_gen_kwargs["do_sample"] = False
            current_gen_kwargs["temperature"] = None
            current_gen_kwargs["top_p"] = None

        # Generate
        cont = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=current_gen_kwargs["do_sample"],
            temperature=current_gen_kwargs["temperature"],
            top_p=current_gen_kwargs["top_p"],
            num_beams=current_gen_kwargs["num_beams"],
            max_new_tokens=current_gen_kwargs["max_new_tokens"],
            use_cache=self.use_cache,
        )

        # Decode generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)
        ]
        answers = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return answers

    def _generate_scene_graphs(
        self, processed_visuals_list: List[List[dict]]
    ) -> List[str]:
        """
        Generate scene graphs for a batch of inputs.

        Args:
            processed_visuals_list: List of processed visual inputs

        Returns:
            List of cleaned scene graph strings
        """
        # Build messages with scene graph prompt
        sg_contexts = [self.scene_graph_prompt] * len(processed_visuals_list)
        sg_messages = self._build_messages(sg_contexts, processed_visuals_list)

        # Generate with limited tokens for scene graphs
        sg_gen_kwargs = {"max_new_tokens": 256, "temperature": 0.0}
        scene_graphs = self._batch_generate(sg_messages, sg_gen_kwargs)

        # Clean scene graphs
        cleaned_scene_graphs = [
            self._clean_scene_graph_output(sg) for sg in scene_graphs
        ]

        if self.verbose_scene_graph and self.rank == 0:
            for i, sg in enumerate(cleaned_scene_graphs[:2]):
                eval_logger.info(f"Scene Graph {i+1}: {sg[:200]}...")

        return cleaned_scene_graphs

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Override generate_until to implement scene graph chain-of-thought.

        Supports two modes:
        1. Unified mode (default): Single-pass generation with scene graph instruction
        2. Two-step mode: First generate scene graph, then answer with context
        """
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        mode_desc = "Unified" if self.use_unified_inference else "Two-Step"
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc=f"Scene Graph ({mode_desc})",
        )
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            gen_kwargs = all_gen_kwargs[0]

            # Setup until conditions
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            until = [item for item in until if item != "\n\n"]

            # Clean contexts
            contexts = list(contexts) if isinstance(contexts, tuple) else contexts
            contexts = [ctx.replace("<image>", "") for ctx in contexts]

            # Process visuals once for efficiency
            processed_visuals_list = self._process_visuals(visual_list)

            if self.use_unified_inference:
                # Unified mode: modify contexts to include scene graph instruction
                modified_contexts = [self._get_unified_prompt(ctx) for ctx in contexts]
                batched_messages = self._build_messages(
                    modified_contexts, processed_visuals_list
                )
                answers = self._batch_generate(batched_messages, gen_kwargs)

                # Parse unified responses to extract final answers
                answers = [self._parse_unified_response(ans) for ans in answers]
            else:
                # Two-step mode: generate scene graphs first
                scene_graphs = self._generate_scene_graphs(processed_visuals_list)

                # Combine scene graphs with original questions
                modified_contexts = [
                    f"Scene: {sg}\n\nQ: {ctx.strip()}"
                    for ctx, sg in zip(contexts, scene_graphs)
                ]
                batched_messages = self._build_messages(
                    modified_contexts, processed_visuals_list
                )
                answers = self._batch_generate(batched_messages, gen_kwargs)

            # Apply until conditions
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            # Process and cache results
            for ans, context in zip(answers, contexts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), clean_ans
                )
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
