# --- Standard Library Imports ---
import dataclasses
from dataclasses import asdict
import os
import logging
import time
import random
import json
import re
from typing import Any

# --- Project Imports ---
from browsergym.experiments.agent import Agent, AgentInfo
from agentlab.agents.agent_args import AgentArgs as AgentLabAgentArgs
from agentlab.llm.chat_api import BaseModelArgs, ChatModel, AnthropicChatModel
from agentlab.llm.base_api import AbstractChatModel
import agentlab.agents.dynamic_prompting as dp
from agentlab.llm.llm_utils import (
    AIMessage,
    Discussion,
    ParseError,
    SystemMessage,
)
from .utils import CustomActionSetArgs, retry
from .vLLM_prompt import VllmMainPrompt, PromptFlags

from anthropic import AnthropicBedrock
from openai import AzureOpenAI, OpenAI

try:
    from litellm import completion as litellm_completion
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    litellm_completion = None

# --- Logging Setup ---
logger = logging.getLogger(__name__)

EVIDENCE_FIRST_SYSTEM_PROMPT_APPENDIX = """
Additional operating rules:
- Treat all user-provided task details as requirements, but treat unknown facts as unknown until observed in the UI.
- Never invent, guess, or approximate factual details (times, names, dates, addresses, IDs, prices, quantities, etc.).
- If a required detail is missing on the current page, navigate to the most relevant app/page to verify it before acting.
- For communication tasks (message/email/note), first collect required facts from source apps, then compose/send.
- Prefer exact copied values from observed UI state when sharing factual details.
- If facts cannot be verified from available UI state, do not fabricate them; continue searching instead.
- Do not assume task completion after a plausible action; verify that the requested result is actually present in the UI.
- Loop avoidance:
  - Never click the same non-input UI element repeatedly when the page is not making progress.
  - If the last two actions did not advance toward the user goal, pick a different target or app.
  - Avoid repeating short click cycles (for example A->B->C->A->B->C); switch strategy immediately.
"""


@dataclasses.dataclass
class ModelArgs(BaseModelArgs):
    model_name: str = "demo"
    model_pretty_name: str = "demo"
    port: str = "8000"
    api_key: str = "AMI_RULZ"
    api_version: str = None
    hostname: str = "0.0.0.0/v1"
    host_name_updated_on: str = "2025-01-01:00:00:00"
    temperature: float = 0.5
    vision_support: bool = True
    max_tokens: int = 100
    client_type: str = "vllm"
    aws_access_key: str = None
    aws_secret_key: str = None
    aws_session_token: str = None
    aws_region: str = "us-west-2"

    def make_model(self) -> AbstractChatModel:
        logger.info(f"Creating Model with model_name: {self.model_name}")
        if self.client_type == "litellm":
            return LiteLLMChatModel(
                model_name=self.model_name,
                api_key=self.api_key,
                api_version=self.api_version,
                hostname=self.hostname,
                port=self.port,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_retry=3,
                min_retry_wait_time=2,
            )

        if self.client_type == "vllm" or self.client_type == "gemini":
            suffix = "v1" if self.client_type == "vllm" else ""
            base_url = f"http://{self.hostname}:{self.port}/{suffix}"
            client_args = {"base_url": base_url}
            client_class = OpenAI
            return VLLMChatModel(
                model_name=self.model_name,
                hostname=self.hostname,
                port=self.port,
                api_key=self.api_key,
                api_version=self.api_version,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                host_name_updated_on=self.host_name_updated_on,
                n_retry_server=3,
                min_retry_wait_time=60,
                client_class=client_class,
                client_args=client_args,
            )

        elif self.client_type == "azure":
            client_args = {
                "azure_endpoint": f"https://{self.hostname}",
                "api_version": self.api_version,
            }
            client_class = AzureOpenAI
            return VLLMChatModel(
                model_name=self.model_name,
                hostname=self.hostname,
                port=self.port,
                api_key=self.api_key,
                api_version=self.api_version,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                host_name_updated_on=self.host_name_updated_on,
                n_retry_server=3,
                min_retry_wait_time=60,
                client_class=client_class,
                client_args=client_args,
            )
        elif self.client_type == "aws":
            return BedrockChatModel(
                model_name=self.api_version,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_retry=3,
                aws_access_key=self.aws_access_key,
                aws_secret_key=self.aws_secret_key,
                aws_session_token=self.aws_session_token,
                aws_region=self.aws_region,
            )
        elif self.client_type == "openai":
            client_args = {"base_url": "https://api.openai.com/v1"}
            client_class = OpenAI
            return VLLMChatModel(
                model_name=self.model_name,
                hostname=self.hostname,
                port=self.port,
                api_key=self.api_key,
                api_version=self.api_version,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                host_name_updated_on=self.host_name_updated_on,
                n_retry_server=3,
                min_retry_wait_time=60,
                client_class=client_class,
                client_args=client_args,
            )
        else:

            raise ValueError(f"Unknown client_type: {self.client_type}.")


class VLLMChatModel(ChatModel):
    def __init__(
        self,
        model_name,
        hostname,
        port,
        api_key="AMI_RULZ",
        api_version=None,
        temperature=0.5,
        max_tokens=100,
        n_retry_server=4,
        min_retry_wait_time=60,
        host_name_updated_on="2025-01-01:00:00:00",
        client_class=None,
        client_args=None,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=n_retry_server,
            min_retry_wait_time=min_retry_wait_time,
            client_class=client_class,
            client_args=client_args,
            pricing_func=None,
        )
        self.host_name_updated_on = host_name_updated_on


class LiteLLMChatModel(AbstractChatModel):
    """Provider-agnostic chat model backed by LiteLLM completion()."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        api_version: str | None = None,
        hostname: str | None = None,
        port: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 100,
        max_retry: int = 3,
        min_retry_wait_time: float = 2.0,
    ):
        if litellm_completion is None:
            raise ImportError(
                "litellm is required for client_type='litellm'. "
                "Install via `uv add litellm`."
            )
        self.model_name = model_name
        self.api_key = api_key
        self.api_version = api_version
        self.hostname = hostname
        self.port = str(port) if port is not None else None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry
        self.min_retry_wait_time = min_retry_wait_time
        self.retries = 0
        self.success = False

    def _resolve_api_base(self) -> str | None:
        if self.hostname is None:
            return None
        host = str(self.hostname).strip()
        if not host:
            return None
        if host.startswith("http://") or host.startswith("https://"):
            return host.rstrip("/")
        port = self.port or "8000"
        return f"http://{host}:{port}/v1"

    def _resolve_extra_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        api_base = self._resolve_api_base()
        if api_base:
            kwargs["api_base"] = api_base
        if self.api_version:
            kwargs["api_version"] = self.api_version
        if self.api_key:
            kwargs["api_key"] = self.api_key
        extra_raw = os.getenv("AGENTCI_LITELLM_KWARGS_JSON", "").strip()
        if extra_raw:
            try:
                parsed = json.loads(extra_raw)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                kwargs.update(parsed)
        return kwargs

    @staticmethod
    def _extract_text_content(message_content: Any) -> str:
        if isinstance(message_content, str):
            return message_content
        if isinstance(message_content, list):
            parts: list[str] = []
            for item in message_content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return str(message_content)

    def __call__(
        self,
        messages: list[dict],
        n_samples: int = 1,
        temperature: float = None,
    ) -> Any:
        self.retries = 0
        self.success = False
        completion = None
        last_exc: Exception | None = None
        sampled_temp = self.temperature if temperature is None else float(temperature)
        request_kwargs = self._resolve_extra_kwargs()

        for attempt in range(self.max_retry):
            self.retries += 1
            try:
                completion = litellm_completion(
                    model=self.model_name,
                    messages=list(messages),
                    n=n_samples,
                    temperature=sampled_temp,
                    max_tokens=self.max_tokens,
                    drop_params=True,
                    **request_kwargs,
                )
                self.success = True
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == self.max_retry - 1:
                    break
                sleep_for = self.min_retry_wait_time * (attempt + 1)
                sleep_for += random.uniform(0.0, 0.35)
                time.sleep(sleep_for)

        if completion is None:
            raise RuntimeError(
                f"LiteLLM completion failed after {self.max_retry} retries. "
                f"Last error: {last_exc}"
            )

        if n_samples == 1:
            content = self._extract_text_content(completion.choices[0].message.content)
            return AIMessage(content)
        return [
            AIMessage(self._extract_text_content(choice.message.content))
            for choice in completion.choices
        ]

    def get_stats(self):
        return {
            "n_retry_llm": self.retries,
        }


class BedrockChatModel(AnthropicChatModel):
    def __init__(
        self,
        model_name,
        temperature=0.5,
        max_tokens=100,
        max_retry=3,
        aws_access_key=None,
        aws_secret_key=None,
        aws_session_token=None,
        aws_region="us-west-2",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry

        self.client = AnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_session_token=aws_session_token,
            aws_region=aws_region,
        )


@dataclasses.dataclass
class AgentArgs(AgentLabAgentArgs):
    """
    This class takes the yaml config and categorize the arguments into different subsets
    It also instantiate the agent.
    """

    model_name: str = "demo"
    model_pretty_name: str = "demo"
    custom_actions: list[str] = dataclasses.field(default_factory=list)
    use_html: bool = False
    use_axtree: bool = False
    use_screenshot: bool = False
    use_som: bool = False
    extract_visible_tag: bool = False
    extract_clickable_tag: bool = False
    extract_coords: bool = False
    filter_visible_elements_only: bool = False  # filter elements that are not visible
    use_focused_element: bool = False  # use focused element in the observation
    # --- Agent Flags ---
    use_memory: bool = False
    use_thinking: bool = False
    use_concrete_example: bool = False
    use_abstract_example: bool = False
    # --- ARGS for history ---
    use_history: bool = False  # enable history
    use_action_history: bool = False  # enable action history
    use_think_history: bool = False  # enable think history
    # --- Prompt Flags ---
    prompt_txt: dict = dataclasses.field(
        default_factory=dict
    )  # prompt text for the agent
    # --- ChatModel Flags ---
    hostname: str = "0.0.0.0/v1"
    port: str = "8000"
    api_key: str = "AMI_RULZ"
    api_version: str = None
    host_name_updated_on: str = "2025-01-01:00:00:00"
    temperature: float = 0.5
    max_tokens: int = 100
    client_type: str = "vllm"
    # --- AWS Bedrock Flags ---
    aws_access_key: str = None
    aws_secret_key: str = None
    aws_session_token: str = None
    aws_region: str = "us-west-2"

    def make_flags(self) -> PromptFlags:
        return PromptFlags(
            # figure out what to include in generic prompt flags
            obs=dp.ObsFlags(
                use_html=self.use_html,
                use_ax_tree=self.use_axtree,
                use_focused_element=self.use_focused_element,
                # --- ARGS for screenshot ---
                use_screenshot=self.use_screenshot,
                use_som=self.use_som,
                extract_visible_tag=self.extract_visible_tag,
                extract_clickable_tag=self.extract_clickable_tag,
                extract_coords=self.extract_coords,
                filter_visible_elements_only=self.filter_visible_elements_only,
                # --- ARGS for history tory---
                use_history=self.use_history,
                use_action_history=self.use_action_history,
                use_think_history=self.use_think_history,
            ),
            action=dp.ActionFlags(
                action_set=CustomActionSetArgs(
                    subsets=["custom"],  # define a subset of the action space
                    custom_actions=self.custom_actions,  # list of custom actions
                    strict=False,  # less strict on the parsing of the actions
                    multiaction=False,  # does not enable the agent to take multiple actions at once
                ),
                multi_actions=False,
            ),
            # --- ARGS for agent ---
            use_thinking=self.use_thinking,  # enable thoughts
            use_concrete_example=self.use_concrete_example,  # keep
            use_abstract_example=self.use_abstract_example,  # keep
        )

    def make_chat_model_flags(self) -> ModelArgs:

        return ModelArgs(
            model_name=self.model_name,
            model_pretty_name=self.model_pretty_name,
            port=self.port,
            api_key=self.api_key,
            api_version=self.api_version,
            hostname=self.hostname,
            host_name_updated_on=self.host_name_updated_on,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            client_type=self.client_type,
            aws_access_key=self.aws_access_key,
            aws_secret_key=self.aws_secret_key,
            aws_session_token=self.aws_session_token,
            aws_region=self.aws_region,
        )

    def make_agent(self) -> Agent:
        print("Creating DemoAgent with model_name: ", self.model_name)
        return VLLMAgent(
            chat_model_args=self.make_chat_model_flags(),
            flags=self.make_flags(),
            prompt_txt=self.prompt_txt,
        )


class VLLMAgent(Agent):
    LOOP_BLOCK_TURNS = 3
    LOOP_MIN_PATTERN = 3
    LOOP_MAX_PATTERN = 5

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: PromptFlags,
        prompt_txt: dict,
        max_retry: int = 3,
    ):
        logging.info("Initializing vllmAgent with flags: %s", asdict(flags))
        self.chat_llm = chat_model_args.make_model()
        self.chat_model_args = chat_model_args
        self.max_retry = max_retry
        self.flags = flags
        self.action_set = flags.action.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)
        self.prompt_txt = prompt_txt
        self.reset(seed=None)
        self.obs_history = []
        self._blocked_click_bids: dict[str, int] = {}

    @staticmethod
    def _extract_click_bid(action: str | None) -> str | None:
        if not action:
            return None
        match = re.search(r"""^click\(\s*['"]([^'"]+)['"]""", action.strip())
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _detect_repeated_click_cycle(click_seq: list[str]) -> list[str] | None:
        if len(click_seq) < VLLMAgent.LOOP_MIN_PATTERN * 2:
            return None
        for size in range(VLLMAgent.LOOP_MIN_PATTERN, VLLMAgent.LOOP_MAX_PATTERN + 1):
            if len(click_seq) < size * 2:
                continue
            prev_chunk = click_seq[-2 * size : -size]
            last_chunk = click_seq[-size:]
            if prev_chunk == last_chunk:
                return list(last_chunk)
        return None

    def _decay_blocked_click_bids(self) -> None:
        if not self._blocked_click_bids:
            return
        expired: list[str] = []
        for bid, turns_left in self._blocked_click_bids.items():
            remaining = turns_left - 1
            if remaining <= 0:
                expired.append(bid)
            else:
                self._blocked_click_bids[bid] = remaining
        for bid in expired:
            self._blocked_click_bids.pop(bid, None)

    def _set_blocked_click_bids(self, bids: list[str], turns: int | None = None) -> None:
        ttl = turns if turns is not None else self.LOOP_BLOCK_TURNS
        for bid in bids:
            if not bid:
                continue
            self._blocked_click_bids[bid] = max(self._blocked_click_bids.get(bid, 0), ttl)

    def _active_blocked_click_bids(self) -> set[str]:
        return {bid for bid, turns_left in self._blocked_click_bids.items() if turns_left > 0}

    def _loop_breaker_replan(
        self,
        *,
        main_prompt: VllmMainPrompt,
        system_prompt: SystemMessage,
        blocked_bids: set[str],
        trigger_reason: str,
    ) -> tuple[dict[str, Any], bool]:
        blocked = ", ".join(sorted(blocked_bids)) if blocked_bids else "(none)"
        loop_guard_message = {
            "role": "user",
            "content": (
                "Loop guard triggered.\n"
                f"Reason: {trigger_reason}\n"
                f"Temporarily blocked click bids: {blocked}\n"
                "Choose one different next action that advances the task. "
                "Do not click any blocked bid. Return only valid <think>/<action>."
            ),
        }
        replan_messages = Discussion([system_prompt, main_prompt.prompt, loop_guard_message])
        try:
            replanned = retry(
                self.chat_llm,
                replan_messages,
                n_retry=1,
                parser=main_prompt._parse_answer,
            )
        except ParseError:
            logging.info("Loop breaker replan failed to parse; falling back to noop.")
            return {"action": "noop(wait_ms=750)", "think": "Loop breaker fallback noop."}, True

        replanned_bid = self._extract_click_bid(replanned.get("action"))
        if replanned_bid and replanned_bid in blocked_bids:
            logging.info(
                "Loop breaker replan still targeted blocked bid %s; forcing noop.",
                replanned_bid,
            )
            replanned["action"] = "noop(wait_ms=750)"
            if not replanned.get("think"):
                replanned["think"] = "Blocked repeated click; inserted noop."
        return replanned, True

    def _apply_loop_breaker(
        self,
        *,
        ans_dict: dict[str, Any],
        main_prompt: VllmMainPrompt,
        system_prompt: SystemMessage,
    ) -> tuple[dict[str, Any], bool]:
        proposed_action = ans_dict.get("action")
        proposed_bid = self._extract_click_bid(proposed_action)
        if not proposed_bid:
            return ans_dict, False

        blocked_bids = self._active_blocked_click_bids()
        if proposed_bid in blocked_bids:
            return self._loop_breaker_replan(
                main_prompt=main_prompt,
                system_prompt=system_prompt,
                blocked_bids=blocked_bids,
                trigger_reason=f"proposed click('{proposed_bid}') is blocked",
            )

        click_history: list[str] = []
        for prior_action in self.displayed_actions:
            bid = self._extract_click_bid(prior_action)
            if bid:
                click_history.append(bid)
        click_seq = click_history + [proposed_bid]

        cycle = self._detect_repeated_click_cycle(click_seq)
        if cycle:
            self._set_blocked_click_bids(cycle)
            blocked_bids = self._active_blocked_click_bids()
            return self._loop_breaker_replan(
                main_prompt=main_prompt,
                system_prompt=system_prompt,
                blocked_bids=blocked_bids,
                trigger_reason=f"repeated click cycle detected: {'->'.join(cycle)}",
            )

        if len(click_seq) >= 3 and click_seq[-1] == click_seq[-2] == click_seq[-3]:
            self._set_blocked_click_bids([proposed_bid])
            blocked_bids = self._active_blocked_click_bids()
            return self._loop_breaker_replan(
                main_prompt=main_prompt,
                system_prompt=system_prompt,
                blocked_bids=blocked_bids,
                trigger_reason=f"same click bid repeated 3x: {proposed_bid}",
            )

        return ans_dict, False

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    def get_action(self, obs: Any):

        self._decay_blocked_click_bids()
        self.obs_history.append(obs)
        main_prompt = VllmMainPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.displayed_actions,  # in most cases same as self.actions, but in UItars we change the API, so displayed actions are the model native action calls, and actions are the browsergym native action calls
            thoughts=self.thoughts,
            flags=self.flags,
            prompt_txt=self.prompt_txt,  # pass the flags to the prompt
            client_type=self.chat_model_args.client_type,
        )

        configured_system_prompt = self.prompt_txt.system_prompt
        if configured_system_prompt is None:
            configured_system_prompt = (
                f"{dp.SystemPrompt().prompt.strip()}\n\n"
                f"{EVIDENCE_FIRST_SYSTEM_PROMPT_APPENDIX.strip()}"
            )
        system_prompt = SystemMessage(
            configured_system_prompt
        )
        logging.info(f"The  prompt is: {str(main_prompt.prompt)}")
        try:
            chat_messages = Discussion([system_prompt, main_prompt.prompt])
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=main_prompt._parse_answer,
            )
            ans_dict, did_loop_replan = self._apply_loop_breaker(
                ans_dict=ans_dict,
                main_prompt=main_prompt,
                system_prompt=system_prompt,
            )
            print("the ans_dict is", ans_dict)
            ans_dict["busted_retry"] = 0
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2 + (1 if did_loop_replan else 0)
        except ParseError:
            ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )
        stats = self.chat_llm.get_stats()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]

        self.actions.append(ans_dict["action"])
        if "displayed_action" in ans_dict:
            self.displayed_actions.append(ans_dict["displayed_action"])
        else:  # catch KeyError, this only happens for faulty parsing and hence None action anyways
            self.displayed_actions.append(ans_dict["action"])
        self.thoughts.append(ans_dict.get("think", None))

        agent_info = AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={"chat_model_args": asdict(self.chat_model_args)},
        )
        return ans_dict["action"], agent_info

    def reset(self, seed=None):
        self.seed = seed
        self.thoughts = []
        self.actions = []
        self.displayed_actions = []
        self.obs_history = []
