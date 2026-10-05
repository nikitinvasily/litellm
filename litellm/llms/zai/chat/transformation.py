from typing import Final

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

ZAI_API_BASE: Final = "https://api.z.ai/api/paas/v4"

# Z.AI accepts reasoning_effort on the Coding Plan endpoint and maps OpenAI-scale
# values server-side (none/minimal/low -> low, medium/high -> high, xhigh/max -> max).
# Our router's tier efforts are tuned per GPT model (MEDIUM sends xhigh to a small
# model, COMPLEX sends low to a large one), so the server-side image of those values
# is not monotonic in task complexity. These per-model tables restore a monotonic
# GLM ladder from the values our tiers actually send:
#   glm-5.3:      MEDIUM xhigh->high, COMPLEX low->high, REASONING medium->max
#   glm-5.3-flash: SIMPLE medium->low
_ZAI_EFFORT_TRANSLATION: Final = {
    "glm-5.3": {"low": "high", "medium": "max", "xhigh": "high"},
    "glm-5.3-flash": {"medium": "low"},
}


class ZAIChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "zai"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = api_base or get_secret_str("ZAI_API_BASE") or ZAI_API_BASE
        dynamic_api_key: Final = api_key or get_secret_str("ZAI_API_KEY")
        return api_base, dynamic_api_key

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: list[AllMessageValues],
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> tuple[list[AllMessageValues], list[ChatCompletionToolParam] | None]:
        """Override to preserve cache_control for GLM/ZAI.

        GLM supports cache_control - don't strip it.
        """
        # GLM/ZAI supports cache_control, so return messages and tools unchanged
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list:
        base_params: Final = [
            "max_tokens",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
            "reasoning_effort",
        ]

        import litellm

        try:
            if litellm.supports_reasoning(model=model, custom_llm_provider=self.custom_llm_provider):
                base_params.append("thinking")
        except Exception:
            pass

        return base_params

    def map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        effort: Final = non_default_params.pop("reasoning_effort", None)
        translated: Final = (
            _ZAI_EFFORT_TRANSLATION.get(model, {}).get(str(effort), effort) if effort is not None else None
        )
        if translated is not None:
            optional_params["reasoning_effort"] = translated
        return super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
