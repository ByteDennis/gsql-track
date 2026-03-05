"""
Standardized prompt registry system and batch inference infrastructure.

Provides:
- A registry for prompt-building functions with metadata about
  token budgets, concept/fewshot requirements, and auto-dispatch between
  text classification and relation extraction inputs.
- A generic batch inference runner for Together API and Anthropic Batch API
  (submit, poll, download).
- Batch job tracking dataclass.

All problem-specific logic (prompt implementations, result parsing, concept
parsing, few-shot generation) belongs in the consuming project.
"""
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .util import fmt_duration


# ──────────────────────────────────────────────────────────────────
# Prompt Registry
# ──────────────────────────────────────────────────────────────────

@dataclass
class PromptInput:
    """Unified input for all prompt functions.

    Handles both text classification (str text) and relation extraction
    (dict with text/entity1/entity2) through a single interface.

    Example
    -------
    >>> inp = PromptInput(text="Some text", labels=["pos", "neg"])
    >>> inp = PromptInput.from_example({"text": "...", "entity1": "A", "entity2": "B"}, labels=["pos", "neg"])
    """
    text: str
    labels: List[str] = field(default_factory=list)
    entity1: Optional[str] = None
    entity2: Optional[str] = None
    concepts: List[Dict[str, str]] = field(default_factory=list)
    fewshot_examples: List[Dict[str, str]] = field(default_factory=list)

    @property
    def is_relation(self) -> bool:
        """True if this is a relation extraction input (has entity pair)."""
        return self.entity1 is not None and self.entity2 is not None

    @classmethod
    def from_example(
        cls,
        text_or_dict: Union[str, Dict[str, Any]],
        labels: Optional[List[str]] = None,
        concepts: Optional[List[Dict[str, str]]] = None,
        fewshot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> "PromptInput":
        """Create PromptInput from a text string or relation dict.

        Example
        -------
        >>> PromptInput.from_example("hello", labels=["a", "b"])
        >>> PromptInput.from_example({"text": "hello", "entity1": "A", "entity2": "B"}, labels=["a", "b"])
        """
        if isinstance(text_or_dict, dict):
            return cls(
                text=text_or_dict["text"],
                labels=labels or [],
                entity1=text_or_dict.get("entity1"),
                entity2=text_or_dict.get("entity2"),
                concepts=concepts or [],
                fewshot_examples=fewshot_examples or [],
            )
        return cls(
            text=text_or_dict,
            labels=labels or [],
            concepts=concepts or [],
            fewshot_examples=fewshot_examples or [],
        )


@dataclass
class PromptMetadata:
    """Metadata describing a prompt's behavior and requirements.

    Example
    -------
    >>> meta = PromptMetadata(name='direct_classify', max_tokens=50)
    """
    name: str
    max_tokens: int = 256
    needs_concepts: bool = False
    needs_fewshot: bool = False
    description: str = ""

    def __repr__(self):
        flags = []
        if self.needs_concepts:
            flags.append("concepts")
        if self.needs_fewshot:
            flags.append("fewshot")
        flag_str = f", needs=[{','.join(flags)}]" if flags else ""
        return f"Prompt({self.name}, max_tokens={self.max_tokens}{flag_str})"


class PromptWrapper:
    """Wrapper that pairs a prompt function with its metadata.

    Example
    -------
    >>> wrapper = PromptWrapper(func, metadata)
    >>> system_prompt, user_prompt = wrapper(inp)
    """

    def __init__(self, func: Callable[[PromptInput], tuple[str, str]], metadata: PromptMetadata):
        self.func = func
        self.metadata = metadata
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __call__(self, inp: PromptInput) -> tuple[str, str]:
        return self.func(inp)

    def __repr__(self):
        return f"<PromptWrapper: {self.metadata}>"


# Global prompt registry
PROMPT_REGISTRY: Dict[str, PromptWrapper] = {}
PROMPT_METADATA: Dict[str, PromptMetadata] = {}


def register_prompt(
    name: str,
    max_tokens: int = 256,
    needs_concepts: bool = False,
    needs_fewshot: bool = False,
    description: str = "",
):
    """Decorator to register prompt-building functions with metadata.

    Example
    -------
    >>> @register_prompt('direct_classify', max_tokens=50)
    ... def direct_classify(inp: PromptInput) -> tuple[str, str]:
    ...     return system_prompt, user_prompt
    """
    def decorator(func: Callable[[PromptInput], tuple[str, str]]):
        metadata = PromptMetadata(
            name=name,
            max_tokens=max_tokens,
            needs_concepts=needs_concepts,
            needs_fewshot=needs_fewshot,
            description=description or func.__doc__ or "",
        )

        wrapped = PromptWrapper(func, metadata)
        PROMPT_REGISTRY[name] = wrapped
        PROMPT_METADATA[name] = metadata

        return wrapped
    return decorator


def get_prompt(name: str) -> PromptWrapper:
    """Get prompt wrapper by name.

    Example
    -------
    >>> prompt_fn = get_prompt('direct_classify')
    >>> system, user = prompt_fn(inp)
    """
    if name not in PROMPT_REGISTRY:
        available = ", ".join(PROMPT_REGISTRY.keys())
        raise KeyError(f"Prompt '{name}' not found. Available: {available}")
    return PROMPT_REGISTRY[name]


def list_prompts(
    needs_concepts: Optional[bool] = None,
    needs_fewshot: Optional[bool] = None,
) -> Dict[str, PromptMetadata]:
    """List all registered prompts, optionally filtered.

    Example
    -------
    >>> prompts = list_prompts(needs_concepts=True)
    """
    filtered = {}
    for name, metadata in PROMPT_METADATA.items():
        if needs_concepts is not None and metadata.needs_concepts != needs_concepts:
            continue
        if needs_fewshot is not None and metadata.needs_fewshot != needs_fewshot:
            continue
        filtered[name] = metadata
    return filtered


# ──────────────────────────────────────────────────────────────────
# Batch Job Tracking
# ──────────────────────────────────────────────────────────────────

@dataclass
class BatchJobInfo:
    """Track batch job information (problem-agnostic)."""
    batch_id: str
    task: str
    split: str
    model: str
    status: str  # pending, in_progress, completed, failed, cancelled, expired
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    error_message: Optional[str] = None
    unit_pricing_usd_per_1m: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "task": self.task,
            "split": self.split,
            "model": self.model,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchJobInfo":
        return cls(**data)


# ──────────────────────────────────────────────────────────────────
# Generic Utilities
# ──────────────────────────────────────────────────────────────────

def extract_response_content(batch_item: Dict[str, Any]) -> str:
    """Extract the raw content string from a Together batch API response item.

    This handles the nested Together AI batch response format:
    ``response.body.choices[0].message.content``

    Problem-specific parsing of the content string should be done by the caller.
    """
    return (
        batch_item.get("response", {})
        .get("body", {})
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )


# ──────────────────────────────────────────────────────────────────
# Model Tag Derivation
# ──────────────────────────────────────────────────────────────────

def derive_model_tag(model_name: str) -> str:
    """Derive a short model tag from a full model name.

    Examples:
        Qwen/Qwen2.5-7B-Instruct-Turbo → qwen2.5-7b
        meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo → llama-3.1-8b
        mistralai/Mistral-7B-Instruct-v0.3 → mistral-7b
        claude-haiku-4-5-20251001 → haiku-4.5
        claude-sonnet-4-5-20250929 → sonnet-4.5
    """
    # Handle Claude model names: claude-{family}-{ver}-{date}
    m = re.match(r"claude-([a-z]+)-(\d+)-(\d+)-\d+", model_name)
    if m:
        family, major, minor = m.group(1), m.group(2), m.group(3)
        return f"{family}-{major}.{minor}"

    # Take the part after the slash (or the whole thing)
    name = model_name.split("/")[-1]
    # Lowercase
    name = name.lower()
    # Remove common suffixes
    for suffix in ["-turbo", "-instruct", "-chat", "-hf"]:
        name = name.replace(suffix, "")
    # Remove active-param suffixes like "-a3b"
    name = re.sub(r"-a\d+\.?\d*b$", "", name)
    # Remove "meta-" prefix
    name = re.sub(r"^meta-", "", name)
    # Extract model family + size pattern like "qwen2.5-7b", "llama-3.1-8b", "qwen3-next-80b"
    # Match: word/digits with optional middle segments, then dash, then size like 7b/8b/70b
    m = re.search(r"([a-z]+[\d.]*(?:-[a-z\d.]+)*)-(\d+\.?\d*b)$", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # Fallback: just use cleaned name
    return name


# ──────────────────────────────────────────────────────────────────
# Batch Inference Runner (problem-agnostic API mechanics)
# ──────────────────────────────────────────────────────────────────

class BatchInferenceRunner:
    """Generic batch inference runner for Together API and Anthropic Batch API.

    Handles only API mechanics: batch file creation (via prompt registry),
    submission, polling, downloading, and cost tracking.

    All problem-specific logic (result parsing, concept generation,
    few-shot generation) should live in the consuming project.

    Parameters
    ----------
    api_key : str
        API key (Together or Anthropic depending on backend).
    model : str
        Model identifier (e.g. "Qwen/Qwen2.5-7B-Instruct-Turbo" or "claude-haiku-4-5-20251001").
    output_dir : Path
        Directory for batch input/output files.
    llm_config : dict, optional
        Override defaults for max_tokens, temperature, etc.
    system_context : str, optional
        Context string prepended to all system prompts (e.g. task description).
    backend : str, optional
        API backend: "together" (default) or "anthropic".
    """

    # Token pricing (per 1M tokens) - Batch API is 50% of real-time pricing
    PRICING = {
        "input": {
            "meta-llama/Llama-3.3-70B-Instruct-Turbo": 0.59 * 0.5,
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": 0.18 * 0.5,
            "meta-llama/Llama-3-8b-chat-hf": 0.15 * 0.5,
            "Qwen/Qwen2.5-7B-Instruct-Turbo": 0.10 * 0.5,
            "Qwen/Qwen2.5-72B-Instruct-Turbo": 0.59 * 0.5,
            "Qwen/Qwen3-Next-80B-A3B-Instruct": 0.15 * 0.5,
            "mistralai/Mistral-7B-Instruct-v0.3": 0.10 * 0.5,
            "claude-haiku-4-5-20251001": 0.80 * 0.5,
            "claude-sonnet-4-5-20250929": 3.00 * 0.5,
            "default": 0.10 * 0.5,
        },
        "output": {
            "meta-llama/Llama-3.3-70B-Instruct-Turbo": 0.79 * 0.5,
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": 0.18 * 0.5,
            "meta-llama/Llama-3-8b-chat-hf": 0.15 * 0.5,
            "Qwen/Qwen2.5-7B-Instruct-Turbo": 0.10 * 0.5,
            "Qwen/Qwen2.5-72B-Instruct-Turbo": 0.79 * 0.5,
            "Qwen/Qwen3-Next-80B-A3B-Instruct": 1.50 * 0.5,
            "mistralai/Mistral-7B-Instruct-v0.3": 0.10 * 0.5,
            "claude-haiku-4-5-20251001": 4.00 * 0.5,
            "claude-sonnet-4-5-20250929": 15.00 * 0.5,
            "default": 0.10 * 0.5,
        },
    }

    def __init__(
        self,
        api_key: str,
        model: str,
        output_dir: Path,
        llm_config: Optional[Dict[str, Any]] = None,
        system_context: str = "",
        backend: str = "together",
    ):
        if backend not in ("together", "anthropic"):
            raise ValueError(f"backend must be 'together' or 'anthropic', got '{backend}'")

        self.backend = backend
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_config = llm_config or {}
        self.system_context = system_context

        if backend == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required for Anthropic batch inference. "
                    "Install with: pip install anthropic"
                )
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            try:
                from together import Together
            except ImportError:
                raise ImportError(
                    "together package required for batch inference. "
                    "Install with: pip install together"
                )
            self.client = Together(api_key=api_key)

    # ── Cost estimation ──────────────────────────────────────────

    def _get_pricing(self, token_type: str) -> float:
        """Get pricing per 1M tokens for this model."""
        return self.PRICING[token_type].get(self.model, self.PRICING[token_type]["default"])

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD."""
        input_cost = (input_tokens / 1_000_000) * self._get_pricing("input")
        output_cost = (output_tokens / 1_000_000) * self._get_pricing("output")
        return input_cost + output_cost

    # ── Real-time generation (single call) ───────────────────────

    def realtime_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Make a single real-time (non-batch) API call. Returns content string."""
        if self.backend == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
            )
            return response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

    # ── Batch file creation ──────────────────────────────────────

    def _create_batch_request(
        self,
        custom_id: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Create a single batch request in the appropriate API format."""
        if self.system_context:
            system_prompt = f"{self.system_context}\n\n{system_prompt}"

        request_max_tokens = max_tokens if max_tokens is not None else self.llm_config.get("max_tokens", 256)
        request_temperature = self.llm_config.get("temperature", 0.0)
        # Claude pretty-prints JSON with markdown fences; add headroom to avoid truncation
        if self.backend == "anthropic":
            request_max_tokens = int(request_max_tokens * 1.5)

        if self.backend == "anthropic":
            return {
                "custom_id": custom_id,
                "params": {
                    "model": self.model,
                    "max_tokens": request_max_tokens,
                    "temperature": request_temperature,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": user_prompt},
                    ],
                },
            }
        else:
            return {
                "custom_id": custom_id,
                "body": {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": request_max_tokens,
                    "temperature": request_temperature,
                },
            }

    def create_batch_file(
        self,
        texts: List,
        ids: List[str],
        task: str,
        saved_name: Optional[str] = None,
        **prompt_kwargs,
    ) -> Path:
        """Create JSONL batch file for submission.

        Dispatches to registered prompts via the PROMPT_REGISTRY.
        ``prompt_kwargs`` are forwarded to ``PromptInput.from_example``
        (e.g. labels, concepts, fewshot_examples).

        Parameters
        ----------
        saved_name : str, optional
            Filename stem for input/output JSONL files (e.g. "concept_train_valid").
            Defaults to ``task``. The same name is used for both
            ``inputs/{saved_name}.jsonl`` and ``outputs/{saved_name}.jsonl``.
        """
        saved_name = saved_name or task
        self._last_saved_name = saved_name

        inputs_dir = self.output_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        batch_file = inputs_dir / f"{saved_name}.jsonl"

        prompt_fn = get_prompt(task)
        max_tokens = prompt_fn.metadata.max_tokens

        requests = []
        for text_id, text in zip(ids, texts):
            inp = PromptInput.from_example(text, **prompt_kwargs)
            system_prompt, user_prompt = prompt_fn(inp)
            request = self._create_batch_request(text_id, system_prompt, user_prompt, max_tokens)
            requests.append(request)

        with open(batch_file, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")

        print(f"Created batch file: {batch_file} with {len(requests)} requests")
        return batch_file

    # ── Batch submission & polling ───────────────────────────────

    def submit_batch(self, batch_file: Path) -> str:
        """Upload batch file and create batch job. Returns batch_id."""
        if self.backend == "anthropic":
            print("Submitting batch to Anthropic API...")
            requests = []
            with open(batch_file, "r") as f:
                for line in f:
                    if line.strip():
                        requests.append(json.loads(line))

            batch = self.client.messages.batches.create(requests=requests)
            batch_id = batch.id
            print(f"Batch created: {batch_id}")
            print(f"Status: {batch.processing_status}")
            return batch_id
        else:
            print("Uploading batch file to Together API...")
            file_resp = self.client.files.upload(file=str(batch_file), purpose="batch-api", check=False)
            print(f"File uploaded: {file_resp.id}")

            print("Creating batch job...")
            batch_response = self.client.batches.create(
                endpoint="/v1/chat/completions",
                input_file_id=file_resp.id,
            )
            batch_id = batch_response.job.id
            print(f"Batch job created: {batch_id}")
            print(f"Status: {batch_response.job.status}")
            return batch_id

    def poll_batch(
        self,
        batch_id: str,
        poll_interval: int = 30,
        timeout: Optional[int] = None,
    ) -> Any:
        """Poll batch job until complete or timeout. Returns batch_job object."""
        print(f"Polling batch {batch_id} (interval={poll_interval}s)...")

        start_time = time.time()
        while True:
            elapsed = time.time() - start_time

            if self.backend == "anthropic":
                batch_job = self.client.messages.batches.retrieve(batch_id)
                status = batch_job.processing_status or "unknown"
                counts = batch_job.request_counts
                total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
                done = counts.succeeded + counts.errored + counts.canceled + counts.expired
                progress = (done / total * 100) if total else 0
                print(f"[{elapsed:.0f}s] Status: {status:15} | Progress: {progress:5.1f}% ({done}/{total})")

                if status == "ended":
                    print(" - DONE!")
                    return batch_job
                elif status in ["canceled", "expired"]:
                    raise Exception(f"Batch {status}")
            else:
                batch_job = self.client.batches.retrieve(batch_id)
                status = batch_job.status or "UNKNOWN"
                progress = getattr(batch_job, 'progress', 0)
                print(f"[{elapsed:.0f}s] Status: {status:15} | Progress: {progress:5.1f}%")

                if status == "COMPLETED":
                    print(" - DONE!")
                    return batch_job
                elif status in ["FAILED", "CANCELLED", "EXPIRED"]:
                    error_msg = getattr(batch_job, 'error', 'Unknown error')
                    raise Exception(f"Batch {status}: {error_msg}")

            if timeout and elapsed > timeout:
                raise Exception(f"Batch polling timeout after {timeout}s")

            time.sleep(poll_interval)

    def check_batch(self, batch_id: str) -> Any:
        """Check batch status without polling. Returns batch_job object."""
        if self.backend == "anthropic":
            return self.client.messages.batches.retrieve(batch_id)
        return self.client.batches.retrieve(batch_id)

    # ── Result downloading ───────────────────────────────────────

    @staticmethod
    def _normalize_anthropic_result(result) -> dict:
        """Convert an Anthropic batch result to the Together-normalized JSONL format.

        This lets all downstream code (extract_response_content, get_batch_stats)
        work uniformly regardless of backend.
        """
        content = ""
        if result.result.type == "succeeded":
            msg = result.result.message
            if msg.content:
                content = msg.content[0].text
            usage = msg.usage
            normalized = {
                "custom_id": result.custom_id,
                "response": {
                    "body": {
                        "choices": [{"message": {"content": content}}],
                        "usage": {
                            "prompt_tokens": usage.input_tokens,
                            "completion_tokens": usage.output_tokens,
                        },
                    },
                },
            }
        else:
            # errored / canceled / expired
            normalized = {
                "custom_id": result.custom_id,
                "response": {
                    "body": {
                        "choices": [{"message": {"content": ""}}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    },
                },
                "error": str(result.result.type),
            }
        return normalized

    def download_results(self, batch_job: Any, saved_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Download and parse batch results JSONL from completed job.

        Parameters
        ----------
        saved_name : str, optional
            Filename stem for the output file. Defaults to the name used
            in the most recent ``create_batch_file`` call.

        Returns raw batch result items in Together-normalized format.
        Use ``extract_response_content`` to get content strings, then
        problem-specific parsing functions to interpret them.
        """
        saved_name = saved_name or getattr(self, '_last_saved_name', 'batch_output')
        outputs_dir = self.output_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        full_output_path = outputs_dir / f"{saved_name}.jsonl"

        if self.backend == "anthropic":
            print("Downloading results from Anthropic API...")
            results = []
            with open(full_output_path, "w") as f:
                for result in self.client.messages.batches.results(batch_job.id):
                    normalized = self._normalize_anthropic_result(result)
                    f.write(json.dumps(normalized) + "\n")
                    results.append(normalized)
            self._last_output_path = full_output_path
            print(f"Results downloaded to {full_output_path}")
            print(f"Parsed {len(results)} results")
            return results
        else:
            if not batch_job.output_file_id:
                raise Exception("No output file ID in completed batch")

            print("Downloading results...")
            response = self.client.files.content(batch_job.output_file_id)
            response.write_to_file(str(full_output_path))
            self._last_output_path = full_output_path
            print(f"Results downloaded to {full_output_path}")

            results = []
            with open(full_output_path, "r") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))

            print(f"Parsed {len(results)} results")
            return results

    # ── Batch stats helper ──────────────────────────────────────

    def get_batch_stats(self, output_path: Optional[Path] = None) -> dict:
        """Read batch stats from an output JSONL file.

        Extracts token counts, cost, and elapsed time (max - min of
        ``response.body.created`` unix timestamps) from the output file.

        Args:
            output_path: Path to output JSONL.  Falls back to the most
                recently downloaded file (``self._last_output_path``).
        """
        path = output_path or getattr(self, '_last_output_path', None)
        input_tokens = 0
        output_tokens = 0
        min_created = float('inf')
        max_created = float('-inf')

        if path and Path(path).exists():
            with open(path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        body = data.get("response", {}).get("body", {})
                        usage = body.get("usage", {})
                        input_tokens += usage.get("prompt_tokens", 0)
                        output_tokens += usage.get("completion_tokens", 0)
                        created = body.get("created", 0)
                        if created:
                            min_created = min(min_created, created)
                            max_created = max(max_created, created)
                    except (json.JSONDecodeError, KeyError):
                        continue

        cost = self._estimate_cost(input_tokens, output_tokens)
        stats = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(cost, 4),
        }
        if min_created < float('inf') and max_created > float('-inf'):
            elapsed = max_created - min_created
            stats["batch_elapsed"] = round(elapsed, 1)
            stats["batch_elapsed_human"] = fmt_duration(elapsed)
        return stats

    def save_batch_stats(
        self,
        stats_key: str,
        stats_path: Path,
        *,
        dataset: str = "",
        output_path: Optional[Path] = None,
    ) -> dict:
        """Compute batch stats and merge into stats.json under ``stats_key``.

        Loads existing stats.json (if any), sets ``dataset`` and ``model``,
        writes ``get_batch_stats()`` under the given key, and saves.
        Returns the batch stats dict.
        """
        stats_path = Path(stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        existing = {}
        if stats_path.exists():
            with open(stats_path, "r") as f:
                existing = json.load(f)

        if dataset:
            existing.setdefault("dataset", dataset)
        existing.setdefault("model", self.model)

        batch_stats = self.get_batch_stats(output_path)
        existing[stats_key] = batch_stats

        with open(stats_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  Batch stats [{stats_key}] saved to {stats_path}")
        return batch_stats



__all__ = [
    # Prompt registry
    "PromptInput",
    "PromptMetadata",
    "PromptWrapper",
    "PROMPT_REGISTRY",
    "PROMPT_METADATA",
    "register_prompt",
    "get_prompt",
    "list_prompts",
    # Model utilities
    "derive_model_tag",
    # Batch inference (generic)
    "BatchJobInfo",
    "extract_response_content",
    "BatchInferenceRunner",
]
