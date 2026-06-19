import json
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List
import string

import numpy as np
import torch
from openai import OpenAI
from qwen_asr import Qwen3ASRModel
from simulstream.server.speech_processors import SAMPLE_RATE, SpeechProcessor
from simulstream.server.speech_processors.incremental_output import IncrementalOutput
from vllm import LLM, SamplingParams

logging.getLogger("fbk_fairseq.simultaneous.metrics").setLevel(logging.INFO)


def longest_common_prefix(s1: str, s2: str) -> str:
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[: min(len(s1), len(s2))]


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def find_end_time(time_stamps, position: int, text: str) -> float:
    if len(time_stamps) != len(remove_punctuation(text).split()):
        print(
            f"number of time stamps and words in text do not match\n"
            f"time_stamps: {time_stamps}\ntext: {text.split()}"
        )
        return None
    n_words_right = len(remove_punctuation(text[position + 1 :]).strip().split())
    return time_stamps[-n_words_right - 1].end_time


@dataclass
class CascadeState:
    speech_id: int = 0
    source: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    utt_timestamps: List[int] = field(default_factory=lambda: [0])
    utt_sources: List[str] = field(default_factory=lambda: [""])
    utt_targets: List[str] = field(default_factory=lambda: [""])
    asr_hypotheses: List[str] = field(default_factory=lambda: [""])
    translation_hypotheses: List[str] = field(default_factory=lambda: [""])
    translations: List[str] = field(default_factory=lambda: [""])
    emission_started: bool = False


class CascadeSpeechProcessor(SpeechProcessor):
    """
    SimulStream speech processor: ASR (Qwen3) + LLM (Qwen3) con epsilon sampling
    y decodificación MBRS configurable.

    Claves de configuración existentes: (sin cambios)
        asr_model_name, llm_model_name, source_lang, target_lang,
        min_start_seconds, max_history_utterances, max_new_tokens,
        temperature, top_p, top_k, repetition_penalty, n_samples,
        latency_unit, ner_results_path, abstract_results_path.

    Claves de configuración nuevas:
        decoding_method   : "none" | "rambr_chrf" | "prunembr_xcomet" | "rerank_kiwi"
                            (default: "none"  →  comportamiento idéntico al original)
        epsilon           : float, umbral de epsilon-sampling (default: 0.02)
        xcomet_model      : str   (default: "myyycroft/XCOMET-lite")
        xcomet_batch_size : int   (default: 16)
        kiwi_model        : str   (default: "Unbabel/wmt22-cometkiwi-da")
        kiwi_batch_size   : int   (default: 8)
    """

    # ------------------------------------------------------------------ #
    #  Carga de modelos (nivel clase, compartida entre instancias)         #
    # ------------------------------------------------------------------ #
    @classmethod
    def load_model(cls, config: SimpleNamespace):
        # gpu_memory_utilization configurable por YAML. Los defaults son los
        # valores históricos (no cambian el comportamiento de experimentos que
        # no fijen estas claves). Bajarlos deja más headroom en la GPU para el
        # forced aligner / workspace de cuBLAS (evita CUBLAS_STATUS_NOT_INITIALIZED
        # en GPUs compartidas).
        asr_gpu_mem = getattr(config, "asr_gpu_memory_utilization", 0.36)
        llm_gpu_mem = getattr(config, "llm_gpu_memory_utilization", 0.40)

        # --- ASR ------------------------------------------------------- #
        if not hasattr(cls, "asr") or cls.asr is None:
            cls.asr = Qwen3ASRModel.LLM(
                model=config.asr_model_name,
                gpu_memory_utilization=asr_gpu_mem,
                max_inference_batch_size=1,
                max_model_len=1024,
                max_new_tokens=1024,
                forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                forced_aligner_kwargs=dict(
                    dtype=torch.bfloat16,
                    device_map="cuda",
                ),
            )

        # --- LLM ------------------------------------------------------- #
        llm_base_url = getattr(config, "llm_base_url", None)
        if llm_base_url is not None:
            if not hasattr(cls, "llm_client") or cls.llm_client is None:
                cls.llm_client = OpenAI(base_url=llm_base_url, api_key="EMPTY")
                from transformers import AutoTokenizer
                cls.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)
            cls.llm = None
        else:
            cls.llm_client = None
            if not hasattr(cls, "llm") or cls.llm is None:
                cls.llm = LLM(
                    model=config.llm_model_name,
                    trust_remote_code=True,
                    gpu_memory_utilization=llm_gpu_mem,
                    tensor_parallel_size=1,
                    max_num_seqs=1,
                    max_model_len=1024,
                    enable_prefix_caching=True,
                )
                cls.tokenizer = cls.llm.get_tokenizer()

        # --- Decoder MBRS ---------------------------------------------- #
        # Solo se carga si todavía no existe en la clase.
        # Si cambias de decoding_method entre experimentos, reinicia el proceso.
        decoding_method = getattr(config, "decoding_method", "none")

        if not hasattr(cls, "mbr_decoder"):
            cls.mbr_decoder = None

        if cls.mbr_decoder is not None:
            return  # ya estaba cargado

        if decoding_method == "rambr_chrf":
            # Aggregate-MBR con chrF (fastchrf=True): lineal en |H|, sin GPU.
            from mbrs.metrics import MetricChrF
            from mbrs.decoders import DecoderAggregateMBR

            metric = MetricChrF(MetricChrF.Config(fastchrf=True))
            cls.mbr_decoder = DecoderAggregateMBR(DecoderAggregateMBR.Config(), metric)
            logging.info("[MBRS] DecoderAggregateMBR + chrF (fast) cargado.")

        elif decoding_method == "prunembr_xcomet":
            # Pruning-MBR con XCOMET-lite: poda basada en confianza, requiere GPU.
            from mbrs.metrics import MetricXCOMET
            from mbrs.decoders import DecoderPruningMBR

            xcomet_model = getattr(config, "xcomet_model", "myyycroft/XCOMET-lite")
            metric = MetricXCOMET(
                MetricXCOMET.Config(
                    model=xcomet_model,
                    batch_size=getattr(config, "xcomet_batch_size", 16),
                    fp16=True,
                )
            )
            cls.mbr_decoder = DecoderPruningMBR(DecoderPruningMBR.Config(), metric)
            logging.info(f"[MBRS] DecoderPruningMBR + XCOMET ({xcomet_model}) cargado.")

        elif decoding_method == "rerank_kiwi":
            # N-best reranking con COMETkiwi: reference-free, solo necesita source.
            from mbrs.metrics import MetricCOMETkiwi
            from mbrs.decoders import DecoderRerank

            kiwi_model = getattr(config, "kiwi_model", "Unbabel/wmt22-cometkiwi-da")
            metric = MetricCOMETkiwi(
                MetricCOMETkiwi.Config(
                    model=kiwi_model,
                    batch_size=getattr(config, "kiwi_batch_size", 8),
                    fp16=True,
                )
            )
            cls.mbr_decoder = DecoderRerank(DecoderRerank.Config(), metric)
            logging.info(f"[MBRS] DecoderRerank + COMETkiwi ({kiwi_model}) cargado.")

        elif decoding_method != "none":
            logging.warning(
                f"[MBRS] decoding_method='{decoding_method}' desconocido. "
                "Usando decodificación argmax (none)."
            )

    # ------------------------------------------------------------------ #
    #  Constructor                                                         #
    # ------------------------------------------------------------------ #
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.load_model(config)

        self.source_lang = getattr(config, "source_lang", "English")
        self.target_lang = getattr(config, "target_lang", "Chinese")
        self.target_sep = "" if self.target_lang in ["Chinese", "Japanese"] else " "
        self.latency_unit = getattr(config, "latency_unit", "word")

        self.min_start_seconds = getattr(config, "min_start_seconds", 2.0)
        self.max_history_utterances = getattr(config, "max_history_utterances", 0)

        self._temperature = getattr(config, "temperature", 1.0)
        self._top_p = getattr(config, "top_p", 0.9)
        self._top_k = getattr(config, "top_k", 20)
        self._max_tokens = getattr(config, "max_new_tokens", 512)
        self._repetition_penalty = getattr(config, "repetition_penalty", 1.05)
        self._llm_model_name = config.llm_model_name

        # Método de decodificación MBRS
        self.decoding_method = getattr(config, "decoding_method", "none")

        # Epsilon sampling: filtra tokens con prob < epsilon antes de muestrear.
        # Con top_k=-1 desactivamos top-k y dejamos que epsilon actúe solo.
        _epsilon = getattr(config, "epsilon", 0.02)
        self.sampling_params = SamplingParams(
            n=getattr(config, "n_samples", 1),
            top_k=-1,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            repetition_penalty=self._repetition_penalty,
            extra_args={"epsilon": _epsilon},
            stop=["\n"],
            seed=123,
        )

        abstract_results_path = getattr(config, "abstract_results_path", None)
        self.abstract_results = (
            self._load_abstract_results(abstract_results_path)
            if abstract_results_path is not None
            else None
        )

        ner_results_path = getattr(config, "ner_results_path", None)
        self.ner_results = (
            self._load_ner_results(ner_results_path)
            if ner_results_path is not None
            else None
        )

        self._state = CascadeState()

    # ------------------------------------------------------------------ #
    #  Métodos auxiliares (sin cambios respecto al original)               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_abstract_results(abstract_results_path: str) -> List[str]:
        with open(abstract_results_path, "r", encoding="utf-8") as f:
            abstract_results = json.load(f)
        return [result["abstract"] for result in abstract_results]

    @staticmethod
    def _load_ner_results(ner_results_path: str) -> List[str]:
        with open(ner_results_path, "r", encoding="utf-8") as f:
            ner_results = json.load(f)
        return [", ".join(result["entities"]) for result in ner_results]

    @staticmethod
    def _n_utterances(text: str) -> int:
        n_utt = text.count(". ") + text.count("! ") + text.count("? ")
        if text.endswith((".", "!", "?")):
            n_utt += 1
        return n_utt

    def _asr_stable_prefix(self, prev_hypo: str, curr_hypo: str) -> str:
        """Prefijo estable del ASR (política de emisión).

        Por defecto: local agreement clásico = prefijo común exacto (char-level)
        entre la hipótesis del instante anterior y la actual. Las subclases
        pueden sobreescribir esto para implementar otras políticas de emisión
        (hold-n, tolerant agreement, ...).
        """
        return longest_common_prefix(prev_hypo, curr_hypo)

    def _mt_stable_prefix(
        self,
        prev_hypo: str,
        curr_hypo: str,
        committed: str,
        n_source_words: int,
    ) -> str:
        """Prefijo estable de la TRADUCCIÓN (política de emisión de MT).

        Por defecto: Local Agreement estricto = prefijo común exacto (char-level)
        entre la hipótesis de traducción del instante anterior y la actual. Las
        subclases pueden sobreescribir esto para implementar otras políticas de
        emisión del target (wait-k, tolerant agreement, híbrido, ...).

        Devuelve el prefijo comprometido COMPLETO (prefijo de ``curr_hypo`` y de
        longitud >= ``committed``); el agente calcula el incremento emitido como
        ``nuevo[len(committed):]``.
        """
        return longest_common_prefix(prev_hypo, curr_hypo)

    def _transcribe_audio(self, state: CascadeState):
        audio = np.array(
            state.source[state.utt_timestamps[-1 - self.max_history_utterances] :]
        )
        if self.ner_results is not None:
            asr_context = self.ner_results[state.speech_id]
        elif self.abstract_results is not None:
            asr_context = self.abstract_results[state.speech_id]
        else:
            asr_context = ""

        asr_outputs = self.asr.transcribe(
            (audio, SAMPLE_RATE),
            language=self.source_lang,
            context=asr_context,
            return_time_stamps=True,
        )

        if asr_outputs[0].time_stamps is not None and \
                asr_outputs[0].time_stamps[-1].end_time > len(audio) / SAMPLE_RATE:
            return None, False

        asr_hypo = asr_outputs[0].text
        state.asr_hypotheses.append(asr_hypo)

        asr_segment = self._asr_stable_prefix(
            state.asr_hypotheses[-2], state.asr_hypotheses[-1]
        )
        if self._n_utterances(asr_segment) >= 1:
            rightest_punct_idx = max(
                asr_segment.rfind(". "),
                asr_segment.rfind("! "),
                asr_segment.rfind("? "),
            )
            if rightest_punct_idx == -1 and asr_segment.endswith((".", "!", "?")):
                rightest_punct_idx = len(asr_segment) - 1
            find_end_time_result = find_end_time(
                asr_outputs[0].time_stamps, rightest_punct_idx, asr_hypo
            )
            if find_end_time_result is None:
                return None, False
            utt_end_time = (
                int(find_end_time_result * SAMPLE_RATE)
                + state.utt_timestamps[-1]
            )
            utt_end_time = min(utt_end_time, len(state.source))
            state.utt_timestamps.append(utt_end_time)
            state.utt_sources.append(asr_segment[: rightest_punct_idx + 1])
            state.asr_hypotheses = [asr_hypo[rightest_punct_idx + 1 :].strip()]
            asr_to_translate = " ".join(
                state.utt_sources[-1 - self.max_history_utterances :]
            )
            return asr_to_translate, True

        if self.max_history_utterances > 0:
            asr_to_translate = " ".join(
                state.utt_sources[-self.max_history_utterances :] + [asr_hypo]
            )
        else:
            asr_to_translate = asr_hypo
        return asr_to_translate, False

    def _prepare_llm_inputs(
        self, asr_segment: str, prev_translation: str, context: str
    ) -> str:
        context_prompt = (
            f"\n\n[CONTEXT]\n\n{context}" if context != "" else ""
        )
        context_hint = (
            "\nUse the provided context only to correctly resolve named entities."
            if context != ""
            else ""
        )
        instruction = (
            f"You are a professional translator.\n\n"
            f"[TASK]\n"
            f"Translate the input text into {self.target_lang}.\n"
            f"Preserve all named entities, such as person names, model names, "
            f"dataset names, and metric names, exactly as they appear in the input text."
            f"{context_hint}\n"
            f"Return only the translated text without any additional explanation."
            f"{context_prompt}\n\n"
            f"[INPUT]\n{asr_segment}"
        )
        messages = [{"role": "user", "content": instruction}]
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        text += prev_translation
        return text

    # ------------------------------------------------------------------ #
    #  Generación LLM: ahora devuelve List[str] con las N muestras        #
    # ------------------------------------------------------------------ #
    def _llm_generate(self, prompt: str) -> List[str]:
        """Genera N hipótesis mediante epsilon-sampling (vLLM local o remoto)."""
        if self.llm_client is not None:
            response = self.llm_client.completions.create(
                model=self._llm_model_name,
                prompt=prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                stop=["\n"],
                n=self.sampling_params.n,
                extra_body={"repetition_penalty": self._repetition_penalty},
            )
            return [c.text.replace("…", "") for c in response.choices]
        else:
            outputs = self.llm.generate(
                [prompt], sampling_params=self.sampling_params, use_tqdm=False
            )
            return [o.text.replace("…", "") for o in outputs[0].outputs]

    # ------------------------------------------------------------------ #
    #  Selección de hipótesis con MBRS                                     #
    # ------------------------------------------------------------------ #
    def _select_hypothesis(self, hypotheses: List[str], source: str) -> str:
        """
        Aplica el decoder MBRS configurado sobre el pool de hipótesis
        generadas por epsilon-sampling.

        Lógica de selección:
          - "none" o n=1         → primera hipótesis (comportamiento original).
          - "rambr_chrf"         → DecoderAggregateMBR con chrF (fast).
          - "prunembr_xcomet"    → DecoderPruningMBR con XCOMET-lite.
          - "rerank_kiwi"        → DecoderRerank con COMETkiwi (reference-free).
        """
        # Deduplicar preservando orden, descartar cadenas vacías
        # (mismo truco que dict.fromkeys del código original)
        seen: set = set()
        unique_hyps = [
            h
            for h in hypotheses
            if h.strip() and not (h in seen or seen.add(h))
        ]

        if not unique_hyps:
            return ""

        # Si solo hay una hipótesis o no hay decoder activo, devolvemos directamente
        if (
            len(unique_hyps) == 1
            or self.decoding_method == "none"
            or self.mbr_decoder is None
        ):
            return unique_hyps[0]

        if self.decoding_method == "rerank_kiwi":
            # Reference-free: DecoderRerank solo necesita (hipótesis, source).
            output = self.mbr_decoder.decode(unique_hyps, source=source, nbest=1)
        else:
            # MBR (rambr_chrf / prunembr_xcomet):
            # el mismo pool actúa como hipótesis Y pseudo-referencias.
            output = self.mbr_decoder.decode(
                unique_hyps, unique_hyps, source=source, nbest=1
            )

        return output.sentence[0]

    # ------------------------------------------------------------------ #
    #  Traducción de segmento (cambio principal: generate + select)        #
    # ------------------------------------------------------------------ #
    def _translate_segment(
        self, state: CascadeState, asr_segment: str, utt_finished: bool
    ) -> str:
        if asr_segment == "":
            return ""

        prefix = (
            ""
            if self.max_history_utterances == 0
            else self.target_sep.join(
                state.utt_targets[-self.max_history_utterances :]
            ) + self.target_sep
        )
        prev_translation = prefix + state.translations[-1]
        llm_context = ""
        llm_inputs = self._prepare_llm_inputs(asr_segment, prev_translation, llm_context)

        # 1) Genera N muestras con epsilon-sampling
        hypotheses = self._llm_generate(llm_inputs)
        # 2) Selecciona la mejor según el método MBRS configurado
        hypothesis = self._select_hypothesis(hypotheses, asr_segment)

        if utt_finished:
            state.utt_targets.append(state.translations[-1] + hypothesis)
            state.translations = [""]
            state.translation_hypotheses = [""]
            if self.target_lang not in ["Chinese", "Japanese"]:
                hypothesis = hypothesis.strip()
            return hypothesis

        full_hypothesis = state.translations[-1] + hypothesis
        state.translation_hypotheses.append(full_hypothesis)
        translation = self._mt_stable_prefix(
            state.translation_hypotheses[-2],
            state.translation_hypotheses[-1],
            state.translations[-1],
            len(asr_segment.split()),
        )
        translation_increment = translation[len(state.translations[-1]) :]
        state.translations.append(translation)

        if self.target_lang not in ["Chinese", "Japanese"]:
            translation_increment = translation_increment.strip()
        return translation_increment

    # ------------------------------------------------------------------ #
    #  Resto de métodos (sin cambios)                                      #
    # ------------------------------------------------------------------ #
    def _text_to_tokens(self, text: str) -> List[str]:
        if text == "":
            return []
        if self.latency_unit in ["word", "spm"]:
            return text.strip().split()
        if self.latency_unit == "char":
            return list(text.strip())
        raise NotImplementedError(f"Unsupported latency_unit: {self.latency_unit}")

    def _build_incremental_output(self, text: str) -> IncrementalOutput:
        if text == "":
            return IncrementalOutput([], "", [], "")

        out_text = text
        if (
            self.latency_unit == "word"
            and self._state.emission_started
            and not out_text.startswith(" ")
        ):
            out_text = " " + out_text
        self._state.emission_started = True

        return IncrementalOutput(
            new_tokens=self._text_to_tokens(text),
            new_string=out_text,
            deleted_tokens=[],
            deleted_string="",
        )

    @torch.inference_mode()
    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        if waveform is None or len(waveform) == 0:
            return IncrementalOutput([], "", [], "")

        self._state.source = np.concatenate(
            [self._state.source, np.asarray(waveform, dtype=np.float32)]
        )
        source_duration = len(self._state.source) / SAMPLE_RATE
        if source_duration < self.min_start_seconds:
            return IncrementalOutput([], "", [], "")

        asr_segment, utt_finished = self._transcribe_audio(self._state)
        if asr_segment is None:
            return IncrementalOutput([], "", [], "")
        translation = self._translate_segment(self._state, asr_segment, utt_finished)
        return self._build_incremental_output(translation)

    @torch.inference_mode()
    def end_of_stream(self) -> IncrementalOutput:
        translation = ""
        if len(self._state.source) > 0:
            asr_segment, utt_finished = self._transcribe_audio(self._state)
            if asr_segment is None:
                self._state.speech_id += 1
                return IncrementalOutput([], "", [], "")
            translation = self._translate_segment(self._state, asr_segment, utt_finished)

            if translation == "" and self._state.asr_hypotheses[-1].strip() != "":
                trailing_asr = self._state.asr_hypotheses[-1].strip()
                if self.max_history_utterances > 0:
                    trailing_asr = " ".join(
                        self._state.utt_sources[-self.max_history_utterances :]
                        + [trailing_asr]
                    )
                translation = self._translate_segment(self._state, trailing_asr, True)

        self._state.speech_id += 1
        return self._build_incremental_output(translation)

    def set_source_language(self, language: str) -> None:
        self.source_lang = language

    def set_target_language(self, language: str) -> None:
        self.target_lang = language
        self.target_sep = "" if language in ["Chinese", "Japanese"] else " "

    def tokens_to_string(self, tokens: List[str]) -> str:
        if self.latency_unit in ["word", "spm"]:
            return " ".join(tokens)
        if self.latency_unit == "char":
            return "".join(tokens)
        raise NotImplementedError(f"Unsupported latency_unit: {self.latency_unit}")

    def clear(self) -> None:
        self._state = CascadeState(speech_id=self._state.speech_id)