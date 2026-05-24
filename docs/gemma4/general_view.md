# Gemma 4 model overview

[Video](https://www.youtube.com/watch?v=jZVBoFOJK-Q)

Gemma is a family of generative artificial intelligence models and you can
use them in a wide variety of generation tasks, including question answering,
summarization, and reasoning. Gemma models are provided with open weights and
permit responsible
[commercial use](https://ai.google.dev/gemma/terms),
allowing you to tune and deploy them in your own projects and applications.

Gemma 4 model family spans three distinct architectures tailored for specific
hardware requirements:

- **Small Sizes:** 2B and 4B effective parameter models built for ultra-mobile, edge, and browser deployment (e.g., Pixel, Chrome).
- **Dense:** A powerful 31B parameter dense model that bridges the gap between server-grade performance and local execution.
- **Mixture-of-Experts:** A highly efficient 26B MoE model designed for high-throughput, advanced reasoning.

You can download Gemma 4 models from
[Kaggle](https://www.kaggle.com/models?query=gemma-4&publisher=google) and
[Hugging Face](https://huggingface.co/collections/google/gemma-4).
For more technical details on Gemma 4, see the
[Model Card](https://ai.google.dev/gemma/docs/core/model_card_4).
Earlier versions of Gemma core models are also available for download. For more
information, see [Previous Gemma models](https://ai.google.dev/gemma/docs/core#previous-models).

[Get it on Kaggle](https://www.kaggle.com/models?query=gemma-4&publisher=google)
[Get it on Hugging Face](https://huggingface.co/collections/google/gemma-4)

## Capabilities

- **Reasoning:** All models in the family are designed as highly capable reasoners, with configurable [thinking
  modes](https://ai.google.dev/gemma/docs/capabilities/thinking).
- **Extended Multimodalities:** Processes Text, [Image](https://ai.google.dev/gemma/docs/capabilities/vision/image) with variable aspect ratio and resolution support (all models), [Video](https://ai.google.dev/gemma/docs/capabilities/vision/video), and [Audio](https://ai.google.dev/gemma/docs/capabilities/audio) (featured natively on the E2B and E4B models).
- **Increased Context Window:** Small models feature a 128K context window, while the medium models support 256K.
- **Enhanced Coding \& Agentic Capabilities:** Achieves notable improvements in coding benchmarks alongside built-in [function-calling
  support](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4), powering highly capable autonomous agents.
- **Native System Prompt Support:** Gemma 4 introduces built-in support for the system role, enabling more structured and controllable conversations.
- **[Multi-Token Prediction](https://ai.google.dev/gemma/docs/mtp/overview):** All Gemma 4 models (E2B, E4B, 31B, and 26B A4B) include a dedicated draft model for speculative decoding, enabling significantly faster inference with no quality loss.

## Parameter sizes and quantization

Gemma 4 models are available in 4 parameter sizes: E2B, E4B, 31B and 26B A4B.
The models can be used with their default precision (16-bit) or with a lower
precision using quantization. The different sizes and precisions represent a set
of trade-offs for your AI application. Models with higher parameters and bit
counts (higher precision) are generally more capable, but are more expensive to
run in terms of processing cycles, memory cost and power consumption. Models
with lower parameters and bit counts (lower precision) have less capabilities,
but may be sufficient for your AI task.

### Gemma 4 Inference Memory Requirements

The following table details the approximate GPU or TPU memory requirements for
running inference with each size of the Gemma 4 model versions.

> [!NOTE]
> **Note:** These numbers may change based on your specific inference tool and environment.

| Parameters | BF16 (16-bit) | SFP8 (8-bit) | Q4_0 (4-bit) |
|---|---|---|---|
| Gemma 4 E2B | 9.6 GB | 4.6 GB | 3.2 GB |
| Gemma 4 E4B | 15 GB | 7.5 GB | 5 GB |
| Gemma 4 31B | 58.3 GB | 30.4 GB | 17.4 GB |
| Gemma 4 26B A4B | 48 GB | 25 GB | 15.6 GB |

**Table 1.** Approximate GPU or TPU memory required to load Gemma 4 models based
on parameter count and quantization level.

### Key Considerations for Memory Planning

- **Efficient Architecture (E2B and E4B):** The "E" stands for "effective" parameters. The smaller models incorporate Per-Layer Embeddings (PLE) to maximize parameter efficiency in on-device deployments. Rather than adding more layers to the model, PLE gives each decoder layer its own small embedding for every token. These embedding tables are large but only used for quick lookups, which is why the total memory required to load static weights is higher than the effective parameter count suggests.
- **The MoE Architecture (26B A4B):** The 26B is a Mixture of Experts model. While it only activates 4 billion parameters per token during generation, **all 26 billion parameters** must be loaded into memory to maintain fast routing and inference speeds. This is why its baseline memory requirement is much closer to a dense 26B model than a 4B model.
- **Base Weights Only:** The estimates in the preceding table *only* account for the memory required to load the static model weights. They don't include the additional VRAM needed for supporting software or the context window.
- **Context Window (KV Cache):** Memory consumption will increase dynamically based on the total number of tokens in your prompt and the generated response. Larger context windows require significantly more VRAM on top of the base model weights.
- **Fine-Tuning Overhead:** Memory requirements for *fine-tuning* Gemma models are drastically higher than for standard inference. Your exact footprint will depend heavily on the development framework, batch size, and whether you are using full-precision tuning versus a Parameter-Efficient Fine-Tuning (PEFT) method like Low-Rank Adaptation (LoRA).

## Previous Gemma models

You can work with previous generations of Gemma models, which are also
available from [Kaggle](https://www.kaggle.com/models?query=gemma) and
[Hugging Face](https://huggingface.co/google/collections).
For more technical details about previous Gemma models, see the following
model card pages:

- Gemma 3 [Model Card](https://ai.google.dev/gemma/docs/core/model_card_3)
- Gemma 2 [Model Card](https://ai.google.dev/gemma/docs/core/model_card_2)
- Gemma 1 [Model Card](https://ai.google.dev/gemma/docs/core/model_card)

Ready to start building?
[Get started](https://ai.google.dev/gemma/docs/get_started)
with Gemma models!