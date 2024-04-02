#!/usr/bin/env python3

from IPython import embed  # For debugging; put `embed()` anywhere
from llama_cpp import Llama
import json
import sys

llm = Llama.from_pretrained(
    # repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
    # filename="*q8_0.gguf",
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="*Q8_0.gguf",
    logits_all=True,
    verbose=True,
)

prompt = "Q: Give a comma separated list of the planets in the solar system? A: "

# Let's do a simple stream test
print("Initializing completion")
output_stream = llm.create_completion(prompt, max_tokens=32, stop=["Q:","\n"], stream=True)
print("Streaming output:")
for output in output_stream:
    print(output["choices"][0]["text"], end="")
    sys.stdout.flush()
# output_stream = llm.create_completion(prompt, max_tokens=32, echo=True, stream=True, logprobs=5)


# tokens = llm.tokenize(prompt)
# for token in llm.generate(tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.1):
#     print(llm.detokenize([token]))

