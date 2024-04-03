#!/usr/bin/env python3

import flask
import time

from flask import request

from IPython import embed  # For debugging; put `embed()` anywhere
from llama_cpp import Llama
import json
import sys

import numpy as np
import numpy.typing as npt

from llama_cpp._internals import _LlamaTokenDataArray

llm = Llama.from_pretrained(
    # repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
    # filename="*q8_0.gguf",
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="*Q8_0.gguf",
    logits_all=True,
    verbose=True,
)

# or "str".encode("utf-8")
# prompt = b"Q: Give a comma separated list of the planets in the solar system? A: "
prompt = (
    "Q: Give a comma separated list of the planets in the solar system? A: ".encode(
        "utf-8"
    )
)

tokens = list(llm.tokenize(prompt))

ctx_main = llm._ctx
n_vocab = ctx_main.model.n_vocab()


def generate_nth(tokens, nth):
    llm.reset()
    llm.eval(tokens)
    logits: npt.NDArray[np.single] = llm._scores[-1, :]
    # logits = llm._scores[-1, :]
    token_data_array = _LlamaTokenDataArray(n_vocab=n_vocab)
    token_data_array.copy_logits(logits)
    ctx_main.sample_softmax(token_data_array)
    nth_token = token_data_array.candidates_data[0][nth][0]
    return {
        "n": nth,
        "token": nth_token,
        "text": llm.detokenize([nth_token]).decode("utf-8"),
        "children": [],
    }


def generate_sequence(tokens, length):
    local_tokens = tokens.copy()
    output_tokens = []
    for n in range(length):
        token = generate_nth(local_tokens, 0)["token"]
        local_tokens.append(token)
        output_tokens.append(token)
    return output_tokens


def generate_tree(tokens, breadth: int, depth: int, continuation: int, path="n"):
    result = []
    for i in range(breadth):
        node_path = f"{path}_{i}"
        node = generate_nth(tokens, i)
        print(f"  {path} -> {node}")
        if depth == 0:
            node["continuation"] = generate_sequence(
                tokens + [node["token"]], continuation
            )
            node["continuation_text"] = llm.detokenize(node["continuation"]).decode(
                "utf-8"
            )
            print(f"  {path} -> {node} -> {node['continuation_text']}")
        else:
            node["children"] = generate_tree(
                tokens + [node["token"]], breadth, depth - 1, continuation, node_path
            )
        result.append(node)
    return result


def node_to_graphviz(node, nth, depth, path="n"):
    path = f"{path}_{nth}"
    graphviz = f"  {path} [label=\"{node['text']}\n[{node['token']}]\"];\n"
    for n, child in enumerate(node["children"]):
        child_path = f"{path}_{n}"
        graphviz += f"  {path} -> {child_path};\n"
        graphviz += node_to_graphviz(child, n, depth + 1, path)
    if "continuation_text" in node:
        graphviz += f"  {path} -> \"{node['continuation_text']}\";\n"
    return graphviz


def tree_to_graphviz(prefix_tokens, tree):
    graphviz = """
      digraph G {
        rankdir=LR;
        node [shape=rectangle];
    """
    prefix = llm.detokenize(prefix_tokens).decode("utf-8")
    for n, node in enumerate(tree):
        graphviz += f' "{prefix}" -> n_{n};\n'
        graphviz += node_to_graphviz(node, n, 0)
    graphviz += "}\n"
    return graphviz


# embed()

# selected_token = llm.sample


# tree = generate_tree(tokens, 3, 2, 5)
# print(tree)
# print(tree_to_graphviz(tokens, tree))

app = flask.Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return """
        <form method="post" action="/start">
            <label for="prompt">Prompt:</label>
            <br/>
            <textarea name="prompt" cols=80 rows=5>Q: Give a comma separated list of the planets in the solar system? A: </textarea>
            <br/>
            <label for="breadth">Breadth:</label>
            <input type="text" name="breadth" value="2" size=5 />
            <label for="depth">Depth:</label>
            <input type="text" name="depth" value="0" size=5 />
            <label for="continuation">Continuation:</label>
            <input type="text" name="continuation" value="3" size=5 />
            <input type="submit" value="Go!" />
        </form>
    """


@app.route("/start", methods=["POST"])
def start():
    breadth = int(request.form.get("breadth", 0))
    depth = int(request.form.get("depth", 0))
    continuation = int(request.form.get("continuation", 0))
    prompt = request.form.get(
        "prompt",
        "Q: Give a comma separated list of the planets in the solar system? A: ",
    )
    return f"""
      Breadth: {breadth}<br/>
      Depth: {depth}<br/>
      Continuation: {continuation}<br/>
      Prompt: {prompt}<br/>
    """


app.run(debug=True)