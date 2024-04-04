#!/usr/bin/env python3

from IPython import embed  # For debugging; put `embed()` anywhere
from llama_cpp import Llama
import sys

from llama_cpp._internals import _LlamaTokenDataArray


def debug(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


llm = Llama.from_pretrained(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="*Q8_0.gguf",
    verbose=False,
    # logits_all=True,
)

prompt = (
    "Q: Give a comma separated list of the planets in the solar system? A: ".encode(
        "utf-8"
    )
)

tokens = list(llm.tokenize(prompt))

ctx_main = llm._ctx
n_vocab = ctx_main.model.n_vocab()


def generate_nth(tokens, nth):
    llm.n_tokens = len(tokens) - 1
    llm.eval([tokens[-1]])
    logits = llm._scores[-1, :]
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
        debug(f"  {path} -> {node}")
        if depth == 0:
            node["continuation"] = generate_sequence(
                tokens + [node["token"]], continuation
            )
            node["continuation_text"] = llm.detokenize(node["continuation"]).decode(
                "utf-8"
            )
            debug(f"  {path} -> {node} -> {node['continuation_text']}")
        else:
            node["children"] = generate_tree(
                tokens + [node["token"]], breadth, depth - 1, continuation, node_path
            )
        result.append(node)
    return result


llm.eval(tokens)  # Seed the whole thing
tree = generate_tree(tokens, 2, 2, 5)
debug(tree)


def node_to_graphviz(node, nth, depth, path="n"):
    path = f"{path}_{nth}"
    graphviz = f"  {path} [label=\"{node['text']}\n[{node['token']}]\"];\n"
    for n, child in enumerate(node["children"]):
        child_path = f"{path}_{n}"
        graphviz += f"  {path} -> {child_path};\n"
        graphviz += node_to_graphviz(child, n, depth + 1, path)
    if "continuation_text" in node:
        graphviz += f"  {path} -> {path}_continuation;\n"
        graphviz += f"  {path}_continuation [label=\"{node['continuation_text']}\"];\n"
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


print(tree_to_graphviz(tokens, tree))
