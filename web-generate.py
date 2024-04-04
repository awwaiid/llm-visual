#!/usr/bin/env python3

import flask
import time

from flask import request

from IPython import embed  # For debugging; put `embed()` anywhere
from llama_cpp import Llama
import json
import sys

import threading

import numpy as np
import numpy.typing as npt

from llama_cpp._internals import _LlamaTokenDataArray

import textwrap
import html

def debug(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


class TokenTree:
    def __init__(self, prompt, breadth, depth, continuation):
        self.state = "init"
        self.prompt = prompt
        self.breadth = breadth
        self.depth = depth
        self.continuation = continuation
        self.tree = []
        self.llm = None
        self.ctx_main = None
        self.n_vocab = None
        self.prompt_tokens = None

    def run(self):
        self.llm = Llama.from_pretrained(
            repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            filename="*Q8_0.gguf",
            verbose=False,
        )
        self.ctx_main = self.llm._ctx
        self.n_vocab = self.ctx_main.model.n_vocab()

        self.prompt_tokens = list(self.llm.tokenize(self.prompt.encode("utf-8")))

        self.llm.reset()
        self.llm.eval(self.prompt_tokens)  # Seed the whole thing
        self.state = "running"
        self.generate_tree(
            self.prompt_tokens,
            self.breadth,
            self.depth,
            self.continuation,
            "n",
            self.tree,
        )
        self.state = "done"

    def stop(self):
        self.state = "stopped"

    def generate_nth(self, tokens, nth):
        if self.state == "stopped":
            raise Exception('Aborting due to stop')
        self.llm.n_tokens = len(tokens) - 1
        self.llm.eval([tokens[-1]])
        logits = self.llm._scores[-1, :]
        token_data_array = _LlamaTokenDataArray(n_vocab=self.n_vocab)
        token_data_array.copy_logits(logits)
        self.ctx_main.sample_softmax(token_data_array)
        nth_token = token_data_array.candidates_data[0][nth][0]
        return {
            "n": nth,
            "prefix_tokens": tokens,
            "token": nth_token,
            "text": self.llm.detokenize([nth_token]).decode("utf-8", "backslashreplace"),
            "children": [],
        }

    def generate_sequence(self, tokens, length):
        local_tokens = tokens.copy()
        output_tokens = []
        for n in range(length):
            token = self.generate_nth(local_tokens, 0)["token"]
            local_tokens.append(token)
            output_tokens.append(token)
        return output_tokens

    def generate_tree(
        self, tokens, breadth: int, depth: int, continuation: int, path, result
    ):
        for i in range(breadth):
            node_path = f"{path}_{i}"
            node = self.generate_nth(tokens, i)
            result.append(node)
            debug(f"  {path} -> {node}")
            if depth == 0:
                node["continuation"] = self.generate_sequence(
                    tokens + [node["token"]], continuation
                )
                node["continuation_text"] = self.llm.detokenize(
                    node["continuation"]
                ).decode("utf-8", "backslashreplace")
                debug(f"  {path} -> {node} -> {node['continuation_text']}")
            else:
                node["children"] = []
                # node["children"] =
                self.generate_tree(
                    tokens + [node["token"]],
                    breadth,
                    depth - 1,
                    continuation,
                    node_path,
                    node["children"],
                )
        return result

    def node_to_graphviz(self, node, nth, depth, path="n"):
        path = f"{path}_{nth}"
        node_text = node["text"].replace('"', '\\"').replace("\\", "\\\\")
        graphviz = f"  {path} [label=\"{node_text}\n[{node['token']}]\"];\n"
        for n, child in enumerate(node["children"]):
            child_path = f"{path}_{n}"
            graphviz += f"  {path} -> {child_path};\n"
            graphviz += self.node_to_graphviz(child, n, depth + 1, path)
        if "continuation_text" in node:
            graphviz += f"  {path} -> {path}_continuation;\n"
            node_continuation_text = (
                node["continuation_text"].replace('"', '\\"').replace("\\", "\\\\")
            )
            node_continuation_text = textwrap.fill(node_continuation_text, 40)
            graphviz += f'  {path}_continuation [label="{node_continuation_text}"];\n'

            all_tokens = node["prefix_tokens"] + [node["token"]] + node["continuation"]
            all_text = self.llm.detokenize(all_tokens).decode("utf-8", "backslashreplace").replace('"', '\\"').replace("\\", "\\\\")
            all_text = textwrap.fill(all_text, 40)
            graphviz += f'  {path}_continuation -> {path}_all [style=dashed];\n'
            graphviz += f'  {path}_all [label="{all_text}" style=dashed];\n'

        return graphviz

    def tree_to_graphviz(self):
        graphviz = """
          digraph G {
            rankdir=LR;
            node [shape=rectangle];
        """
        prefix = self.llm.detokenize(self.prompt_tokens).decode("utf-8")
        prefix = prefix.replace('"', '\\"').replace("\\", "\\\\")
        prefix = textwrap.fill(prefix, 40)
        for n, node in enumerate(self.tree):
            graphviz += f' "{prefix}" -> n_{n};\n'
            graphviz += self.node_to_graphviz(node, n, 0)
        graphviz += "}\n"
        return graphviz


app = flask.Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    if global_tree is not None:
        global_tree.stop()
    breadth = int(request.args.get("breadth", 2))
    depth = int(request.args.get("depth", 1))
    continuation = int(request.args.get("continuation", 10))
    prompt = html.escape(request.args.get(
        "prompt",
        "Q: Give a comma separated list of the planets in the solar system? A: ",
    ))
    return f"""
        <form method="post" action="/start">
            <label for="prompt">Prompt:</label>
            <br/>
            <textarea name="prompt" cols=80 rows=5>{prompt}</textarea>
            <br/>
            <label for="breadth">Breadth:</label>
            <input type="text" name="breadth" value="{breadth}" size=5 />
            <label for="depth">Depth:</label>
            <input type="text" name="depth" value="{depth}" size=5 />
            <label for="continuation">Continuation:</label>
            <input type="text" name="continuation" value="{continuation}" size=5 />
            <input type="submit" value="Go!" />
        </form>
    """


global_tree = None


def build_tree(prompt, breadth, depth, continuation):
    global global_tree
    global_tree = TokenTree(prompt, breadth, depth - 1, continuation)
    global_tree.run()


@app.route("/start", methods=["POST"])
def start():
    breadth = int(request.form.get("breadth", 1))
    depth = int(request.form.get("depth", 1))
    continuation = int(request.form.get("continuation", 10))
    prompt = request.form.get(
        "prompt",
        "Q: Give a comma separated list of the planets in the solar system? A: ",
    )

    if global_tree is not None:
        global_tree.stop()

    threading.Thread(
        target=build_tree, args=(prompt, breadth, depth, continuation)
    ).start()

    return (
        f"""
        Breadth: {breadth}<br/>
        Depth: {depth}<br/>
        Continuation: {continuation}<br/>
        Prompt: {prompt}<br/>
        State: <span id="state">loading</span><br/>
        <button id="stop">Stop</button>
        <form method="get" action="/">
            <input type="submit" value="Restart" />
            <input type="hidden" name="breadth" value="{breadth}" />
            <input type="hidden" name="depth" value="{depth}" />
            <input type="hidden" name="continuation" value="{continuation}" />
            <input type="hidden" name="prompt" value="{prompt}" />
        </form>
        """
        + """
        <div id="graphviz"></div>

        <script src="https://unpkg.com/@viz-js/viz"></script>

        <script>
            function renderGraphviz(dot) {
                Viz.instance().then(viz => {
                  document.getElementById("graphviz").innerHTML = ""
                  document.getElementById("graphviz").appendChild(viz.renderSVGElement(dot));
                });
            }

            function stop() {
                fetch("/stop")
                    .then(response => response.text())
                    .then(data => {
                        console.log(data);
                    });
            }


            let interval = setInterval(() => {
                fetch("/current")
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById("state").innerText = data.state;
                        if(data.state == "running") {
                            renderGraphviz(data.dot);
                        } else if(data.state == "done") {
                            renderGraphviz(data.dot);
                            clearInterval(interval);
                        }
                    });
            }, 1000);
            document.getElementById("stop").addEventListener("click", () => {
                clearInterval(interval);
                stop();
            });
        </script>
        <style>
            #graphviz svg {
                width: 90vw;
                overflow: visible;
            }
            #graphviz svg .node polygon {
              fill: white;
              filter: drop-shadow(2px 2px 4px rgba(0, 0, 0, 0.5));
            }
        </style>
    """
    )

@app.route("/stop", methods=["GET"])
def stop():
    global global_tree
    if global_tree is not None:
        global_tree.stop()
    return "stopped"

@app.route("/current", methods=["GET"])
def current():
    global global_tree
    if global_tree is None:
        return {
            "state": "loading",
            "dot": """digraph G { "loading" }"""
        }
    elif global_tree.state == "init":
        return {
            "state": "init",
            "dot": """digraph G { "initializing" }"""
        }
    elif global_tree.state == "running":
        return {
            "state": "running",
            "dot": global_tree.tree_to_graphviz()
        }
    elif global_tree.state == "done":
        return {
            "state": "done",
            "dot": global_tree.tree_to_graphviz()
        }
    else:
        return {
            "state": "error",
            "dot": """digraph G { "error" }"""
        }


app.run(debug=True)
