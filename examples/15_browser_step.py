"""One step of a browser agent: Beveridge-Pro-C-Web picks the next operation and the element it applies to.

    pip install "bev-decisions[pro]"       # one GPU with about 60 GB
    python examples/15_browser_step.py

Describe the page (URL, title, visible text, numbered elements, recent actions) and the goal; ask which operation comes next
and, for each operation that needs one, which element. Your agent then performs the action and asks again on the new page.
Beveridge picks the step; the words to type come from the goal or from a text model you choose. Unsure answers (below 0.8)
stop the agent so a person can take over. Beveridge can make mistakes: keep a person in the loop for anything that buys,
sends, deletes or signs.
"""

import bev

model = bev.load(bev.WEB_MODEL)

page = {
    "recent_actions": [],
    "elements": ["[1] input: Destination (empty)", "[2] button: Find stays", "[3] checkbox: Free cancellation (not checked)"],
    "page": {"url": "https://example-travel.test/", "title": "Find a stay", "text": "Where are you going? ... Find stays"},
}
goal = "goal: Find stays in Lisbon with free cancellation."
elements = {"1": "[1] Destination (empty)", "2": "[2] Find stays", "3": "[3] Free cancellation (not checked)"}

answers = model.decide(page, {
    "operation": {"type": "choice", "instructions": goal + "\nrules: Pick the next single operation on the current page.",
                  "criteria": {"CLICK": "Click an element", "TYPE_TEXT": "Enter text in a field",
                               "WAIT": "Wait for the page to finish loading", "DONE": "Every requirement is visibly satisfied."}},
    "type_text_target": {"type": "choice", "instructions": goal + "\noperation: TYPE_TEXT", "criteria": {"1": elements["1"]}},
    "click_target": {"type": "choice", "instructions": goal + "\noperation: CLICK",
                     "criteria": {"2": elements["2"], "3": elements["3"]}},
})

sure, review = bev.gate(answers, threshold=0.8)
op = answers["operation"]["choice"]
if "operation" in review:
    print("unsure about the next step: ask a person", answers["operation"]["probabilities"])
elif op in ("CLICK", "TYPE_TEXT") and op.lower() + "_target" in review:
    print(f"{op}, but unsure on which element: ask a person", answers[op.lower() + "_target"]["probabilities"])
elif op in ("CLICK", "TYPE_TEXT"):
    target = answers[op.lower() + "_target"]
    print(f"{op} on {elements[target['choice']]} (operation {answers['operation']['confidence']:.2f}, element {target['confidence']:.2f})")
else:
    print(op, round(answers["operation"]["confidence"], 2))
