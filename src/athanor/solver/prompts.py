"""Solver prompts for the Agentic ARC solver.

All string prompts used to guide the model during solving, reflection,
and context compression are defined here.
"""

# ── In-flight context compression ────────────────────────────────────────────
# Triggered when the running conversation exceeds the token threshold.
# The model writes a memory checkpoint; the conversation is then replaced
# by this summary so solving can continue without hitting the hard limit.
#
# Framing note: intentionally *not* mentioned that "context limit is approaching"
# because urgency framing biases the model toward brevity, causing it to drop
# important details.  The model should feel free to write as long as needed.

CONTEXT_COMPRESSION_PROMPT = """Your conversation history is about to be reset to free up space. \
Write a thorough memory checkpoint so you can continue exactly where you left off.

After the reset, you will receive the original puzzle data (training/test grids), \
your last submitted code and hypothesis, test predictions, any reviewer feedback, \
and this memory checkpoint. Write as if briefing a fresh version of yourself — be exhaustive and specific. \
There is no length limit on your response; include anything that might be relevant.

Critical constraint:
- Previous `run_code` and `run_code_in_previous_runtime` transcripts do NOT survive reset.
- Any live exploratory Python runtime does NOT survive reset.
- If any exact code, helper function, variable name, error message, or key stdout/stderr output matters, copy it into this checkpoint explicitly.

### 1. Puzzle Understanding & Verified Invariants
- What you've observed about the transformation rule (confirmed facts and current hypothesis)
- Grid properties: dimensions, color palette, spatial structure, symmetries
- What stays the same vs. what changes between input and output
- Unusual aspects or edge cases you've noticed

### 2. Experiments & Learnings
For each significant exploratory code experiment (`run_code` or `run_code_in_previous_runtime`):
- What you tested and why
- What you learned, confirmed, or ruled out
- Key outputs, measurements, or patterns (include code snippets if helpful)

### 3. Hypothesis Evolution
Your last submitted hypothesis will be shown separately in the next context window — do not restate it here. Instead focus on:
- Failed approaches and the specific reason each failed
- What is wrong or incomplete about the current hypothesis, and how you plan to fix it
- Confidence level (1–5) and what would raise or lower it

### 4. Verified Invariants
Facts confirmed across ALL training examples:
- (e.g., "All outputs are exactly 20×20")
- (e.g., "Background color 8 is always preserved")

### 5. Current State & Next Steps
- Where you are in the solving process
- What you were about to try next
- Any edge cases you haven't yet resolved
- Implementation details or code that would help you continue

### 6. Other Notes
Anything that doesn't fit the above but might be important to remember.
"""


# ── Post-solve reflection: training-pass sanity check ────────────────────────
# Triggered when the model achieves 100 % accuracy on the training set.
# Asks the model to self-audit before accepting its solution.

TEST_GENERALIZATION_REFLECTION_PROMPT = """\
Accuracy achieved 100% on **training set**. Now do a **test-set** generalization sanity check.

### What to check
1) **Assumption audit (generalization)**
   - Did your logic accidentally rely on any training-only coincidences (exact grid sizes, \
absolute coordinates, specific counts, or hard-coded colors without semantic role)?
   - If you used any constants, explain why they are *semantic* (role-based) rather than incidental.

2) **Prediction plausibility**
   - Do the predicted test outputs look consistent with the same rule that explains *all* training pairs?
   - Call out any artifacts that suggest overfitting (extra pixels, missing structures, wrong recolors, \
wrong alignment).

### Output format (strict)
- `CONFIDENCE: <1-5>` (how much you trust your `solve` function implements the correct rule)
- `DECISION: ACCEPT` or `DECISION: RETRY`
- `REASONS:` (explanation of your decision and any concerns)

Do NOT call any tools in this reflection.

Think thoroughly--don't rush to give a final answer.
"""


# ── Post-solve reflection: training failure ───────────────────────────────────
# Triggered when execute_python_solution reports failures on training examples.
# After reflection, CONTEXT_COMPRESSION_PROMPT is used to compress the session.

TRAIN_FAILURE_REFLECTION_PROMPT = """\
The solution failed on some training examples. \
Review the execute_python_solution output above and reflect on this attempt by addressing the following.

### 1. Grid Observations
Summarize your observations for the input/output grids in each example:
- What shapes, patterns, or structures do you see?
- What color changes or rigid transforms occur?
- What commonalities exist across examples?

### 2. Code Execution Insights
If you previously called exploratory code tools, what did you learn or confirm from those experiments?
(Recall any key outputs that informed your understanding)

### 3. Diff Analysis
Look at the execute_python_solution output above. For each failed training example:
- What's the difference between your predicted output and the ground truth?
- Where specifically did your logic go wrong?

### 4. Failure Root Cause
Before planning next steps, identify WHY your approach failed:
- Was it a **wrong assumption** about the transformation rule?
- Was it a **coding bug** (logic correct, implementation wrong)?
- Was it an **edge case** you didn't handle?
- What **specific insight** were you missing?

Be specific: "I assumed X, but actually Y" is better than "my code was wrong."

### 5. Verified Rules
List any **invariant observations** that are TRUE regardless of your current hypothesis.
These are facts about the puzzle that you've confirmed and should remember. For example,
- Output dimensions (e.g., "output is always 20x20")
- Color palette constraints (e.g., "only colors 0 and 3 appear")
- Structural invariants (e.g., "output always has a diagonal pattern")
- And more

Format each as a single bullet point. Only include observations you're confident about.

### 6. Next Steps
- What hypothesis refinements do you want to try?
- What experiments might help clarify the transformation rule?

Think deeply and answer these questions verbally.
Do NOT call any tools in this reflection - just analyze and verbalize your thoughts.

Take your time to reflect on the current failure. Think thoroughly--don't rush to give a final answer.
"""

BEST_EFFORT_PROMPT = """\
⚡ Strategy shift — training accuracy constraint lifted.

You've used most of your turn budget without achieving 100% training accuracy.
This sometimes happens when the rule is unusually subtle or when the training
examples aren't fully self-consistent.

The constraint is now removed — your next execute_python_solution call will be
accepted regardless of training score.

Refocus on maximizing the chance of getting the test examples right. You may
submit up to 2 candidate outputs per test example.

When ready, call execute_python_solution.
"""


def build_candidate_expansion_guidance_prompt(
    *,
    reflector_response: str,
    bypass_compression: bool = False,
) -> str:
    """Build the post-reflector guidance prompt for EXPAND_CANDIDATES.

    Quotes the reflector's full response verbatim so the solver agent
    receives the complete reasoning chain (Deliberation, FINDINGS, CONCERNS,
    and all structured sections) without information loss.

    When bypass_compression=True (small context), the solver keeps full tool
    access and can act immediately. When False (compression mode), the solver
    writes a memory checkpoint first, then gets tools in the next iteration.
    """
    if bypass_compression:
        closing = (
            "Critically evaluate the reviewer's analysis. "
            "The reviewer works from artifacts only and can make mistakes. "
            "Decide which concerns are valid, then revise your hypothesis and "
            "submit an updated solve() implementation that emits the additional candidate."
        )
    else:
        closing = (
            "Critically evaluate the reviewer's analysis as you write your checkpoint. "
            "The reviewer works from artifacts only and can make mistakes. "
            "Note which concerns seem valid and how you plan to address the ambiguity "
            "— you will have full tool access in the next iteration to implement "
            "the additional candidate."
        )
    return f"""\
The independent reflector reviewed your solution and identified a potential ambiguity in the test output. It recommends emitting an additional candidate to hedge against the alternative interpretation. Here is the reflector's full analysis:

---

{reflector_response}

---

{closing}
"""
