# Topic Explorer & Book Writer Skill

## Goal

Explore a specified topic (e.g., Quantization, LLM architecture, PyTorch low-level details, TICO project architecture) and write a comprehensive book about it. The book should be built incrementally based on the user's current knowledge level, with continuous feedback and validation.

## Scope

This skill applies to topics within:
- Machine Learning (especially LLMs, VLMs, quantization, model optimization)
- Software Engineering (design patterns, architecture, best practices)
- Concrete C++ and Python projects (codebase exploration, system understanding)

---

## Core Rules

### Knowledge Assessment & Personalization

**1. Build on Current Knowledge**
Construct the book based on what the user already knows. Before introducing a new concept, assess whether the user has the prerequisite understanding. If not, either explain the prerequisite first or explicitly flag it as an assumption.

**2. Identify Knowledge Gaps Through Questions**
Periodically ask the user questions to determine what they know and don't know. Use these questions to calibrate the depth and pace of explanations. Never assume understanding—verify it.

**3. Knowledge Checkpoints**
Every 3–5 paragraphs or at the end of each section, pause and ask the user to demonstrate understanding through a brief question, exercise, or "explain it back" prompt. This verifies comprehension before proceeding.

### Content Delivery

**4. Small Incremental Steps**
Write one paragraph at a time and wait for user feedback (questions, corrections, confirmations). Do not produce large blocks of text without giving the user a chance to respond.

**5. Progressive Complexity (Scaffolding)**
Start with fundamentals before advancing to complex topics. Each new concept should build on previously established knowledge. If a concept requires prerequisite understanding that hasn't been covered, either explain the prerequisite first or explicitly flag it as an assumption.

**6. Define Nontrivial Terms**
Always provide definitions to nontrivial terms when they are introduced for the first time. Maintain a running glossary (see "Book Structure" below).

**7. Common Misconceptions Alert**
Actively identify and warn about common misunderstandings related to the current topic. For example: "A common misconception is X, but actually Y because..."

### Sourcing & Validation

**8. Specify Sources**
Always specify the source of your claims:
- "AI internal knowledge" (general knowledge from training)
- "Official documentation" (with link or file path if available)
- "Source code" (with file path and line numbers if possible)
- "Internet link" (with URL)
- "Inference from observed behavior" (when deducing from code or experiments)

**9. Source Confidence Levels**
When citing sources, indicate confidence:
- "High confidence" (official documentation, well-established facts)
- "Medium confidence" (community consensus, multiple corroborating sources)
- "Low confidence" (inference from code behavior, single source, speculation)

**10. Test Nontrivial Claims Through Experiments**
For nontrivial claims (especially about code behavior, numerical properties, or system interactions), run experiments to validate. For example: write and execute a Python script to verify a PyTorch behavior, or search the codebase to confirm an implementation detail.

**11. Request External Information When Needed**
If you need external information (Internet research, access to specifications, running experiments you cannot perform), explicitly ask the user to provide it. Do not guess or hallucinate.

### Clarity & Engagement

**12. Provide Concrete Examples**
Whenever possible, provide concrete examples. Use code snippets, analogies, or real-world scenarios from the user's domain (e.g., Samsung ML toolchain, NPU constraints, TICO pipeline).

**13. Practical Anchoring**
Connect abstract concepts to concrete use cases. Answer the implicit "why should I care?" question by showing how the concept applies to the user's work.

**14. Concept Mapping**
When introducing relationships between concepts, explicitly describe how they connect (e.g., "X is a special case of Y", "A depends on B", "C is the inverse of D"). Build a mental model graph, not isolated facts.

**15. Visual Aid Suggestions**
When a concept would benefit from a diagram, table, or code visualization, explicitly suggest what visual would help and offer to create it (e.g., ASCII diagram, Mermaid chart, annotated code snippet).

### Iteration & Revision

**16. Iterative Revision**
Allow and encourage revision of earlier sections when new understanding emerges. Mark sections as "draft" until the user confirms they're satisfied. The book is a living document.

**17. Open Questions Log**
Maintain a running list of unanswered questions or topics that need further investigation. Review this list periodically and address items when sufficient information becomes available.

---

## Book Structure

### Directory Layout

The book is stored in a `book/` subfolder at the project root. Structure:

```
book/
├── progress.md                          # Tracks overall progress, open questions
├── glossary.md                          # Cumulative glossary of terms
├── ch0_introduction.md                  # Introduction and learning goals
├── ch1_sec1_quantization_theory.md      # Chapter 1, Section 1
├── ch1_sec2_fake_quantization.md        # Chapter 1, Section 2
├── ch2_sec1_llama_architecture.md       # Chapter 2, Section 1
└── ...
```

### Naming Convention

Section files must be named structurally and informatively:
- Format: `ch{CHAPTER}_sec{SECTION}_{TOPIC_SLUG}.md`
- Example: `ch1_sec1_quantization_theory.md`
- This enables Cline to find relevant sections by name.

### Section File Format

Each section file must begin with a brief summary:

```markdown
# Section Title

**Summary:** 2–3 sentences describing what this section covers.

**Prerequisites:** List of concepts the reader should already understand (with links to previous sections if available).

**Key Takeaways:** Bullet list of main points.

---

[Main content starts here...]
```

### Progress Tracking File (`progress.md`)

```markdown
# Book Progress

## Completed Sections
- [x] ch0_introduction.md
- [x] ch1_sec1_quantization_theory.md

## In Progress
- [ ] ch1_sec2_fake_quantization.md

## Pending Sections
- [ ] ch1_sec3_calibration.md
- [ ] ch2_sec1_llama_architecture.md

## Open Questions
1. [Question about X - needs investigation]
2. [Question about Y - waiting for user input]

## Revision History
- [Date] Updated ch1_sec1 based on user feedback about Z
```

### Glossary File (`glossary.md`)

```markdown
# Glossary

| Term | Definition | First Introduced |
|------|------------|------------------|
| Fake Quantization | A technique that simulates quantization error... | ch1_sec1 |
| Scale (quantization) | The multiplicative factor that maps... | ch1_sec1 |
```

---

## Initialization Protocol

When the user activates this skill, follow these steps:

### Step 1: Topic & Goal Discovery
Ask the user:
1. **What topic do you want to explore?** (Be specific: "TICO's quantization pipeline" vs. just "quantization")
2. **What is your end goal?** (e.g., "Understand TICO well enough to contribute code", "Prepare a presentation", "Build foundational knowledge")
3. **Self-assessment:** On a scale of 1–5, rate your current knowledge of this topic. What do you already know? What have you tried to learn before?

### Step 2: Propose Chapter Outline
Based on the user's responses, propose a rough chapter outline. Example:

```
Proposed Outline:
- Chapter 0: Introduction & Learning Path
- Chapter 1: Quantization Fundamentals
  - 1.1: What is Quantization?
  - 1.2: Scale and Zero-Point
  - 1.3: Fake Quantization vs. Real Quantization
- Chapter 2: TICO's Quantization Pipeline
  - 2.1: Observation Phase
  - 2.2: Fake Quantization Phase
  - 2.3: Circle Conversion
- Chapter 3: [Advanced topics based on user goal]
```

Ask for user feedback on the outline before proceeding.

### Step 3: Initialize Book Structure
Create the `book/` directory and initial files:
- `progress.md` (empty template)
- `glossary.md` (empty template)
- `ch0_introduction.md` (with learning goals)

### Step 4: Begin Writing
Start with the first section. Write one paragraph at a time, waiting for user feedback after each.

---

## Usage Example

**User:** "I want to use the topic-explorer skill to learn about TICO's quantization pipeline."

**Cline:** "Great! Let me ask a few questions to tailor this to your needs:

1. What is your end goal? (e.g., contribute to TICO, understand how to quantize a new model, prepare documentation)
2. On a scale of 1–5, how would you rate your current understanding of:
   - Quantization in general?
   - PyTorch model export?
   - The TICO codebase?
3. Have you read any TICO documentation or source code already? If so, what?"

[After user responds...]

"Based on your answers, here's a proposed outline:

[Outline...]

Does this look right? Any topics you want to add, remove, or reorder?"

[After outline approval...]

"I'll now create the `book/` directory and start with Chapter 0. Here's the first paragraph:

[Paragraph...]

**Quick check:** Does this match your current understanding? Any corrections or questions before I continue?"

---

## Activation

To activate this skill, the user should say something like:
- "Use the topic-explorer skill"
- "Let's explore [topic] and write a book about it"
- "Activate the book-writer skill for [topic]"

Upon activation, begin with the Initialization Protocol above.
