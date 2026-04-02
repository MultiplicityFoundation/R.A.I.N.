# R.A.I.N. Lab README Repositioning Design

Date: 2026-04-01

## Goal

Rewrite the repository README so it matches the new public product story:

- R.A.I.N. Lab is the product
- James is the assistant inside the product
- The primary audience is users, not contributors
- The README should explain the research-panel experience first and the runtime/developer story second

The README should become more YC-readable by making the user, problem, and outcome legible immediately instead of leading with infrastructure language.

## Core Positioning

Primary product line:

- `A private-by-default expert panel in a box for researchers, independent thinkers, and R&D teams.`

Primary product action:

- Ask a raw research question
- Receive multiple expert perspectives
- Ground strong claims in papers or explicit evidence
- Leave with the strongest explanations, disagreements, and next moves

Product naming:

- Lead with `R.A.I.N. Lab` as the product
- Treat `James` as the assistant inside R.A.I.N. Lab

## Audience

The README should explicitly name the intended users near the top:

- Researchers
- Independent thinkers
- R&D teams

This is better than trying to sound universal. The README should reduce ambiguity fast.

## Recommended README Strategy

Recommended approach:

- Product-first README
- User-facing opening
- Public web experience first
- Local/private CLI path second
- Developer and architecture details lower in the file

Rejected alternatives:

1. Hybrid README
   - Better than the current file, but still too easy to dilute the product story

2. Developer-first rewrite
   - Preserves the current repo shape, but fails to fix the main positioning problem

3. Generic research-AI framing
   - Too broad and too crowded
   - Loses the specific differentiation of an expert panel in a box

## README Structure

### 1. Title and One-Line Promise

Top of file should present:

- `# R.A.I.N. Lab`
- `A private-by-default expert panel in a box for researchers, independent thinkers, and R&D teams.`

This should replace the current coding-agent-runtime line as the headline story.

### 2. What It Does

Add one short paragraph explaining:

- the user starts with a raw research question
- R.A.I.N. Lab assembles multiple expert perspectives
- strong claims are tied to papers or explicit evidence
- the output includes the strongest explanations, disagreements, and next moves

This section should make the product legible in under a minute.

### 3. Sharp Contrast

Include the core contrast line:

- `Most tools help you find papers. R.A.I.N. Lab helps you think with a room full of experts.`

This should frame the category difference without attacking literature-search products.

### 4. Try It Now

The first action path should lead with the public research-panel experience.

Recommended order:

1. Public web experience
2. Local/private CLI experience

The README should not lead with the CLI anymore.

### 5. Why It Is Different

Explain the main advantages in plain language:

- Different perspectives, not one flat answer
- Claims tied to papers or explicit evidence
- Private by default
- Synthesis you can act on

This can stay in a short value table or a short bullet list.

### 6. Who It Is For

Keep a compact section that names:

- Researchers
- Independent thinkers
- R&D teams

This should support the top-level promise, not introduce a new audience.

### 7. What You Can Do

Describe practical outcomes such as:

- Pressure-test a research question
- Surface competing explanations
- See where expert perspectives disagree
- Leave with concrete next reads, tests, or decisions

This should read like user outcomes, not feature inventory.

### 8. Local and Private Workflow

Introduce James here:

- James is the assistant inside R.A.I.N. Lab
- James helps run the panel, guide local workflows, and carry questions through private sessions

The existing local runtime and setup details can live here, but lower in the document.

### 9. Developer Section

Keep the contributor setup and architecture material, but move it behind the product story.

This section should preserve:

- local setup steps
- Python and Rust implementation details
- testing commands
- architecture references

The developer section should remain available without defining the product in the first screenful.

## Messaging Stack

### Top

- `# R.A.I.N. Lab`
- `A private-by-default expert panel in a box for researchers, independent thinkers, and R&D teams.`

### Product Summary

- `Ask a raw research question. R.A.I.N. Lab assembles multiple expert perspectives, grounds strong claims in papers or explicit evidence, and returns the strongest explanations, disagreements, and next moves.`

### Category Contrast

- `Most tools help you find papers. R.A.I.N. Lab helps you think with a room full of experts.`

### James Framing

- `James is the assistant inside R.A.I.N. Lab who helps run the panel, guide the workflow, and carry the question through local sessions.`

### Differentiators

- `Different perspectives, not one flat answer`
- `Claims tied to papers or explicit evidence`
- `Private by default`
- `Synthesis you can act on`

## What To Remove From The Top

The top of the README should stop leading with:

- `local-first autonomous coding agent runtime`
- Rust/Python/hardware-adjacent teams as the primary buyer/user
- implementation-stack framing before product framing
- infrastructure-first language that makes the product sound like internal tooling

These details can still exist lower in the file for contributors.

## Success Criteria

The README rewrite succeeds if:

1. A new visitor can identify the user in the first screenful
2. A new visitor can identify the product action in the first screenful
3. The README matches the public homepage positioning
4. The product sounds like a user-facing research tool before it sounds like a runtime
5. Contributor information remains present without dominating the story

## Tone

Primary tone:

- Clear
- Serious
- Scientific
- Product-focused

Avoid:

- overly abstract AI language
- runtime-first framing
- generic "AI for research" vagueness
- founder-poetry or manifesto tone
