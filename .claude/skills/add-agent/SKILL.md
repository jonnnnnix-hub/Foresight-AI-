---
name: add-agent
description: Use when adding a new specialist agent, including prompt file, implementation, schema, and graph wiring.
allowed-tools: Read Write Edit Grep Glob Bash
---

# Add Agent

Use this skill when:
- adding a new specialist agent
- refactoring an agent into a clearer unit
- wiring a new agent into scoring or debate flows

## Required workflow
1. Create or update the agent instruction file
2. Create the typed output schema
3. Create the implementation module
4. Register the agent in graph orchestration
5. Add persistence for the agent output
6. Add tests ensuring output schema validity

## Rules
- no untyped agent outputs
- no vague agent purpose
- every agent must map to scoring or synthesis value
- every nontrivial claim must be evidence-backed
- avoid overlapping agent roles unless there is a reason

## Deliverables
- prompt or instruction
- schema
- implementation
- wiring
- tests
