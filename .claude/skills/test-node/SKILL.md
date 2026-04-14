---
name: test-node
description: Use when creating tests for graph nodes, ingestion steps, extraction steps, or scoring logic.
allowed-tools: Read Write Edit Grep Glob Bash
---

# Test Node

Use this skill when:
- adding a graph node
- changing ingestion logic
- changing evidence extraction
- changing scoring or synthesis logic

## Required workflow
1. Identify the unit under test
2. Build fixture inputs
3. Assert typed output contract
4. Assert side effects where relevant
5. Cover one failure path
6. Keep tests deterministic

## Test expectations
- small tests first
- integration tests only where needed
- no flaky network dependence in core tests
- use frozen fixtures where possible

## Anti-patterns
- giant integration spaghetti
- tests that only assert no exception
- hidden fixture magic
- nondeterministic timestamps or randomness without control
