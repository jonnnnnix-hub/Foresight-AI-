---
name: scaffold-graph
description: Use when creating or refactoring LangGraph nodes, parent graphs, subgraphs, and typed state models.
allowed-tools: Read Write Edit Grep Glob Bash
---

# Scaffold Graph

Use this skill when:
- adding a new graph
- refactoring orchestration
- splitting parent graph and subgraphs
- adding or updating typed state models
- wiring new nodes into the FlowEdge workflow

## Goals
- keep orchestration clear
- keep state explicit
- keep subgraph boundaries clean
- avoid giant God-node files

## Required workflow
1. Inspect current graph structure under `src/.../graphs/`
2. Create or update typed state models first
3. Add nodes as isolated modules
4. Keep parent graph orchestration thin
5. Prefer explicit input and output contracts
6. Add or update tests for graph behavior

## Checklist
- state schemas exist
- node interfaces are typed
- subgraph boundaries are clear
- imports are clean
- tests cover happy path and failure path
- graph code does not mix business logic with provider plumbing

## Anti-patterns
- untyped dict soup
- giant orchestration files
- hidden state mutation
- unclear node contracts
- graph nodes doing unrelated work
