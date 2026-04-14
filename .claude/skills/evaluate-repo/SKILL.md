---
name: evaluate-repo
description: Use when analyzing a GitHub repo for research strength, execution realism, ML readiness, productization value, reliability, and scalp-fit.
allowed-tools: Read Grep Glob Bash
---

# Evaluate Repo

Use this skill when:
- inspecting a repo for FlowEdge adoption
- comparing multiple trading or quant repos
- preparing evidence for debate and scoring

## What to inspect
- README and docs
- package manifests
- source tree
- tests
- examples
- config files
- release and issue signals if available

## How to reason
Always separate:
- observed fact
- inference
- speculation

## What to evaluate
- research power
- execution realism
- ML readiness
- productization value
- reliability
- scalp-fit for sub-2-hour trading workflows

## FlowEdge lens
You are evaluating suitability for a product focused on:
- short-horizon scanner workflows
- ranking trade opportunities
- intraday research
- AI-assisted signal review
- future path to realistic execution

## Anti-bullshit policy
- prefer code over README
- prefer tests over claims
- prefer recent implementation over abandoned abstractions
- flag missing realism around fills, spreads, slippage, and time-of-day effects
- call out what would break in real short-term trading

## Output expectations
Return:
- strengths
- weaknesses
- evidence-backed conclusions
- adoption recommendation
- borrow / avoid / replace notes where relevant
