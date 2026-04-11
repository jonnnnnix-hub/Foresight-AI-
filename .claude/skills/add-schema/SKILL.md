---
name: add-schema
description: Use when creating or updating Pydantic schemas for evidence, scoring, reports, and agent outputs.
allowed-tools: Read Write Edit Grep Glob Bash
---

# Add Schema

Use this skill when:
- creating a new Pydantic schema
- tightening an existing schema
- replacing loose JSON blobs with typed models

## Required workflow
1. Define the minimum stable contract
2. Use Pydantic v2
3. Add field constraints where useful
4. Keep nested models readable
5. Add parser or validation tests
6. Update downstream consumers

## Rules
- do not use dict-any soup
- prefer explicit models over loose structures
- add docstrings when the schema intent is not obvious
- do not add fields just because they might be useful someday
- favor stable contracts over speculative breadth
