Prefect Integration Brief
Audiology Research Copilot Backend

## Purpose

This document defines the requirements and constraints for integrating **Prefect workflow orchestration** into the existing backend of the Audiology Research Copilot project.

The goal is to introduce **observability, structured orchestration, and retry capability** around the research paper analysis pipeline while preserving the current backend architecture and API behavior.

This integration should be **minimal, surgical, and low-risk**.

The backend already exists and must **not be broadly rewritten**.

---

# Key Principle

**Wrap the existing pipeline with orchestration, do not redesign it.**

Prefect should monitor and coordinate existing steps rather than replace working logic.

---

# Context

The repository includes a full PRD describing the intended system behavior.
Before making changes, you must read the PRD to understand the intended architecture and constraints.

However, the **actual codebase may differ from the PRD**, so the implementation plan must be based on **the real code**, not assumptions.

---

# Project Overview

The backend analyzes cochlear implant research papers and produces a structured critique.

Typical pipeline stages include:

1. Input ingestion (PDF upload or research discovery item)
2. Text extraction
3. Paper classification
4. Structured field extraction
5. Evaluation steps
6. Report generation
7. Return structured JSON to the frontend

The final response contains structured outputs such as:

- executive snapshot
- score dashboard
- deep dive sections
- evaluation results
- warnings and metadata

---

# Prefect Integration Goals

Prefect should provide the following capabilities:

### 1. Workflow visibility
The full analysis pipeline should appear as a **Prefect flow graph**.

### 2. Step-level monitoring
Each major pipeline stage should be visible as a **task node**.

### 3. Debugging clarity
Logs should clearly show boundaries between major processing steps.

### 4. Retry capability
External calls (LLM calls, network requests, etc.) should support retries.

### 5. Optional parallel execution
Independent evaluator stages may run in parallel if safe.

---

# What Prefect Should NOT Do

The following changes are **out of scope**:

- Major architectural refactors
- Breaking existing API endpoints
- Changing response schemas used by the frontend
- Replacing existing logic unnecessarily
- Implementing full Prefect deployment infrastructure
- Adding Prefect scheduling / workers / enterprise configuration

For the hackathon environment, a **local Prefect runtime is sufficient**.

---

# Integration Strategy

Prefect should be layered onto the current pipeline.

Existing functions should be wrapped with:

`@task`

The overall orchestration should be defined using:

`@flow`

Example conceptual structure:

```
analyze_paper_flow
    extract_text_task
    classify_paper_task
    structured_extraction_task
    evaluation_tasks (parallel)
    report_builder_task
```

This structure should reflect the **actual implementation discovered in the codebase**.

---

# Retry Behavior

Retries should only be added where appropriate.

Typical retry candidates:

- OpenAI API calls
- network requests
- PDF extraction libraries that occasionally fail

Avoid adding retries to deterministic internal processing steps.

---

# Logging Expectations

Backend logs should clearly indicate major pipeline boundaries.

Example:

```
Starting PDF extraction
Extraction complete
Running classification
Classification complete
Running evaluator: statistical rigor
Running evaluator: methodological soundness
...
```

These logs should help debug failures during hackathon demos.

---

# Partial Failures

Partial failures do **not need special handling for this integration**.

If a stage fails, the failure should be visible through Prefect's workflow visualization.

This behavior is acceptable for the hackathon environment.

---

# Prefect Usage Scope

The integration should use only the core Prefect features:

- task decorator
- flow decorator
- retry configuration
- logging
- optional parallel execution

Avoid advanced Prefect infrastructure unless necessary.

---

# Required Assessment Process

Before implementing any changes, perform the following analysis.

## Step 1 â Repository Analysis

Identify:

- backend architecture
- API entrypoints
- analysis pipeline structure
- key modules and functions
- external API dependencies

Document the actual processing sequence.

---

## Step 2 â Pipeline Mapping

Determine:

- natural task boundaries
- possible parallel execution points
- safe retry locations

Map the pipeline into candidate Prefect tasks.

---

## Step 3 â Integration Design

Define:

- Prefect flow structure
- Prefect task boundaries
- file modifications required
- any compatibility concerns

---

# Required Output From This Assessment

Produce a **Prefect Integration Plan** with the following sections:

## 1. Observed Current State

Describe:

- backend structure
- analysis pipeline
- major modules
- key functions

---

## 2. Target Prefect Integration Design

Describe:

- proposed flow
- proposed task boundaries
- parallel execution opportunities
- retry locations

---

## 3. Files to Change

List specific files that require modification.

Explain what changes will occur in each file.

---

## 4. Implementation Steps

Provide a **step-by-step implementation plan** ordered from lowest risk to highest risk.

Example:

1. Install Prefect dependency
2. Introduce Prefect flow wrapper
3. Convert key functions into tasks
4. Add retry logic where appropriate
5. Test pipeline execution locally
6. Validate API compatibility
7. Validate Prefect workflow visualization

---

## 5. Risks

Identify any potential risks such as:

- API compatibility issues
- unexpected async behavior
- dependency conflicts
- performance changes

---

## 6. Questions / Assumptions

List any uncertainties discovered during repo analysis.

---

# Implementation Constraint

Do **not implement any changes yet.**

Your task is only to produce the Prefect integration 