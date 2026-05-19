
# SkyClaw: An Agent-Harnessing Framework on HAPS with Information Bottlenecked Memory Compression

High altitude platform stations (HAPS) are promising aerial base stations for future air-to-ground networks due to their wide coverage and flexible deployment. However, their stratospheric operation makes maintenance, real-time tuning, and fault recovery difficult. Moreover, HAPS coverage performance is affected by non-stationary node distributions, emergency demands, and tidal mobility patterns. This dynamic environment requires continuous three-dimensional trajectory adaptation.

Existing methods (convex optimization, heuristic search, reinforcement learning) rely on fixed models, suffering from scalability limitations and lacking autonomous reasoning for long-horizon deployment. Large language model (LLM) agents provide semantic understanding, task decomposition, and tool-augmented decision-making, but directly applying them to HAPS control faces key challenges: HAPS states are continuous and high-dimensional, their physical actions must satisfy strict mobility and safety constraints, and memory accumulation expands the context window and degrades reasoning performance.

**SkyClaw** addresses these challenges by:
- Decoupling reactive execution from proactive reasoning
- Grounding physical actions through deterministic tools
- Introducing an **Information Bottleneck (IB)-guided memory compression** to preserve task-relevant knowledge and filter redundant observations, enabling stable long-horizon operation

<img width="719" height="540" alt="SkyClaw" src="https://github.com/user-attachments/assets/41be24f8-33ed-4d67-aba3-0e485fbf673d" />

## Simulation Results

| Scenario | Metric | Performance |
|---|---|---|
| Routine | Average covered nodes (out of 500) | 328 |
| Routine | Final covered nodes (out of 500) | 421 |
| Non-stationary | Emergency task completion rate | 90% |
| Non-stationary | Tidal task completion rate | 100% |
| Non-stationary | Average tidal coverage ratio | 91.3% |

## Overview

- 500 ground users (50% mobile) with cluster-based movement patterns
- 4 HAPS agents with LLM-powered reasoning (DeepSeek) for autonomous decision-making
- Tidal gathering events and emergency response scenarios
- Information Bottleneck-based memory compression and communication optimization
- 288-frame simulation (24 hours, 5-min steps) with visualization output

## Citation

If you use SkyClaw in your research, please cite:

```
@misc{SkyClaw2025,
  title = {An Agent Harnessing Framework on HAPS with Information Bottlenecked Memory Compression},
  author = {Kang, Songqi and others},
  year = {2025}
}
```

## Project Structure

```
core_agent/       Agent logic (HAPS agents, gateway, LLM client)
environment/      Physical world simulation
skills/           Agent skills (Lagrangian memory optimizer, memory consolidator, state compressor)
tools/            Agent tools (optimizer, partitioner, follower)
utils/            Utility functions and visualization
```

## Quick Start

```bash
python main_simulation.py
```

## Configuration

Edit `config.yaml` to adjust simulation parameters (agent count, user distribution, events, etc.).
