# Week 4 – Claude Conversation Log  
## Hospital Emergency Response RL System

### Phase 1: Domain Understanding

Goal:
Design a reinforcement learning system to optimize ambulance dispatch decisions in a hospital emergency scenario.

System Components:
- 5 patients (random positions)
- 2 ambulances (start at hospital)
- 1 hospital
- 5x5 grid

Key Objective:
Minimize response time, reduce unattended emergencies, and improve ambulance utilization.

---

### Phase 2: Initial Problem Framing

This is a resource allocation problem, not a navigation problem.

Actions:
- Dispatch ambulance to patient
- Hold ambulance

Travel Time:
- Manhattan distance (|Δx| + |Δy|)

---

### Phase 3: State Design (Initial Thoughts)

State includes:
- Ambulance positions
- Ambulance availability (idle/busy)
- Patient positions
- Patient urgency and deadlines

---

### Phase 4: Reward Design (Initial Thinking)

Initial reward goals:
- Reward fast dispatch
- Reward successful pickup and delivery
- Penalize delays
- Penalize unattended emergencies

Trade-offs:
- Speed vs urgency
- Coverage vs efficiency

---

### Phase 5: Generalization and Evaluation Design

To address policy generalization, the trained Q-learning agent was evaluated across multiple randomized hospital emergency scenarios rather than a single fixed setup. Each evaluation episode used a fresh environment seed, changing patient locations, urgency mixes, and deadlines. This ensured the learned policy was tested for robustness and not just memorization of one scenario.

Evaluation focused on:
- Average response time
- Average unattended emergencies
- Average ambulance utilization
- Average completed patients

The trained agent was also compared against a random baseline to verify that the learned dispatch policy improved decision quality beyond chance behavior.

---

### Phase 6: Sanity Checks and System Validation

To ensure correctness and robustness, validation was performed at each stage of development rather than waiting until final evaluation.

**Environment Validation**
- Confirmed correct initialization:
  - 5 patients, 2 ambulances, 1 hospital
- Verified:
  - State structure consistency across resets
  - Valid action generation (dispatch + hold)
  - Manhattan distance calculations for travel time

Validation approach:
- Reset environment multiple times
- Printed state and action space
- Manually inspected edge cases (no available ambulances, expired patients)

---

**Agent Validation**
- Confirmed Q-table updates correctly across episodes
- Verified epsilon-greedy behavior:
  - Early training → exploration-heavy
  - Later training → exploitation-focused
- Checked reward accumulation trends for stability

Observation:
- Early instability was expected and decreased as epsilon decayed

---

**Training Validation**
- Ran short training cycles (10 episodes) for quick checks
- Scaled to larger runs (50–1000 episodes) for convergence validation
- Verified:
  - Reward trends begin to stabilize
  - Dispatch decisions become less random over time

Adjustment:
- Implemented epsilon decay to balance exploration vs. exploitation

---

**Evaluation Validation (Generalization Requirement)**

To align with real-world expectations and assignment requirements, evaluation was conducted across multiple randomized scenarios.

Each evaluation run:
- Used a new environment seed
- Varied:
  - Patient locations
  - Urgency levels
  - Deadlines

This ensured the agent was not memorizing a single configuration but instead learning a **generalizable dispatch policy**.

---

### Phase 7: Iteration and Refinement

Several refinements were made based on observed system behavior:

- Adjusted reward weights to balance:
  - Response speed
  - Emergency coverage
  - Urgency prioritization

- Added penalties for:
  - Expired patients (missed service deadlines)
  - Invalid dispatch attempts (assigning already-served patients)

- Expanded evaluation metrics to include:
  - High-urgency service rate
  - Resource utilization efficiency

---

### Phase 8: Key Observations

- Reinforcement learning optimizes what the reward function defines—not necessarily what is intuitively “best.”
- Lower response time alone does not indicate a better policy.
- The trained agent demonstrated improved:
  - Coverage (fewer unattended emergencies)
  - Resource utilization
  - High-urgency prioritization

- Slight increases in response time reflected more balanced system-wide decision-making rather than inefficiency.

---

### Phase 9: Final Reflection

The most critical challenge in this project was not implementing Q-learning, but translating a real-world emergency response system into a structured reinforcement learning framework.

Success depended on:
- Thoughtful state representation
- Careful reward design
- Iterative validation and debugging

This process reinforced that applied reinforcement learning is fundamentally a **system design problem**, not just an algorithmic one.