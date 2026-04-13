# Hospital Emergency Response Optimization Using Reinforcement Learning

## Overview
This project implements a Q-learning agent to optimize ambulance dispatch decisions in a GridWorld-based hospital emergency response system. The model treats ambulance dispatch as a **resource allocation problem**, not a movement navigation problem. Ambulance travel time is abstracted using Manhattan distance, allowing the agent to focus on decision quality rather than pathfinding mechanics.

The system is designed to improve:
- response time
- emergency coverage
- ambulance utilization
- high-urgency patient prioritization

---

## Problem Statement
In a real emergency response system, dispatch decisions must be made under pressure using limited resources. Ambulances must be assigned efficiently based on:
- patient urgency
- travel distance
- ambulance availability
- service deadlines

Traditional rule-based systems often struggle in dynamic, high-variability environments. This project uses reinforcement learning to allow the system to learn better dispatch policies over time.

---

## Environment Design
The simulation uses a **5x5 GridWorld** with:

- **1 hospital**
- **2 ambulances**
- **5 patients**
- randomized patient locations
- urgency levels and service deadlines

### Key Modeling Decision
This assignment was implemented as a **dispatch-level decision system**.

- **Action space** = which ambulance to send to which patient, or hold
- **Travel time** = Manhattan distance between ambulance and patient
- **No step-by-step movement modeling**

This aligns the reinforcement learning problem more closely with real-world hospital dispatch operations.

---

## Reinforcement Learning Approach
The model uses **Q-learning** with an **epsilon-greedy policy**.

### State Representation
The state captures the system condition at each decision point, including:
- ambulance positions
- ambulance availability
- number of active patients
- nearest patient distance
- highest urgency level
- deadline urgency bucket

### Action Space
Valid actions include:
- dispatch ambulance `i` to patient `j`
- hold an idle ambulance

### Reward Design
The reward function was designed to balance speed, coverage, and urgency.

#### Reward Structure
- `+10` for successful patient pickup
- `+20` for successful delivery to hospital
- `-1 × distance` penalty for travel time
- `-15` for expired unattended emergencies
- `-5` for invalid dispatch attempts
- urgency bonus for serving high-priority patients
- hold penalty when urgent patients remain unattended

This design encourages the policy to optimize **system-wide effectiveness** rather than speed alone.

---

## Files
- `environment.py` - hospital emergency dispatch environment
- `q_agent.py` - Q-learning agent implementation
- `training.py` - training loop and history tracking
- `evaluation.py` - randomized evaluation and baseline comparison
- `visualization.py` - plots for training and results analysis
- `main.py` - main execution script
- `claude_conversation_log.md` - domain modeling and iterative design log
- `requirements.txt` - required Python packages
- `.gitignore` - excluded files and folders

---

## How to Run

### Activate the virtual environment
```bash
source venv/bin/activate
```
### Run the project
```bash
python3 main.py
```
## Evaluation Metrics

The trained agent was evaluated using the following metrics:

- **Average response time** — lower is better  
- **Average unattended emergencies** — lower is better  
- **Ambulance utilization** — higher is better  
- **Completed patients** — higher is better  
- **High-urgency service rate** — higher is better  

To validate policy generalization, evaluation was performed across **multiple randomized scenarios**, not a single fixed starting condition.

The trained Q-learning policy was also compared against a **random dispatch baseline**.

---

## Results

### Trained Q-Learning Agent
- **Average Response Time:** 2.658  
- **Average Unattended Emergencies:** 2.240  
- **Ambulance Utilization:** 0.283  
- **Completed Patients:** 2.760  
- **High-Urgency Service Rate:** 0.450  

### Random Baseline
- **Average Response Time:** 2.553  
- **Average Unattended Emergencies:** 2.400  
- **Ambulance Utilization:** 0.259  
- **Completed Patients:** 2.600  
- **High-Urgency Service Rate:** 0.431  

### Interpretation

The trained agent improved:
- emergency coverage  
- resource utilization  
- patient completion rate  
- prioritization of high-urgency cases  

While response time was slightly higher than the baseline, this reflects a strategic trade-off.

The learned policy prioritizes **system-wide decision quality** rather than only minimizing time.

---

## Generalization and Validation

A major focus of this project was demonstrating **policy generalization**.

To address this, the model was evaluated across:
- randomized patient placements  
- varied urgency mixes  
- varied service deadlines  
- multiple environment seeds  

This ensures the learned policy is **robust** and not simply memorizing one scenario.

---

## Key Takeaways

- Reinforcement learning can model real-world dispatch systems as resource allocation problems  
- Reward design is the most important driver of agent behavior  
- Generalization matters more than single-scenario performance  
- Optimal decisions often require trade-offs, not just speed  

---

## References

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

GeeksforGeeks. (2024, June 6). *Optimizing production scheduling with reinforcement learning*. https://www.geeksforgeeks.org/optimizing-production-scheduling-with-reinforcement-learning/

Jones, A. (n.d.). *Debugging reinforcement learning systems*. https://andyljones.com/posts/rl-debugging.html

---

## Author

**Brent Oatis** 
CST 643 – Week 4  
Concordia University St. Paul