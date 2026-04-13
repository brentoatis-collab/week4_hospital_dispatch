from typing import Dict

from environment import HospitalDispatchEnv
from q_agent import QLearningAgent


def run_policy_episode(env: HospitalDispatchEnv, agent: QLearningAgent, greedy: bool = True) -> Dict[str, float]:
    state = env.reset()
    done = False

    while not done:
        valid_actions = env.get_valid_actions()

        if greedy:
            # Force exploitation during evaluation
            original_epsilon = agent.epsilon
            agent.epsilon = 0.0
            action = agent.choose_action(state, valid_actions)
            agent.epsilon = original_epsilon
        else:
            # Random baseline behavior
            import random
            action = random.choice(valid_actions) if valid_actions else ('hold', None, None)

        next_state, reward, done, info = env.step(action)
        state = next_state

    avg_response_time = (
        env.total_response_time / env.completed_patients
        if env.completed_patients > 0
        else 0.0
    )

    max_possible_busy_time = env.max_steps * env.num_ambulances
    utilization = (
        env.total_busy_time / max_possible_busy_time
        if max_possible_busy_time > 0
        else 0.0
    )

    high_urgency_total = sum(1 for p in env.patients.values() if p.urgency == 2)
    high_urgency_served = sum(
        1 for p in env.patients.values() if p.urgency == 2 and p.served
    )

    return {
        "avg_response_time": avg_response_time,
        "expired_emergencies": float(env.expired_patients),
        "utilization": utilization,
        "completed_patients": float(env.completed_patients),
        "high_urgency_service_rate": (
            high_urgency_served / high_urgency_total if high_urgency_total > 0 else 0.0
        ),
    }


def evaluate_agent(
    agent: QLearningAgent,
    num_eval_episodes: int = 100,
    seed: int = 100,
) -> Dict[str, float]:
    response_times = []
    expired_counts = []
    utilizations = []
    completed_counts = []
    high_urgency_rates = []

    for episode in range(num_eval_episodes):
        env = HospitalDispatchEnv(seed=seed + episode)
        metrics = run_policy_episode(env, agent, greedy=True)

        response_times.append(metrics["avg_response_time"])
        expired_counts.append(metrics["expired_emergencies"])
        utilizations.append(metrics["utilization"])
        completed_counts.append(metrics["completed_patients"])
        high_urgency_rates.append(metrics["high_urgency_service_rate"])

    results = {
        "avg_response_time": sum(response_times) / len(response_times),
        "avg_unattended_emergencies": sum(expired_counts) / len(expired_counts),
        "avg_utilization": sum(utilizations) / len(utilizations),
        "avg_completed_patients": sum(completed_counts) / len(completed_counts),
        "avg_high_urgency_service_rate": sum(high_urgency_rates) / len(high_urgency_rates),
    }

    return results


def evaluate_random_baseline(
    num_eval_episodes: int = 100,
    seed: int = 500,
) -> Dict[str, float]:
    response_times = []
    expired_counts = []
    utilizations = []
    completed_counts = []
    high_urgency_rates = []

    dummy_agent = QLearningAgent()

    for episode in range(num_eval_episodes):
        env = HospitalDispatchEnv(seed=seed + episode)
        metrics = run_policy_episode(env, dummy_agent, greedy=False)

        response_times.append(metrics["avg_response_time"])
        expired_counts.append(metrics["expired_emergencies"])
        utilizations.append(metrics["utilization"])
        completed_counts.append(metrics["completed_patients"])
        high_urgency_rates.append(metrics["high_urgency_service_rate"])

    results = {
        "avg_response_time": sum(response_times) / len(response_times),
        "avg_unattended_emergencies": sum(expired_counts) / len(expired_counts),
        "avg_utilization": sum(utilizations) / len(utilizations),
        "avg_completed_patients": sum(completed_counts) / len(completed_counts),
        "avg_high_urgency_service_rate": sum(high_urgency_rates) / len(high_urgency_rates),
    }

    return results