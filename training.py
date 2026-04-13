from typing import Dict, List, Tuple

from environment import HospitalDispatchEnv
from q_agent import QLearningAgent


def train_agent(
    episodes: int = 1000,
    seed: int = 42,
) -> Tuple[QLearningAgent, Dict[str, List[float]]]:
    env = HospitalDispatchEnv(seed=seed)
    agent = QLearningAgent()

    history = {
        "episode_rewards": [],
        "response_times": [],
        "expired_counts": [],
        "utilization": [],
        "completed_counts": [],
        "epsilon": [],
    }

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            current_valid_actions = env.get_valid_actions()
            action = agent.choose_action(state, current_valid_actions)

            next_state, reward, done, info = env.step(action)
            next_valid_actions = env.get_valid_actions()

            agent.update_q_value(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_valid_actions=next_valid_actions,
                current_valid_actions=current_valid_actions,
            )

            state = next_state
            total_reward += reward

        # Episode metrics
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

        history["episode_rewards"].append(total_reward)
        history["response_times"].append(avg_response_time)
        history["expired_counts"].append(env.expired_patients)
        history["utilization"].append(utilization)
        history["completed_counts"].append(env.completed_patients)
        history["epsilon"].append(agent.epsilon)

        agent.decay_epsilon()

        # Optional training progress print every 100 episodes
        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Avg Response Time: {avg_response_time:.2f} | "
                f"Expired: {env.expired_patients} | "
                f"Utilization: {utilization:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    return agent, history