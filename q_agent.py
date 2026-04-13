import random
from collections import defaultdict
from typing import Dict, List, Tuple


class QLearningAgent:
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        seed: int = 42,
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.random = random.Random(seed)

        # Q-table maps state -> action_index -> Q-value
        self.q_table: Dict[Tuple, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

    def choose_action(
        self,
        state: Tuple,
        valid_actions: List[Tuple[str, int | None, int | None]],
    ) -> Tuple[str, int | None, int | None]:
        """
        Epsilon-greedy action selection from valid action set only.
        """
        if not valid_actions:
            return ('hold', None, None)

        if self.random.random() < self.epsilon:
            return self.random.choice(valid_actions)

        q_values = [self.q_table[state][idx] for idx in range(len(valid_actions))]
        max_q = max(q_values)

        best_indices = [idx for idx, q in enumerate(q_values) if q == max_q]
        chosen_index = self.random.choice(best_indices)

        return valid_actions[chosen_index]

    def update_q_value(
        self,
        state: Tuple,
        action,
        reward: float,
        next_state: Tuple,
        next_valid_actions: List[Tuple[str, int | None, int | None]],
        current_valid_actions: List[Tuple[str, int | None, int | None]],
    ) -> None:
        """
        Q-learning update using action index within current valid action list.
        """
        if action not in current_valid_actions:
            return

        action_index = current_valid_actions.index(action)
        current_q = self.q_table[state][action_index]

        if next_valid_actions:
            next_q_values = [
                self.q_table[next_state][idx] for idx in range(len(next_valid_actions))
            ]
            max_next_q = max(next_q_values)
        else:
            max_next_q = 0.0

        updated_q = current_q + self.lr * (
            reward + self.gamma * max_next_q - current_q
        )

        self.q_table[state][action_index] = updated_q

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_values_for_state(
        self,
        state: Tuple,
        valid_actions: List[Tuple[str, int | None, int | None]],
    ) -> Dict[Tuple[str, int | None, int | None], float]:
        """
        Helpful for debugging and inspection.
        """
        return {
            action: self.q_table[state][idx]
            for idx, action in enumerate(valid_actions)
        }