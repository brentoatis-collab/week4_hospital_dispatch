from training import train_agent
from evaluation import evaluate_agent, evaluate_random_baseline
from visualization import plot_training_curves, plot_baseline_comparison


def main():
    print("Starting Week 4 Hospital Emergency Dispatch RL System...\n")

    # Train agent
    episodes = 1000
    agent, history = train_agent(episodes=episodes)

    # Evaluate trained agent
    trained_results = evaluate_agent(agent, num_eval_episodes=100)

    # Evaluate random baseline
    baseline_results = evaluate_random_baseline(num_eval_episodes=100)

    # Print results
    print("\n=== Trained Agent Results ===")
    for key, value in trained_results.items():
        print(f"{key}: {value:.3f}")

    print("\n=== Random Baseline Results ===")
    for key, value in baseline_results.items():
        print(f"{key}: {value:.3f}")

    # Generate plots
    plot_training_curves(history)
    plot_baseline_comparison(trained_results, baseline_results)


if __name__ == "__main__":
    main()