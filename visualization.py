import matplotlib.pyplot as plt


def plot_training_curves(history: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["episode_rewards"])
    axes[0].set_title("Training Reward per Episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")

    axes[1].plot(history["response_times"])
    axes[1].set_title("Average Response Time per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Response Time")

    axes[2].plot(history["expired_counts"])
    axes[2].set_title("Unattended Emergencies per Episode")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Expired Emergencies")

    plt.tight_layout()
    plt.show()


def plot_baseline_comparison(trained_results: dict, baseline_results: dict) -> None:
    metrics = [
        "avg_response_time",
        "avg_unattended_emergencies",
        "avg_utilization",
        "avg_completed_patients",
        "avg_high_urgency_service_rate"
    ]

    trained_values = [trained_results[m] for m in metrics]
    baseline_values = [baseline_results[m] for m in metrics]

    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width / 2 for i in x], trained_values, width=width, label="Trained Agent")
    plt.bar([i + width / 2 for i in x], baseline_values, width=width, label="Random Baseline")

    plt.xticks(
        list(x),
        [
            "Response Time",
            "Unattended\nEmergencies",
            "Utilization",
            "Completed\nPatients",
            "High Urgency\nService Rate"
        ],
    )
    plt.title("Trained Agent vs Random Baseline")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.tight_layout()
    plt.show()