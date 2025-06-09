import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_context("talk")


os.makedirs("plots", exist_ok=True)


def plot_gradient_errors(file_path):
    data = np.load(file_path)

    base_fpm_grad = data["base_fpm_grad"]
    # base_fpm_steps = data["fpm_steps_base_DTO"]
    fpm_grads_DTO = data["fpm_grads_DTO"]
    fpm_steps_DTO = data["fpm_steps_DTO"]
    fpm_grads_REV = data["fpm_grads_REV"]
    fpm_steps_REV = data["fpm_steps_REV"]
    fpm_grads_fd = data["fpm_grads_fd"]

    # Compute absolute errors
    fpm_error_DTO = [abs(base_fpm_grad - grad) for grad in fpm_grads_DTO]
    fpm_error_REV = [abs(base_fpm_grad - grad) for grad in fpm_grads_REV]

    # Compute errors relative to finite differences
    fpm_error_FD_DTO = [
        abs(fd_grad - dto_grad) for fd_grad, dto_grad in zip(fpm_grads_fd, fpm_grads_DTO)
    ]
    fpm_error_FD_REV = [
        abs(fd_grad - otd_grad) for fd_grad, otd_grad in zip(fpm_grads_fd, fpm_grads_REV)
    ]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Gradient Error vs Steps in Subplot 1
    ax1.plot(
        fpm_steps_DTO,
        fpm_error_DTO,
        marker="o",
        linestyle="-",
        label="EfficientLeapFrog DTO",
    )
    ax1.plot(
        fpm_steps_REV,
        fpm_error_REV,
        marker="s",
        linestyle="--",
        label="EfficientLeapFrog Reverse",
    )

    # Customize Subplot 1
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Initial Field Gradient Error Gradient Error (|base - obtained|)")
    ax1.set_title("Initial Field Gradient Error Gradient Error vs Steps")
    ax1.legend()
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Plot Finite Difference Comparison in Subplot 2
    ax2.plot(fpm_steps_DTO, fpm_error_FD_DTO, marker="o", linestyle="-", label="DTO vs FD")
    ax2.plot(fpm_steps_REV, fpm_error_FD_REV, marker="s", linestyle="--", label="Reverse vs FD")

    # Customize Subplot 2
    # ax2.set_yscale("log")  # Log scale for gradient errors
    # ax2.set_ylim(1e-8, 1e-3)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Initial Field Gradient Error (|finite difference - obtained|)")
    ax2.set_title("Comparison with Finite Difference")
    ax2.legend()
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("plots/GS_FPM_initial_field_gradient_error.pdf", dpi=600, transparent=True)
    plt.show()


def old_plot_memory_runs(file_path):
    # Data Preparation
    data = np.load(file_path)

    fpm_grad_base_DTO = data["fpm_grad_base_DTO"]
    fpm_steps_base_DTO = data["fpm_steps_base_DTO"]
    # fpm_memories_base_DTO = data["fpm_memories_base_DTO"]
    fpm_grad_base_REV = data["fpm_grad_base_REV"]
    # fpm_steps_base_REV = data["fpm_steps_base_REV"]
    fpm_memories_base_REV = data["fpm_memories_base_REV"]
    fpm_grads_DTO = data["fpm_grads_DTO"]
    # fpm_steps_DTO = data["fpm_steps_DTO"]
    fpm_checkpoints_DTO = data["fpm_checkpoints_DTO"]
    fpm_memories_DTO = data["fpm_memories_DTO"]

    # Data Preparation
    checkpoints = fpm_checkpoints_DTO

    # Error baselines
    fpm_grad_base_REV_err = abs(
        fpm_grad_base_DTO - fpm_grad_base_REV
    )  # Baseline absolute error for OTD

    # Errors for DTO and OTD
    fpm_error_DTO = [abs(fpm_grad_base_DTO - grad) for grad in fpm_grads_DTO]
    mean_err = sum(fpm_error_DTO) / len(fpm_error_DTO)
    fpm_error_DTO = [mean_err if err == 0 else err for err in fpm_error_DTO]

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Define colors
    error_color = "tab:blue"
    memory_color = "tab:orange"

    # Primary Y-axis: Absolute Error
    ax1.set_xlabel("Checkpoints")
    ax1.set_ylabel("Absolute Error (log scale)", color=error_color)

    ax1.axhline(
        fpm_grad_base_REV_err,
        color=error_color,
        linestyle="-.",
        label=f"Reverse Baseline steps {fpm_steps_base_DTO}",
    )
    # Plot errors
    ax1.plot(
        checkpoints,
        fpm_error_DTO,
        marker="o",
        color=error_color,
        label=f"FPM Error (DTO) steps {fpm_steps_base_DTO}",
    )

    # Add horizontal line for baseline errors
    ax1.tick_params(axis="y", labelcolor=error_color)

    ax1.legend(loc="upper left")
    ax1.set_yscale("log")
    # Secondary Y-axis: Memory Usage
    ax2 = ax1.twinx()  # Create a secondary y-axis
    ax2.set_ylabel("Memory Usage (bytes)", color=memory_color)

    # Plot memory usage
    ax2.plot(
        checkpoints,
        fpm_memories_DTO,
        marker="o",
        linestyle="-",
        color=memory_color,
        label="Memory (DTO)",
    )

    # Add horizontal line for memory baselines
    ax2.axhline(fpm_memories_base_REV, color=memory_color, linestyle=":", label="Memory (Reverse)")

    ax2.tick_params(axis="y", labelcolor=memory_color)
    ax2.legend(loc="upper right")

    # Title and grid
    plt.title("Memory Usage and Absolute Error vs Checkpoints")
    plt.grid(which="both", linestyle="--", linewidth=0.5)  # Add x-y grid
    plt.minorticks_on()  # Enable minor ticks for finer grid
    plt.grid(which="minor", linestyle=":", linewidth=0.5)  # Minor grid lines

    # Show or Save
    plt.tight_layout()
    plt.savefig("plots/GS_FPM_memory_usage_and_error.pdf", dpi=600, transparent=True)
    plt.show()


def plot_memory_runs(file_path):
    data = np.load(file_path)

    fpm_grad_base_DTO = data["fpm_grad_base_DTO"]
    # fpm_steps_base_DTO = data["fpm_steps_base_DTO"]
    fpm_grad_base_REV = data["fpm_grad_base_REV"]
    fpm_memories_base_REV = data["fpm_memories_base_REV"]
    fpm_grads_DTO = data["fpm_grads_DTO"]
    fpm_checkpoints_DTO = data["fpm_checkpoints_DTO"]
    fpm_memories_DTO = data["fpm_memories_DTO"]

    checkpoints = fpm_checkpoints_DTO
    fpm_grad_base_REV_err = abs(fpm_grad_base_DTO - fpm_grad_base_REV)
    fpm_error_DTO = [abs(fpm_grad_base_DTO - grad) for grad in fpm_grads_DTO]
    mean_err = sum(fpm_error_DTO) / len(fpm_error_DTO)
    fpm_error_DTO = [mean_err if err == 0 else err for err in fpm_error_DTO]

    fig, ax1 = plt.subplots(figsize=(12, 8))  # Smaller figure

    # Primary Y-axis: Absolute Error
    error_color = "tab:blue"
    memory_color = "tab:green"  # Light-mode friendly orange

    ax1.set_xlabel("Checkpoints")
    ax1.set_ylabel("Absolute Error (log scale)", color=error_color)

    ax1.plot(
        checkpoints,
        fpm_error_DTO,
        marker="o",
        color=error_color,
        label="Checkpointing Error",
    )
    ax1.axhline(
        fpm_grad_base_REV_err,
        color=error_color,
        linestyle="-.",
        label="Reverse Adjoint Error",
    )
    ax1.tick_params(axis="y", labelcolor=error_color)
    ax1.set_yscale("log")
    ax1.legend(loc="upper left")

    # Secondary Y-axis: Memory
    ax2 = ax1.twinx()
    ax2.set_ylabel("Memory Usage (bytes)", color=memory_color)

    ax2.plot(
        checkpoints,
        fpm_memories_DTO,
        marker="o",
        linestyle="-",
        color=memory_color,
        label="Checkpointing Memory",
    )
    ax2.axhline(
        fpm_memories_base_REV,
        color=memory_color,
        linestyle=":",
        label="Reverse Adjoint Memory",
    )
    ax2.tick_params(axis="y", labelcolor=memory_color)
    ax2.legend(loc="upper right")

    # Title + Layout
    plt.title("Reverse Adjoint vs. Checkpointing: Gradient Accuracy & Memory")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.minorticks_on()
    plt.grid(which="minor", linestyle=":", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("plots/adjoint_vs_checkpointing_memory.png", dpi=600, transparent=True)
    plt.show()
