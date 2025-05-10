# ────────────────────────────────────────────────────────────────────
# Simple matplotlib dashboard for Part-2 results
# ────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

def plot_results(omega_vals, d_max_vals, weight_vals, fill_vals):
    """
    Draw three line-charts:
        1) ω  vs  d_max
        2) ω  vs  achieved priority weight
        3) ω  vs  fill-rate  (assigned / total positions)
    """
    if not omega_vals:              # nothing to plot
        print("No results collected – nothing to plot.")
        return

    # Chart 1 – ω vs d_max
    plt.figure()
    plt.plot(omega_vals, d_max_vals, marker='o')
    plt.xlabel('ω (weight retention %)')
    plt.ylabel('d_max (mean absolute difference)')
    plt.title('ω vs d_max')
    plt.grid(True)

    # Chart 2 – ω vs achieved weight
    plt.figure()
    plt.plot(omega_vals, weight_vals, marker='o')
    plt.xlabel('ω (weight retention %)')
    plt.ylabel('Achieved priority weight')
    plt.title('ω vs achieved weight')
    plt.grid(True)

    # Chart 3 – ω vs fill-rate
    plt.figure()
    plt.plot(omega_vals, fill_vals, marker='o')
    plt.xlabel('ω (weight retention %)')
    plt.ylabel('Fill rate (assigned / total positions)')
    plt.title('ω vs fill rate')
    plt.grid(True)

    plt.show()
