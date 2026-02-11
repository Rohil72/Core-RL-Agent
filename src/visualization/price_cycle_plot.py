import matplotlib.pyplot as plt
import pandas as pd


def plot_price_and_cycles(
    dates: pd.Index,
    close_prices,
    cycles,
    title,
):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(dates, close_prices, label="Close Price", color="blue")
    ax1.set_ylabel("Price")

    # shade detected cycles
    for c in cycles:
        # Use start_date and end_date from the new Cycle object
        ax1.axvspan(
            c.start_date,
            c.end_date,
            color="green",
            alpha=0.15,
            label="Detected Cycle" if c == cycles[0] else ""
        )

    ax1.set_title(title)
    ax1.legend(loc="upper left")
    
    plt.tight_layout()
    plt.show()
