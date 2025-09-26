import matplotlib.pyplot as plt
import streamlit as st

def plotSingle_params(x, y, plot_type="line", title="Single Plot", xlabel="X-axis", ylabel="Y-axis"):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if plot_type == "line":
        ax.plot(x, y, marker='o')
    elif plot_type == "bar":
        ax.bar(x, y)
    elif plot_type == "scatter":
        ax.scatter(x, y)
    elif plot_type == "histogram":
        ax.hist(y, bins=len(x))
    else:
        ax.plot(x, y, marker='o')  # Default to line plot
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    return fig

def plotSingle(x=None, y=None, plot_type="line", title="Single Plot", xlabel="X-axis", ylabel="Y-axis"):
    """Streamlit-specific plotting function"""
    import streamlit as st
    fig = plotSingle_params(x, y, plot_type, title, xlabel, ylabel)
    st.pyplot(fig)
    plt.close(fig)
    
def plotRow(x=None, y=None, plot_type="line", title="Row Plot", xlabel="X-axis", ylabel="Y-axis"):
    """Streamlit-specific plotting function for multiple subplots in a row"""
    import streamlit as st
    num_plots = len(y)
    fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    if num_plots == 1:
        axs = [axs]
    
    for i in range(num_plots):
        if plot_type == "line":
            axs[i].plot(x[i], y[i], marker='o')
        elif plot_type == "bar":
            axs[i].bar(x[i], y[i])
        elif plot_type == "scatter":
            axs[i].scatter(x[i], y[i])
        elif plot_type == "histogram":
            axs[i].hist(y[i], bins=len(x[i]))
        else:
            axs[i].plot(x[i], y[i], marker='o')  # Default to line plot
        
        axs[i].set_title(f"{title} {i+1}")
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].grid(True)
    
    st.pyplot(fig)
    plt.close(fig)