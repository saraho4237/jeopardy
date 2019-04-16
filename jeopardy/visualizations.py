import matplotlib.pyplot as plt

def make_bar(x,y,color="blue",x_label=None,y_label=None,title=None):
    plt.bar(x, y, color=color)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
