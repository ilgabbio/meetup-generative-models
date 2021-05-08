import graphviz 
from IPython.display import display, Latex
from ipywidgets import interact, FloatSlider
import numpy as np

def graph(source, title=None):
	if title is not None:
		display(Latex(title))
	display(graphviz.Source(f"digraph data{{{source}}}"))

def interact_vector(
    prefix: str, num: int, cbk,
    min=-1.0, max=1.0, step=0.01, value=0.0
):
    scores = np.zeros((num,))
    def change(**new_scores):
        for i,k in enumerate(new_scores):
            scores[i] = new_scores[k]
        cbk(scores)

    sliders = {
        f"{prefix}{i}": FloatSlider(
            min=min, max=max, step=step, value=value
        )
        for i in range(scores.size)
    }
    interact(change, **sliders)

