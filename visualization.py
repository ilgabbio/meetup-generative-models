import graphviz 
from IPython.display import display, Latex

def graph(source, title=None):
	if title is not None:
		display(Latex(title))
	display(graphviz.Source(f"digraph data{{{source}}}"))

