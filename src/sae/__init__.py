"""SAE dataset, model, and training scaffolding for activation runs.

The first training backend is a small local PyTorch SAE. It is meant to make the
post-inference path executable while keeping the public boundary simple enough to
swap in a larger SAE library later.
"""
