"""SAE dataset, model, training, inspection, and baseline-loading scaffolding.

The first training backend is a small local PyTorch SAE. It is meant to make the
post-inference path executable while keeping the public boundary simple enough to
swap in a larger SAE library later.
"""
