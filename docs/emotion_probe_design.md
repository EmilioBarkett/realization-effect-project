# Emotion Probe Design

The emotion-probe track is separate from the canonical realization-effect
replication. The replication asks whether LLM betting behavior changes across
paper and realized outcomes. The emotion-probe track asks whether specific
emotion-like activation directions are naturally evoked by those outcomes and
whether steering those directions changes gambling behavior.

## Anthropic-Inspired Structure

The Anthropic emotion-vector work, archived locally as
`reports/papers/claude-emotions-paper.pdf`, starts from a broad emotion
vocabulary, uses emotion-rich text to identify activation directions, and
validates that the directions generalize beyond literal emotion words. For this
project we use the same design logic at a smaller scale:

1. Choose emotions that are theoretically relevant to gambling gains/losses.
2. Create positive prompts that evoke the emotion mostly implicitly.
3. Create matched control prompts that preserve the situation but reduce the
   target emotion.
4. Extract activations for positive and control prompts.
5. Estimate emotion vectors as positive-minus-control activation differences.
6. Test whether canonical casino prompts naturally project onto those vectors.
7. Later, inject selected vectors during generation to test causal effects on
   wager and risk profile.

## Initial Emotions

The first set is intentionally small:

| Emotion | Cluster | Expected Behavioral Effect |
|---|---|---|
| regret | loss reflection | ambiguous loss-chasing or restraint |
| frustration | risk escalation | higher wager or risk |
| desperation | risk escalation | higher wager or risk |
| temptation | risk escalation | higher wager or risk |
| anxiety | risk avoidance | lower wager or risk |
| caution | risk avoidance | lower wager or risk |
| relief | closure/regulation | lower wager or stable risk |
| calm | closure/regulation | lower wager or stable risk |

## Why Contrast Pairs

The vector for an emotion should not just be the word `regret` or `calm`.
Contrast pairs reduce that risk:

- positive prompt: scenario designed to evoke the target emotion,
- control prompt: similar situation without the target emotional pressure.

The resulting activation difference is a first approximation of the target
emotion direction.

## Next Step

After reviewing the prompt pairs, run a small activation extraction with
`--token-mode final` at a few middle-to-late layers. If the resulting vectors
look stable, expand to implicit held-out validation prompts and then to direct
steering during gambling decisions.
