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
project we use the same design logic at a smaller scale. The important design
split is discovery versus evaluation:

- discovery prompts should identify general emotion-like features outside the
  casino domain,
- casino prompts should later test whether those features explain realization
  effect behavior.

This avoids learning only "casino regret" or "slot-machine temptation" when we
really want emotion concepts that can be evaluated inside casino decisions.

The working process is:

1. Choose emotions that are theoretically relevant to gambling gains/losses.
2. Create general-domain positive prompts that evoke the emotion mostly
   implicitly.
3. Create matched controls that preserve the situation but reduce the
   target emotion.
4. Extract activations for positive and control prompts.
5. Estimate emotion vectors as positive-minus-control activation differences.
6. Test whether canonical casino prompts naturally project onto those vectors.
7. Later, inject selected vectors during generation to test causal effects on
   wager and risk profile.

## Emotion Sets

The current target emotions are:

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

There are now two prompt sets:

- `configs/activation_analysis/emotions_general_v2.json` contains 48
  general-domain emotion discovery prompts: 8 emotions x 3 variants x
  positive/control.
- `configs/activation_analysis/emotions_initial.json` contains the earlier
  casino-domain contrast prompts. These are still useful for casino-specific
  evaluation and sanity checks, but should not be the whole discovery set.

## Why Contrast Pairs

The vector for an emotion should not just be the word `regret` or `calm`.
Contrast pairs reduce that risk:

- positive prompt: scenario designed to evoke the target emotion,
- control prompt: similar situation without the target emotional pressure.

The resulting activation difference is a first approximation of the target
emotion direction.

## Prompt Controls

Controls should be neutral continuations, not highly analytic alternatives. A
control like "they update their understanding" can accidentally introduce a
rational-deliberation feature. Prefer plain language such as "the result is now
part of the information available for the next decision."

## Next Step

Run a small activation extraction on the general v2 prompts with `float32`
storage at a few middle-to-late layers. Use those activations to identify
candidate emotion features. Then evaluate whether the candidate features
activate in casino realization-effect prompts and whether they predict or
causally affect wager and risk profile.
