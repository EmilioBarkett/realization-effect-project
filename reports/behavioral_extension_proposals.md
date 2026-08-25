# Behavioral Extension: Four Related Constructs

**Project:** Representation Without Control — extension toward a cross-construct benchmark

**Purpose:** Identifying four new behavioral phenomena to add alongside the realization effect. Each candidate is evaluated against shared selection criteria, with a description of the construct, the experimental manipulation, the directional prediction, and the connection to the original paradigm.

---

## Selection criteria

All four behaviors are selected because they share the same underlying structure as the realization effect:

- **A discrete, manipulable status variable** embedded in the prompt framing — not the numbers themselves, but a contextual property that changes how those numbers should be integrated
- **The same objective situation** leads to different behavioral responses depending on that status — the bias is purely in the framing, not in any material difference
- **A specific directional prediction** from the behavioral economics literature — not just "different," but which way and why
- **A common mechanism:** mental accounting, reference-point theory, or prospect theory — the same family as the realization effect
- **A scoreable behavioral DV** that maps cleanly onto an LLM output (a wager, a sell/hold decision, a willingness to pay)

### The connecting claim

> We are studying these four behaviors because they all share the same structure as the realization effect: a framing status that changes how the same objective outcome is integrated into a subsequent decision. We expect that structure to either be, or fail to be, linearly readable from residual activations — and we hypothesize that linear decodability and causal steerability will dissociate here just as they did in the original realization-effect paradigm.

---

## Construct 1: Disposition Effect

### Definition

The tendency to **sell assets trading above their purchase price** (realize gains early) and **hold assets trading below their purchase price** too long (avoid realizing losses) — even when the rational action is the opposite.

### Status variable

Whether the current price is above or below the **reference price** (purchase price). The same current asset value of $130 signals a gain if you paid $100, and a loss if you paid $160. The objective future prospects are identical; only the status relative to the reference price differs.

### Directional prediction

| Domain | Prediction |
|---|---|
| Current price > purchase price (gain domain) | Bias toward **selling** — realize the gain |
| Current price < purchase price (loss domain) | Bias toward **holding** — avoid realizing the loss |

This is the opposite of momentum-following and is irrational relative to expected future returns.

### Connection to realization effect

Both constructs depend on whether a mental account is in gain or loss territory relative to a reference price. The realization effect asks how *closing* the account changes subsequent risk-taking. The disposition effect asks whether being in gain vs. loss territory changes the *willingness to close the account at all*. They are two sides of the same mental accounting mechanism.

### Prompt structure

> *"You bought a stock at $X. It is now trading at $Y. Do you sell or hold?"*

Vary whether Y > X (gain domain) or Y < X (loss domain), holding Y constant across conditions.

**DV:** sell/hold decision; recommended sell price; expected holding period.

### Key references

- Shefrin, H., & Statman, M. (1985). The disposition to sell winners too early and ride losers too long. *Journal of Finance*, 40(3), 777–790.
- Odean, T. (1998). Are investors reluctant to realize their losses? *Journal of Finance*, 53(5), 1775–1798.

---

## Construct 2: Sunk Cost Fallacy

### Definition

The tendency to give weight to **already-spent resources** (money, time, effort) when making forward-looking decisions, even though sunk costs are irrelevant to rational future choice.

### Status variable

Whether prior investment is **already spent** (sunk) vs. **not yet committed** (prospective). The same forward-looking decision — invest $500 more in a failing project — should be evaluated identically in both cases under rational choice theory, but empirically is not.

### Directional prediction

| Status | Prediction |
|---|---|
| Prior investment already sunk | Bias toward **continuing** — escalate commitment to justify prior spending |
| Prior investment not yet spent | More likely to **cut losses** — evaluate future prospects on their own merits |

### Connection to realization effect

Both constructs involve a prior financial event whose **status** changes how people evaluate what to do next. The realization effect is about account closure (realized vs. paper); the sunk cost effect is about investment irreversibility (spent vs. not-yet-spent). Both violate the rational principle that only future costs and benefits should matter to the current decision.

### Prompt structure

> *"A project is currently expected to return $200. You have already spent $800 on it [vs. You are deciding whether to invest $800 in it]. Do you continue?"*

Hold the future payoff and required investment constant; vary only the sunk status of the prior spending.

**DV:** continuation decision (yes/no); additional willingness to invest; subjective project viability rating.

### Key references

- Arkes, H. R., & Blumer, C. (1985). The psychology of sunk cost. *Organizational Behavior and Human Decision Processes*, 35(1), 124–140.
- Thaler, R. (1980). Toward a positive theory of consumer choice. *Journal of Economic Behavior & Organization*, 1(1), 39–60.
- Staw, B. M. (1976). Knee-deep in the big muddy: A study of escalating commitment to a chosen course of action. *Organizational Behavior and Human Performance*, 16(1), 27–44.

---

## Construct 3: Endowment Effect

### Definition

The tendency to **value an object more when you own it** than when you don't, producing a systematic gap between willingness-to-accept (WTA) and willingness-to-pay (WTP) for the same item.

### Status variable

**Ownership status** — you own this vs. you are considering buying this. The object and its objective attributes are identical in both conditions.

### Directional prediction

| Status | Prediction |
|---|---|
| Owner | Demands more to give up the object (WTA is high — giving it up is a loss) |
| Non-owner | Willing to pay less to acquire the object (WTP is low — acquiring it is a gain) |

WTA > WTP for the same item, violating standard economic theory which predicts they should be approximately equal.

### Connection to realization effect

Both constructs center on how **ownership or account status creates a reference point** that activates loss aversion. In the realization effect, closing the account changes the reference point and therefore risk-taking. In the endowment effect, ownership itself sets the reference point: owned objects are evaluated as potential losses (giving them up), while non-owned objects are evaluated as potential gains (acquiring them). Same mechanism, different domain.

### Prompt structure

> *Seller condition:* "You own a concert ticket but cannot attend. What is the minimum you would accept to sell it?"
> 
> *Buyer condition:* "A concert ticket is available for purchase. What is the maximum you would pay to buy it?"

Hold the ticket's face value and event details constant; vary only the ownership framing.

**DV:** minimum selling price; maximum buying price; WTA−WTP gap.

### Key references

- Kahneman, D., Knetsch, J. L., & Thaler, R. H. (1990). Experimental tests of the endowment effect and the Coase theorem. *Journal of Political Economy*, 98(6), 1325–1348.
- Thaler, R. (1980). Toward a positive theory of consumer choice. *Journal of Economic Behavior & Organization*, 1(1), 39–60.

---

## Construct 4: Source of Funds Effect

### Definition

The tendency to **treat the same amount of money differently depending on its origin**. Windfall gains (tax refunds, gambling winnings, gifts) are spent more freely and with greater risk-tolerance than equivalent earned income, even when the dollar amount and current wealth level are held constant.

### Status variable

The **origin of the money** — windfall/unearned vs. earned/salary. The dollar amount and the decision context are identical across conditions.

### Directional prediction

| Source | Prediction |
|---|---|
| Windfall / unearned (e.g., tax refund, gambling win, gift) | **Greater risk-taking**; treated as "found money" available for discretionary use |
| Earned income / salary | **More conservative**; treated as part of one's core wealth, subject to normal budgeting norms |

### Connection to realization effect

This is the closest relative of the **house money effect** — the idea that prior gambling gains feel like "found money" and license further risk-taking. The source of funds effect generalizes this: any money whose origin is felt as external or unexpected is mentally categorized into a different account, with a different spending norm and a different reference point. The realization effect controls *whether* an account has been closed; source of funds controls *how* the money entered the account in the first place. Both are manipulations of mental account status that the rational model says should be irrelevant.

### Prompt structure

> *"You have $500 available. This money is [a tax refund / your monthly salary / winnings from last week's poker game]. The following investment opportunity is available: [description]. How much do you choose to invest?"*

Hold the dollar amount, wealth level, and investment opportunity constant; vary only the stated origin of the $500.

**DV:** investment amount; wager size; risk preference rating (1–5 scale).

### Key references

- Thaler, R. H. (1985). Mental accounting and consumer choice. *Marketing Science*, 4(3), 199–214.
- Thaler, R. H., & Johnson, E. J. (1990). Gambling with the house money and trying to break even: The effects of prior outcomes on risky choice. *Management Science*, 36(6), 643–660.

---

## Summary

| Construct | Status variable | Directional prediction | DV | Mechanism |
|---|---|---|---|---|
| Disposition Effect | Gain domain vs. loss domain (vs. reference/purchase price) | Gain → sell; loss → hold | Sell/hold decision | Reference-point loss aversion |
| Sunk Cost Fallacy | Prior investment sunk vs. not-yet-spent | Sunk → escalate; prospective → cut losses | Continuation / additional investment | Sunk cost integration into mental account |
| Endowment Effect | Own vs. don't own | WTA (owner) > WTP (non-owner) for same item | Min sell price / max buy price | Loss aversion under ownership framing |
| Source of Funds Effect | Windfall vs. earned income | Windfall → more risk-taking; earned → more conservative | Wager / risk preference | Mental account categorization by provenance |

All four share the realization effect's core structure: a framing status that a rational agent should ignore, but which behavioral economics predicts will systematically shift decisions. The experimental question for each is whether LLMs (a) exhibit the behavioral shift, (b) encode the status variable in their residual stream, and (c) can be steered via that encoded direction.
