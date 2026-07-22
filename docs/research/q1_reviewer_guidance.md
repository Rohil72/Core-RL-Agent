# Q1 Reviewer Guidance and Notes

What Q1 reviewers want

A Q1 reviewer wants something like:

Paper A

Event-Aware Temporal Fusion Transformer for Sparse Fundamental Signals

Novel claim:

Existing TFTs treat fundamentals as static features.

We introduce event-conditioned attention with temporal decay.

That's publishable.

Paper B

Persistent Memory for Financial Regime Tracking

Novel claim:

Memory slots learn latent market regimes and improve long-horizon decision making.

Also publishable.

Paper C

Learning Technical Analysis Rules Instead of Hand-Crafting Them

Novel claim:

Variable-selection networks discover optimal MA windows automatically.

Also publishable.

Your current proposal bundles all three papers together.

Reviewers hate that.

What is genuinely novel?

Among everything in the document, the most interesting part is:

Event-aware sparse fundamental fusion

The literature is full of:

LSTMs
TFTs
Technical indicators
Earnings data

But relatively few papers treat earnings reports as:

sparse high-information events with temporal decay and explicit event representations

That part is genuinely interesting.

Novelty score:

7.5/10

TFT + external memory

Moderately novel.

Problem:

Memory-augmented transformers already exist.

Examples:

Memformer
Memory Transformer
Transformer-XL variants

Reviewers will say:

Why does finance need memory?

You must prove:

memory learns regimes
memory improves returns
memory improves generalization

Without that:

Novelty = 5/10

With proof:

Novelty = 8/10

Learning moving-average windows

Interesting but not highly novel.

Researchers have been learning indicator parameters for years.

Novelty:

4/10

Useful? Absolutely.

Novel? Not much.

The real Q1-worthy angle

If I were writing this for publication, I would center the paper around:

Sparse Event-Aware Financial Forecasting

Problem:

Technical data arrives daily.
Fundamentals arrive quarterly.
Existing models flatten fundamentals into daily rows.
Information gets diluted.

Method:

Event token representation
Freshness decay
Event-aware attention
Long-horizon forecasting

This is a clean research story.

Dataset concerns

This is actually where reviewers may attack hardest.

Your proposed split:

Train: 2011-2019
Skip 2020
Fine-tune: 2021+

looks sensible for engineering.

For publication?

Not enough.

They'll ask:

Why exclude 2020?

If answer:

because COVID was weird

that's not scientific.

Instead:

train including 2020
train excluding 2020
compare robustness

Now you have a research result.

Weakest part of the proposal

The stage labels.

This concerns me most.

You have:

weak detector → stage labels → auxiliary supervision

Problem:

If stage labels are generated from technical rules and model inputs contain those same technical indicators:

then the model may merely learn:

reproduce detector

instead of

discover alpha

This is a very common finance ML trap.

Novelity:

2/10

Risk:

8/10

What top journals will ask

A reviewer from journals like:

Expert Systems with Applications
Information Sciences
Decision Support Systems
Quantitative Finance
European Journal of Operational Research

will likely ask:

Why TFT instead of PatchTST?
Why memory?
Why event tokens?
Why fundamentals?
Why stage supervision?
Which component contributes most?
Does alpha survive transaction costs?
Does alpha survive regime shifts?
Is performance statistically significant?
Can results be reproduced?

If you answer all ten convincingly, you're in Q1 territory.

My estimate
Current proposal as-is

Q1 acceptance probability:

20-35%

because it feels more like a sophisticated system implementation.

Reframed around event-aware fundamental fusion

Q1 acceptance probability:

50-70%

because now there's a clear research contribution.

Reframed around event-aware fusion + memory for regime persistence with strong ablations

Q1 acceptance probability:

70-85%

assuming results are actually strong.

Add what you typed and this below instrucuture to a folder, said folder you create and add these as md files
