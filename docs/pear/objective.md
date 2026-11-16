# RL objective

## Episodes and reward

An episode is \(\tau = (o_1, y_1, \dots, o_T, y_T)\), where \(o_t\) is the observation of round \(t\)
and \(y_t\) the model's response (reasoning + action). The policy \(\pi_\theta\) generates each response
token given the context \(h_t = (p, o_1, y_1, \dots, o_t)\), with \(p\) the system prompt. The reward
is **binary**: \(R(\tau) = 1\) if the simulator reports success by the end of the episode, else 0.
Training maximizes \(\mathbb{E}[R(\tau)]\).

## Advantages: GRPO (Eq. 1)

Sample a group of \(G\) episodes \(\tau_1, \dots, \tau_G\) that share **a task and an initial state**:

\[
\hat A_i = \frac{R(\tau_i) - \operatorname{mean}_j R(\tau_j)}{\operatorname{std}_j R(\tau_j) + \epsilon}
\]

**Every generated token** of episode \(i\) receives \(\hat A_i\), whether it is reasoning or action,
in whichever round.

## Gradient: REINFORCE + score centering (Eq. 2)

Averaged over the \(N\) generated tokens of a batch:

\[
g = \frac{1}{N} \sum_i \hat A_i \sum_{t,k} \Big[ \nabla_\theta \log \pi_\theta\big(y^k_{i,t} \mid c^k_{i,t}\big)
 - \mathbb{E}_{v \sim \mu(\cdot \mid c^k_{i,t})} \nabla_\theta \log \pi_\theta\big(v \mid c^k_{i,t}\big) \Big]
\]

Here \(y^k_{i,t}\) is the \(k\)-th token of the round-\(t\) response of episode \(i\),
\(c^k_{i,t} = (h_{i,t}, y^{<k}_{i,t})\) is its context, and \(\mu\) is **the policy that generated the episode**.

- **First term: REINFORCE** (Williams, 1992) with the group mean as baseline, used **without PPO
  clipping**. The paper calls this "bypass mode".
- **Second term: score centering** (Marek & Ryabinin, 2026). Rollouts and updates run **fully
  asynchronously**, so \(\mu\) lags one or two versions behind \(\pi_\theta\). Score centering corrects
  for this by subtracting from each token's score function its expected value under the rollout
  policy's next-token distribution. PEAR uses it **alone, without truncated importance sampling**.

!!! info "Why the correction vanishes on-policy"
    \(\sum_v \pi_\theta(v)\,\nabla_\theta \log \pi_\theta(v) = \nabla_\theta \sum_v \pi_\theta(v) = \nabla_\theta 1 = 0.\)
    So when \(\mu = \pi_\theta\), the second term is exactly zero, and it only matters for stale rollouts.
    The same identity gives
    \(\mathbb{E}_{\mu}[\nabla \log \pi_\theta] = \sum_v (\mu_v - \pi_{\theta,v}) \nabla \log \pi_{\theta,v}\),
    which is how vlmrl evaluates it on \(\mu\)'s top-k tokens while staying exactly zero on-policy
    (see [Algorithm](../concepts/algorithm.md#score-centering)).

There is **no KL penalty and no entropy bonus**. One optimizer update is taken per step, and the
loss is the token-mean over the batch.

## DAPO group filter

With a binary reward, \(\hat A_i = 0\) for every member of a group in which **every episode
succeeds or every episode fails**. As in DAPO (Yu et al., 2025), such groups are **discarded and
replaced**, so each step uses 128 groups × 8 rollouts *counted after the filter*.

## Why a curriculum follows

A task the model never solves yields all-fail groups, which the filter removes, so it gives no
learning signal at all. At the start, the base model succeeds in ~2% of LIBERO Goal/Spatial/Object
episodes, 0.4% of LIBERO-Long, and almost never on most RoboDojo tasks. Training therefore starts
where there is *some* signal and relies on transfer for the rest (see [Recipe](recipe.md#curriculum)).
