> **Loss-mask bug (fixed 2026-11-18, commit `2e2b7d2`): full fine-tuning runs before the fix are affected.** The trainer's
> loss mask covered one token the model never generates: the newline that chat templates append after the end-of-turn
> token (`<|im_end|>\n`, `<turn|>\n`, `<end_of_utterance>\n`). Sparse rewards and long failed episodes make the
> token-weighted advantage net-negative, so every update pushed that token's log-prob down. It is never sampled, so
> nothing pushed back, and it ran away to **-108** (MiniCPM-V). Its gradient then dominated (norms 20-185), and replies
> drifted into prose or junk. That drift produced the **collapses to ~100% invalid replies** below (MiniCPM-V and
> SmolVLM2-2.2B at lr 5e-6, SmolVLM-256M at lr 3e-5) and the falling `train/mean_logp` while vLLM's own sampled
> log-probs stayed near -0.3. PEAR (veRL) trains exactly the generated token ids, so it never hit this and needed no
> KL. The mask now ends at the stop token. Every other run in this report started before the fix, including the SmolVLM
> runs, so their late-training drift and the KL ablations are partly this bug. Reruns: "after the mask fix" panels.
