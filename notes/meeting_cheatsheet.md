# Meeting cheat sheet — 2026-08-19, 21:30 LA

## 0. Opening (30 sec)

> Thank you for meeting. Let me first say what the code does, then what in the paper
> is affected and what is not, and then the two options. Please stop me at any point.

If asked "why did you not say earlier":

> I did not realize it. I implemented h_i as the events from neutrinos above 2 MeV,
> following Fig. 2 and Ref. [6], and I did not check that this is not the h_i that
> goes with Sum_j R_ij Phi_j in Eq. 5.4. I saw it only when I traced the code after
> Danny's remark about the vertical scale. I should have checked the correspondence
> between the code and the theory section explicitly. I am sorry.

(Say it once. Then move to facts.)

---

## 1. The one-sentence statement

> Eq. 4.7's boundary term vanishes on [0, infinity), but not on [0, 2 MeV].
> That term, Phi(2 MeV) * Sum_j R_ij, is inside my h_i, and it is what turns the
> minimization variables from Phi_j into DeltaPhi_j.

## 2. Two definitions of h_i (both exact, they differ by the boundary term)

  h_i^{Fig.2}  = Eq. 4.6 integrated from 2 MeV to infinity   (events from E_nu > 2 MeV)
               = "CEvNS contribution due to neutrinos with E_nu > 2 MeV" (text below Fig. 2)
               = h_j of Ref. [6], Eq. 6;  green line of Fig. 2;  matches Danny's Fig. 3
               --> goes with signal Sum_j R_ij DeltaPhi_j   <-- THIS IS WHAT THE CODE USES

  h_i^{Eq.4.9} = Eq. 4.9 integrated from 2 MeV to infinity   (Phi-weighted response)
               --> goes with signal Sum_j R_ij Phi_j        <-- THIS IS WHAT Eq. 5.4 NEEDS

  Relation:   h_i^{Fig.2} = h_i^{Eq.4.9} + Phi(2 MeV) * Sum_j R_ij

  Total is identical either way:
      N_i^tot = Sum_j R_ij Phi_j      + h_i^{Eq.4.9}
              = Sum_j R_ij DeltaPhi_j + h_i^{Fig.2}

## 3. Numbers to have ready (1 eV threshold, counts/(kg yr) per bin, solid flux)

| bin [eV] | Sum R_ij DPhi | Phi(2)SumR | Sum R_ij Phi | h^{Eq.4.9} | h^{Fig.2} | N^tot |
|----------|--------------|-----------|-------------|-----------|----------|-------|
|    1--3  |   232.5      |    96.9   |    329.4    |    0.8    |   97.7   | 330.2 |
|  11--13  |   121.9      |    92.3   |    214.2    |    4.9    |   97.3   | 219.2 |
|  21--23  |    76.1      |    83.7   |    159.8    |    9.1    |   92.8   | 168.9 |
|  41--43  |    33.0      |    66.5   |     99.5    |   17.3    |   83.8   | 116.8 |
|  51--56  |    52.0      |   141.5   |    193.5    |   55.1    |  196.6   | 248.6 |
|  81--120 |    32.7      |   316.7   |    349.3    |  806.5    | 1123.1   |1155.8 |

  Phi(2 MeV) = 4.59e11 cm^-2 s^-1 = 24% of Phi(0.18 MeV) = 1.88e12
  N^tot agrees with Danny's Fig. 3 to ~1%.
  Key line: in the 1-3 eV bin, h^{Eq.4.9} = 0.8 but h^{Fig.2} = 97.7 — almost all
  of the >2 MeV rate IS the boundary term.

## 4. What in the paper is NOT affected  (say this early — it is the calming fact)

  - Section 2 entirely: Eqs. 2.1-2.6, HI method, convex geometry, Fenchel-Eggleston,
    finite Dirac sum, Neyman chi^2.
  - Section 3 entirely: cross section, resolution, efficiency, detector setup, binning.
  - Section 4 essentially entirely: Eqs. 4.1-4.9 are correct as written. In particular
    Eq. 4.7 and the statement that the boundary term vanishes are correct — they refer
    to the full domain [0, infinity). Eqs. 4.10-4.12, Fig. 3.
  - Section 5 framework: ansatz Eq. 5.1, R_ij Eq. 5.3, chi^2 Eq. 5.7, constraints
    Eq. 5.8, choice of N_int / Fig. 6.
  - Appendices A, B, C (including the MC band procedure).
  - The code beyond this issue: same R_ij, same N_int, same solver, same MC.

## 5. What IS affected

  1. The sentence defining h_i (below Fig. 2) and Eq. 5.4 — they currently use two
     different definitions of h_i.
  2. The step from Eq. 4.9 (infinite domain) to Eq. 5.2 (domain cut at 2 MeV):
     the boundary term at 2 MeV must be written explicitly. This is new text,
     a few lines, not a rewrite of the formalism.
  3. Figures: Fig. 2 (cyan/green) if we go to Phi; flux-figure axes and theory curves
     if we go to DeltaPhi.
  4. The comparison with Ref. [6].

## 6. The two options

  (a) FIT Phi, as Eq. 5.4 is written.
      - signal = Sum_j R_ij Phi_j = (present signal) + Phi(2 MeV) Sum_j R_ij
      - h_i = h_i^{Eq.4.9}
      - ALL fits and confidence bands recomputed (1 eV and 5 eV, all scenarios).
      - Fig. 2 changes a lot: green becomes almost zero at low E', cyan approaches black.
      - The h_i sentence must be rewritten: h_i is then NOT "the events from neutrinos
        above 2 MeV"; it is the Phi-weighted response above 2 MeV.
      - Comparison with Ref. [6] must be redone.
      - Cost: recompute bands. [fill in: ~X hours/days of CPU]

  (b) KEEP the numerics, rewrite in DeltaPhi.
      - Eqs. 5.1, 5.2, 5.4, 5.7, 5.8 and the flux figure axes in terms of
        DeltaPhi_j = Phi_j - Phi(2 MeV).
      - Constraints still hold: DeltaPhi_j >= 0 and non-increasing (since Phi is
        non-increasing and Phi(2 MeV) is a constant).
      - h_i, Fig. 2, all fits and all bands unchanged.
      - Theory curves in the flux figures shifted down by Phi(2 MeV).
      - Physics goal survives: the NC/no-NC discrimination is unaffected, because the
        difference between the two curves is unchanged by a constant shift.

  My recommendation: (a), because the paper and the physics question are stated for
  Phi(E_nu). But (b) is legitimate and much cheaper; the choice is ours.

## 7. Questions I expect

  Q: Is the boundary term really not in the paper?
  A: Eq. 4.7 has the boundary term and correctly says it vanishes — for the full domain.
     The paper never writes the split at 2 MeV as an integration by parts, so the
     boundary term at 2 MeV never appears. That is the gap.

  Q: What is the function multiplying Phi(2 MeV)?
  A: dH/dE' at E_nu = 2 MeV, integrated over the bin. By construction that equals
     the row sum of the response matrix, Sum_j R_ij. It is in the paper, in Fig. 5.

  Q: Was there a problem fitting Phi that made you switch?
  A: No. There was no switch. The choice of h_i determined the variable.

  Q: Is the code otherwise correct?
  A: The chi^2 is Eq. 5.7, constraints Eq. 5.8, bands per Appendix C. N^tot agrees with
     Danny's Fig. 3. I verified the boundary-term identity numerically to better than 1%.

  Q: How long to redo everything?
  A: [fill in before the meeting: band MC time per scenario x number of scenarios]

## 8. If I do not know something

  > Let me check that and come back to you today.

  Do not improvise a number or a claim in the meeting.

---

## 9. Handling strong language and interruptions

### Frame (for me, not to say out loud)
Strong words = her anxiety about a public paper, not a verdict on me.
I am the corresponding author. My job in this meeting is to run it and to decide.
Do not match the temperature. Slow speech beats loud speech.

### Announce the roadmap FIRST — this is the best defence against interruptions

> Before we start: I would like to go through four points — (1) what the code
> computes, (2) what in the paper is not affected, (3) what is affected,
> (4) the two options and my recommendation. It should take ten minutes.
> Please hold questions that fit a later point, and I will come back to anything
> I miss.

Then interruptions have a place to land: "That is point 3 — may I finish point 2 first?"

### Phrases to reclaim the floor (polite, firm, short)

  - "May I finish this point? It is two more sentences."
  - "Let me answer that in a moment — it belongs to point 3."
  - "I want to make sure I answer this precisely. Can I show you the number?"
  - "Can I share my screen? It is easier to see the two definitions side by side."
  - "Let me put that on the list so we do not lose it." (then actually write it down)

### If the tone gets personal

One acknowledgement, then back to substance. Do not defend twice.

  - "I understand this is serious, and I take responsibility for it.
     Let me show exactly what is affected so we can decide what to do."
  - "That is fair. I should have checked it. What I can do now is X."

If it repeats:

  - "I have apologised for this and I will not repeat it — can we use the time
     to decide between the two options?"

### If I am cut off before the key fact

Get THIS sentence out early, even out of order — it is the fact that reduces the panic:

> Sections 2, 3 and Eqs. 4.1-4.9 are correct as written. What is inconsistent is
> the definition of h_i below Fig. 2 versus Eq. 5.4, and the missing boundary term
> when the domain is cut at 2 MeV.

### If the discussion spirals

  - "Can we separate two questions: is the calculation correct, and what do we
     write in the paper? I suggest we settle the first one first."
  - "Should we look at the note together? Page 2 has the derivation."

### If I cannot get through it verbally

  > I will send a written summary of what we decide tonight.

  As corresponding author I own the written record. Whatever gets muddled in
  speech, I fix in the follow-up email.

### Decisions I want out of this meeting (do not leave without them)

  1. Option (a) fit Phi, or (b) rewrite in DeltaPhi.
  2. Timeline for the recomputation (if (a)) and for arXiv v2.
  3. Whether the journal editor needs to be notified, and who writes.
  4. Who rewrites which part of the text.
