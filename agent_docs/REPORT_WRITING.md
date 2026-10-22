# Writing experiment reports

User-approved reporting conventions, distilled from feedback on the agentic
experiments and the organization of [Predicting Future Behaviors in Reasoning
Models Enables Better Steering](https://arxiv.org/html/2606.11172), especially
Sections 3–4 and Appendices B–F. This guide is self-contained; agents do not need
to reread the paper. Apply it when writing or updating experiment reports.

## Rules

1. **Organize around a research question.** State what we want to learn and why
   this experiment answers it. Keep motivation short and specific.
2. **Make the setup sufficient to understand the results.** Briefly introduce
   data, method, comparisons and metrics. Readers should understand the findings
   without opening code or reproduction instructions.
3. **Describe the method conceptually.** For probes, explain their architecture,
   training data and target, input activations, and how predictions are obtained.
   State whether they are fixed or retrained. Module names and implementation
   terminology do not substitute for this explanation.
4. **Explain why each dataset is included.** Give relevant characteristics,
   evaluation size and relationship to training data. Explain distinctions such
   as on-policy versus off-policy in ordinary language.
5. **Define every reported metric before using it.** Explain what is counted,
   how it is averaged, and what better performance means. Token accuracy weights
   longer conversations more; mean conversation accuracy gives each equal weight.
   Explain denominators and how failed or excluded cases are treated.
6. **Introduce baselines explicitly.** Say what each baseline predicts and why
   it is informative. A majority-class classifier and predicting mean
   probabilities are different baselines. Put the relevant baseline beside the
   result rather than introducing it afterward.
7. **Keep main results selective.** Include only evidence needed to answer the
   question. Use readable column names, sensible precision and short explanations
   of the important comparisons. Secondary diagnostics belong in the appendix.
8. **Separate observation from interpretation.** First state the measured
   pattern with supporting numbers. Then explain what it suggests, what it does
   not establish, and how it answers the question. Do not narrate every table cell.
9. **Give each document a clear purpose.** Use the division below. Do not turn
   the main report into an execution log or inventory of settings.
10. **Explain consequential limitations in the main text.** Briefly state their
    effect on interpretation; put mechanics elsewhere. For example, explain
    that splitting long conversations removes earlier context from later
    segments. Put the exact segment size in reproduction details.
11. **Use short, connected paragraphs.** Descriptive labels such as **Data**,
    **Probes**, and **Metrics and baselines** can make the setup easy to scan.
    Avoid dense setting lists, unexplained jargon and repeated caveats.
12. **Perform a reader's check.** Using only the main report, can a reader explain
    what was tested, what every displayed number means, what the comparison
    shows, and the conclusion?

## Document responsibilities

| Location | Content |
|---|---|
| `README.md` | Motivation & RQ → conceptual evaluation setup → factual observations → message/conclusion. End with a brief reproduction/viewer pointer and a link to `RUNNING.md`. |
| `APPENDIX.md` | Confusion matrices, detailed breakdowns, examples, ablations and supporting analyses. Define supplementary metrics where they first appear. Link relevant sections from the main report. |
| `RUNNING.md` | Commands, environments, numerical precision/quantization, exact token budgets, seeds, paths, execution details and artifact/viewer generation. Link authoritative configuration and provenance. |
| Code/configuration/manifests | Exact revisions, hashes, complete settings and machine-readable provenance. Do not copy them into the main scientific narrative. |

Keep scientifically meaningful details—such as the compared layers, probe
architecture, sample sizes and train/evaluation separation—in the main setup.
This is a relevance rule, not a ban on numbers. Include implementation details in
the main text only when the experiment specifically studies them, and explain why.

## Application and maintenance

- A table must not introduce unexplained columns. Remove operational columns
  such as “windowed traces” from the main table unless they answer the question;
  define and report segmentation coverage in the appendix instead.
- Separate task performance from probe accuracy. State clearly when one does not
  measure the other. Do not bury an invalid benchmark comparison in reproduction
  notes merely to keep the main report short.
- Identify preparation-only, intermediate and historical reports, and link to
  the final experiment. Do not invent findings for an unexecuted experiment.
- Preserve measurements and provenance when reorganizing prose; move rather than
  discard details needed to audit results.
- Treat the scientific narrative as authored text. Regeneration may update
  metrics, tables, appendices and viewers, but should not silently overwrite the
  curated README or reproduction instructions with generic boilerplate.
