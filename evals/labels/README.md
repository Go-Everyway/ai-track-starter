# labels

Human-labelled examples for checking a grader (Module 10).

`outreach_labelled.jsonl` holds 40 outreach-angle outputs, one per line:
`{id, company_id, company, output, human: {verdict, failure_mode, reason}}`.

These were written for the course, not produced by a model. Each one targets a real
profile in `data/companies.jsonl` and is either good (16) or carries one deliberate
defect (21): invented evidence, a generic opener, too long, a planted link, the wrong
pain point, or high confidence for a weak fit. Three more are borderline cases where a
careful reviewer passes something a strict rule fails. The labels are our judgement;
you may disagree with a few, and that disagreement is the point of Module 10.

`make_outreach_labelled.py` rebuilds the file and checks that the code grader in
`lib/grader.py` behaves on each item the way it was designed to.

When you have a real model, label your own: run your step on 40 companies, judge each
output yourself before looking at any grader, and save the results in the same format.
