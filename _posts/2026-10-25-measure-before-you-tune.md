---
layout: post
title: "Measure before you tune: an honest evaluation for video RAG"
date: 2026-10-25 10:00:00 +0530
author: Sujeet Srivastava
description: "A retriever that found the evidence for 39% of dev questions fell to 13% on held-out ones. How the evaluation harness for a local video RAG was built, and why the held-out number is the one to believe."
tags: [rag, video, evaluation, retrieval, multimodal]
---

A retriever I was about to call the winner found the right scene for **39%** of my development questions. On questions it had never seen, it found **13%**.

| Evidence in the top 5 | *Charade* dev | *Charade* held-out | *Destroy All Planets* held-out |
| --- | --- | --- | --- |
| image search (`image_mean`) | **39%** | 13% | 19% |
| dialogue search (`text`) | 18% | **37%** | **36%** |

Plain dialogue search, which I had nearly written off, held steady.

The system is a video RAG. It answers questions about a full-length film by finding the moments that matter, in the dialogue and in the frames, and handing them to a model with the timecodes it used. It runs on one Mac Studio with open models.

This post is about the evaluation that caught that reversal, and why I now treat the evaluation harness, not the retriever, as the real deliverable of a RAG project. Once you can say "this change moved 2 of 28 questions, and here are the floor and the ceiling," every chunker, embedder and fusion rule becomes a cheap experiment. Without that, every change is a guess.

Every number below comes with its sample size. The samples are small, and I say so where it matters.

* TOC
{:toc}

## Why measure this carefully

A media company's catalogue is described at the level of titles: genre, cast, synopsis. Anything *inside* a film is found by someone scrubbing footage. Editors tag by hand, promo producers hunt for moments, and standards teams re-screen for regional edits. An index over the content itself, built once at ingest, could serve all of them. Grounding answers in timecoded segments keeps them tied to footage someone can check.

That only pays off at catalogue scale. There, every pipeline stage is a cost multiplied by every title and every question: describing every frame of one film with a vision model took 2.5 hours here. So before anything moves to the cloud, each stage has to earn its place. The same frozen question sets then become the regression test when a local model is swapped for a hosted one. Section 8 covers how the pipeline is built to make that move without a rewrite.

## The setup

The system answers questions about full-length films. It runs locally on a Mac Studio: the film is cut into 60-second windows, each window gets a dialogue vector (Whisper transcript, Qwen3-Embedding-0.6B) and frame vectors (one frame every 2 seconds, SigLIP2), and a question is matched against both. The test material is public-domain film: *Charade* (1963) for development, and *Destroy All Planets* and *Out of This World* as second films. Only questions that genuinely need the video are scored; anything a viewer could answer without watching is kept on the sheet but reported separately.

The code, node contracts, question sheets and every result behind the numbers in this post are on GitHub: [sujeet-srivastava/video-rag](https://github.com/sujeet-srivastava/video-rag). The films themselves are not included; `data/SOURCES.md` lists where to get them.

## 1. Ground truth is the silent work

Three films have question sheets:

| Film | Dev questions | Held-out questions | Used for |
| --- | --- | --- | --- |
| *Charade* | 53 (28 video-dependent and scorable) | 30 | building and tuning; a frozen test set |
| *Out of This World* | 12 | — | fixing choices before any held-out run |
| *Destroy All Planets* | — | 31 | an untuned second film |

Every question carries a time-stamped evidence span and three tags: **modality** (visual, spoken or both), **scope** (a single moment, several pieces, or unanswerable) and whether it can be answered **without watching**. The shape follows V-RAGBench, which pairs each question with an evidence span and screens out questions answerable without the video.[^vrag] The single-moment versus whole-video split comes from MovieChat-1K[^moviechat] and the visual versus spoken split from Video-MME.[^videomme] Multi-piece evidence, unanswerable questions and full-length films are additions for this project.

Writing a question is a loop: watch the scene, scrub back, rewatch, timestamp the main evidence and any extra pieces, then check the question against a list of criteria (is it answerable from the span alone, does it leak the answer, could you answer it from general knowledge of the film). Thirty questions for *Destroy All Planets* took about four hours, roughly eight minutes each.

That is the most expensive artefact in the project and the least visible one. Every metric that follows depends on it, so budget for it like engineering work, not like an afterthought.

## 2. Evaluation as service tests

I treat the evaluation the way I would treat service tests in a backend: every stage has a check that must pass before its numbers mean anything.

**The oracle is the ceiling, and it must score 1.0.** It ranks one correct chunk per evidence piece first. If it scores less than perfect, the bug is in evidence parsing, chunk mapping or scoring, not in a retriever. That is enforced in code:

```python
assert (SUMMARY.query("retriever == 'oracle'")["recall@5"] == 1.0).all(), \
    "oracle must be perfect: scorer bug"
```

The oracle also documents the metric. Its mean reciprocal rank on multi-piece questions is 0.5 by construction, because MRR here uses the rank at which the *last* required piece appears, and with two pieces the best possible rank is 2.

**The random floor moves with the chunking.** It is the mean of 200 seeded shuffles, reported with its 5–95% range, because a single shuffle can land anywhere from 0% to 10%. The floor depends on how many chunks there are and how many count as correct:

| Chunking | Random recall@5 |
| --- | --- |
| scene chunks (266, ~28 s) | 0.019 |
| overlapping 60 s windows every 30 s | 0.037 |
| overlapping 90 s windows every 45 s | 0.057 |
| fixed 120 s windows | 0.080 |

Bigger chunks make recall@k easier and less informative. A retriever is judged against the floor of its own chunking, never against another chunking's floor.

**Two guardrails on the path to improvement:**

1. **Matched reference.** A new chunker has to beat fixed windows of the same mean size, averaged over 50 start offsets. This is the check that showed scene detection was offset luck (section 3).
2. **Pre-registered gain rule.** A change counts as a gain only if it moves at least two video-dependent questions and shows up on the subset it targets. For LLM steps, the prompt is frozen and hashed before the eval, and the eval runs once.

**Supporting checks.** The retriever only ever sees the question text; evidence timestamps are the answer key and join only at scoring time. Vector quality is checked with neighbour rank and a z-score rather than raw cosine (more on that in section 4).

**Held-out discipline.** Test sheets are frozen and their hash is written into every run record. Design choices, such as which frames to keep, are fixed on a separate dev film first. The held-out sets are then run once.

## 3. Chunking: size beat placement

The first pipeline used scene detection, on the reasoning that scripted film has natural boundaries. Before comparing chunkers, I had to separate label noise from real failures.

**Rounding is not a bad cut.** Evidence is annotated to the whole second, so a 0.2 s sliver of evidence in the next chunk is rounding. Ignoring slivers under 1 s dropped the count of questions whose evidence was cut by a chunk boundary from 13 to 11 (of 20 at the time). Going further would hide real splits: one 7 s moment was cut 3.6 / 3.4.

**Scene detection did no better than dumb windows.** Fixed 29 s windows, the same mean size as the scene chunks, cut 11.7 of 20 on average across start offsets, against 11 for scene chunks, well inside the offset range of 9 to 15. Watching the cuts showed why: the detector cuts at shot / reverse-shot changes *inside* a conversation, because alternating close-ups look different, and the `min_scene_seconds` setting decided most chunk lengths.

**What matters is size, and overlap.** The median evidence moment is about 20 s. With chunks of about 28 s, a boundary usually lands inside the moment wherever you put it. Overlapping windows fix that with a simple rule: with windows **W** seconds long starting every **S** seconds, any moment up to **W − S** seconds long fits whole inside at least one window.

![A 22-second moment at 48–70 s is cut by fixed 60 s windows; with overlapping 60/30 windows, the 30–90 s window holds all of it.]({{ '/assets/measure-before-you-tune/overlap-windows.svg' | relative_url }})

On the 50-question sheet, overlapping 60/30 windows cut **none** of the 39 moments of 30 s or less, at every start offset. All remaining cuts were moments longer than 30 s. Sending the neighbouring window on each side to the answering model fixed those too, with no extra embeddings: nothing was left cut in the answer context.

The change had a contract consequence. The chunk node's post-validation used to check that chunks were *contiguous* (no gaps, no overlaps). With overlapping windows it now checks *coverage*: every second of the film is in at least one chunk.

## 4. Retrieval: the dev winner did not hold

Here is evidence in the top 5, counted in questions, on all three sets:

| Retriever | *Charade* dev | *Charade* held-out | *Destroy All Planets* held-out |
| --- | --- | --- | --- |
| random floor (expected) | 1.0 / 28 | 1.2 / 30 | 1.6 / 31 |
| `text` (dialogue) | 5 / 28 | **11 / 30** | 11 / 31 |
| `image_mean` (mean of frame vectors) | **11 / 28** | 4 / 30 | 6 / 31 |
| `desc_max+nms` (frame descriptions) | **11 / 28** | 3 / 30 | 9 / 31 |
| `fused3` (text + image + descriptions) | **11 / 28** | 8 / 30 | **12 / 31** |
| `fused3+nms` | **11 / 28** | 7 / 30 | 11 / 31 |
| oracle | 28 / 28 | 30 / 30 | 31 / 31 |

![Recall@5 by retriever on the three question sets. Image and description retrievers lead on Charade dev; dialogue search leads or ties on both held-out sets.]({{ '/assets/measure-before-you-tune/dev-vs-heldout.svg' | relative_url }})

Images beat text only on the dev set. On both held-out sets, plain dialogue search is best or within one question of best. Two things moved, and splitting by question type shows both:

- **The question mix changed.** On spoken questions `text` wins on every set (recall@5 of 0.55 on *Charade* dev, 0.54 on *Charade* held-out, 0.38 on *Destroy All Planets*), while `image_mean` scores 0.08 to 0.18. The held-out sheets are about half spoken (13 of 30 and 16 of 31). The dev sheet is not (11 of 50).
- **Images also got worse on visual questions.** They still lead there, but the gap shrinks from 0.55 against 0.05 (*Charade* dev, n = 23) to 0.30 against 0.20 (*Charade* held-out, n = 10) and 0.33 against 0.22 (*Destroy All Planets*, n = 9). On *Destroy All Planets*, descriptions and fusion do best on visual questions (0.44).

These groups are 6 to 16 questions, so one question moves a cell by 6 to 17 points. The dev split also covers all 50 dev questions, including those answerable without the video. The direction is clear; the exact sizes are not.

**Fusion is not free.** Reciprocal rank fusion only looks at ranks, so a weak list can outvote a strong one. On *Charade* held-out, `fused3` loses to `text` (8 against 11). On *Destroy All Planets* it leads by one question, which is noise: 5 questions won, 6 lost, with a 90% interval for the difference of −0.23 to +0.13.

**Non-maximum suppression helped on dev and hurt held out.** Removing near-duplicate overlapping windows from the top 5 lowered `fused3` on both held-out sets (0.267 to 0.233 and 0.387 to 0.355). It fails badly when the evidence is split across two windows: a better-ranked neighbour that holds too little of the evidence pushes the correct window to the back.

**A note on vector checks.** SigLIP cosines for one film sit in a narrow band near 0.9, which made the image vectors look as if they did not track content (neighbouring windows 0.98, random pairs 0.90). Cosine has no true zero, so that ratio is misleading. Measured by rank instead, a window's time neighbour is its nearest vector for 87% of image windows and 95% of dialogue windows, against 0.8% by chance; removing the film's mean vector lifts the image z-score to the dialogue level. Ranks are all retrieval uses, so the compressed scale does not hurt it, but a fixed similarity threshold would not transfer between the two models.

## 5. Frame descriptions: turning pixels into words

The misses on the dev set had a pattern: small printed text and small objects. A label reading "A4" on a rack, a city name on a pillow, the colour of a hat. A single 768-number vector per frame summarises the scene and loses the detail. So I had a local vision-language model (Qwen3.6 35B, 4-bit, through Ollama) describe every frame in under 100 words, embedded the descriptions as text, and scored each window by its best-matching description.

On *Charade* dev (28 video-dependent questions), `desc_max+nms`, `fused3` and `fused3+nms` reached 0.393, tying `image_mean`, and the description retriever recovered 3 of the 9 video-dependent questions every earlier retriever had missed. It read "Los Angeles" off the pillow without any OCR step.

Held out, the gain mostly disappeared: `desc_max+nms` scored 0.100 on *Charade* and 0.290 on *Destroy All Planets*, below `text` on both. The cost was about 2.5 hours of one-off indexing compute for one film.

The limits are visible in the text. A description sees one frame, so it misses actions. A stream of water became a "laser beam". In one *Destroy All Planets* question a claw grabbing a rock became "hands gripping a rock", and the rock in the creature's mouth became a "tongue". A wrong description is searched as confidently as a right one.

## 6. A negative result, reported in full: LLM dialogue summaries

**Hypothesis.** Raw dialogue is full of pronouns. A summary such as "the woman tells Sylvie that Charles is hiding something" should sit closer to a question's wording than the raw lines.

**Verdict: no gain.** The summary retriever tied plain dialogue at 0.179 recall@5 on the *Charade* dev video-dependent questions. The best fusion using it was one question ahead of the previous best, which the pre-registered rule counts as noise. On the three spoken questions that motivated the experiment, the first correct window still ranked between 41 and 103 of 253.

**Faithfulness was the hard part.** The summariser was told to use only the dialogue it was given, and a post-validation step checked every name and capitalised word against that text, with a 2% warn and 10% stop threshold. The first prompt flagged 28.6% of windows and the run stopped before any retrieval was scored. Some flags were the checker's own false alarms, but the real ones were instructive: the model copied names from the few-shot example into its summaries and respelled names from the film. The second prompt brought the flag rate to 6.2% and the eval ran once.

One honest caveat: the second prompt was written after reading the first run's flagged windows, some of which sit next to question evidence. The changes targeted faithfulness only, but the protocol was not fully blind. I would rather say that than have someone find it.

## 7. End to end: where the answers are lost

Retrieval recall is a proxy. The end-to-end eval asks the answering model each held-out question with different contexts and has a judge mark the answer correct or not:

| Context given to the answering model | *Charade* held-out | *Destroy All Planets* held-out |
| --- | --- | --- |
| nothing (closed book) | 1 / 30 | 3 / 31 |
| top retrieved dialogue windows | 10 / 30 | 11 / 31 |
| the perfect windows, with frames picked by relevance (default) | 13 / 30 | 17 / 31 |
| the perfect windows, with frames from the evidence span (answer key) | 21 / 30 | 21 / 31 |

The ladder shows where the points go. Closed book is near zero, so the questions really do need the film. Perfect retrieval recovers only part of the gap. The last two rows send the same number of images (up to 8) and differ only in *which* frames. Picking them from the evidence span instead of by relevance is worth another 4 to 8 questions. So two things lose answers: retrieval finding the wrong windows, and the frame choice inside the right windows missing the moment.

The judge and a human review agreed on 396 of 403 *Destroy All Planets* answers. The judge scores correctness, not grounding: in one case the dialogue retriever produced "a large rock", which was correct, with no evidence for it in its context.

### See it run

The demo runs on *Destroy All Planets*: ask a question, see the retrieved windows with their timestamps, then the answer.

<video controls preload="metadata" width="100%">
  <source src="{{ '/assets/measure-before-you-tune/demo_1.mp4' | relative_url }}" type="video/mp4">
  Your browser does not play embedded video. <a href="{{ '/assets/measure-before-you-tune/demo_1.mp4' | relative_url }}">Download the demo (MP4)</a>.
</video>

## 8. Contracts that carry into production

The experiments ran in notebooks, but every stage was written as a node with a contract, so the same checks carry into a deployed pipeline. Each node has a YAML contract covering its input and output schema, runtime and compute limits, timeouts and cost guardrails, retries and a circuit breaker, and two parts I find most useful:

- **Cross-field constraints** that encode rules across settings, not just types (an example follows).
- **Contextual limits** that scale timeouts and budget with input size.

A schema can check that `window_seconds` and `stride_seconds` are positive numbers. It cannot check that they make sense *together*. Two constraints from the chunk node's contract (`contracts/indexing/chunk.yaml`, abridged) do that. The first rejects a stride longer than the window, which would leave parts of the film in no chunk at all. The second turns the W − S rule from section 3 into a warning, so a config change that quietly breaks the whole-moment guarantee is flagged before anything is indexed:

```yaml
cross_field_constraints:
  - id: VAL-CHUNK-102-STRIDE-WITHIN-WINDOW
    description: Stride must not exceed the window, or parts of the video fall between windows.
    condition:
      expression: "params['stride_seconds'] > params['window_seconds']"
    action: reject_contract
    remediation: Set indexing.chunking.stride_seconds <= window_seconds.
  - id: VAL-CHUNK-103-WHOLE-MOMENT-GUARANTEE
    description: >
      window - stride is the longest moment guaranteed to sit whole in one chunk;
      the median evidence moment in the question sheet is 20 s.
    condition:
      expression: "params['window_seconds'] - params['stride_seconds'] < 20"
    action: throw_warning
    remediation: window - stride is under 20 s, so typical evidence moments may be cut;
      widen the window or shorten the stride.
```

With 60/30 the margin is 30 s and both pass. Change the stride to 50 s and the node still runs, but warns that only moments up to 10 s are now guaranteed to fit. Set it to 90 s and the contract is rejected. The useful pattern is that an eval finding (the median evidence moment is 20 s) becomes a guard on the config, rather than a fact someone has to remember.

Each node also has a pre-validation step (is the previous node's output fit to be my input?) and a post-validation step (does my output meet my own standards?). Both write result files and decide stop, warn or proceed. The dialogue summariser's 2% warn and 10% stop thresholds on flagged chunks, from section 6, are one example.

The pipeline is event-driven. Every stage is a file-in, file-out task, started when a file lands in an inbox folder, with events shaped like the notifications Cloud Storage sends when an object is uploaded. Locally that folder is a directory; in production it becomes a bucket, and each stage runs as its own container on Cloud Run. Each node is a thin callback, so heavy steps such as frame description and embedding can move there while local development runs on sample assets. The seam for that is in the code today, though the Cloud Run runner itself is not built yet. The orchestrator supports cycles, with loop-safety thresholds, even though today's flow is linear.

## 9. Limitations

- **Small samples.** The held-out sets are n = 30 and n = 31, so one question is about 3.3 points; *Charade* dev has 28 video-dependent questions (3.6 points). The winning retriever changed between dev and held-out, so no signal can be called best yet. The frame selection did the same: `peak` beat `uniform` 7 to 3 on the dev film and tied it 20 to 20 on *Destroy All Planets*.
- **The dev sheet was reused for design.** It shaped the frame-description pass, so its dev score (0.393 for `fused3+nms`) was optimistic. Held out, the same retriever scored 0.233 and 0.355.
- **Some evidence spans are a whole scene** while the answer takes a few seconds, so scores on long moments are an upper bound.
- **The transcript stage has its own error rate**, not measured separately. Whisper large-v3-turbo with voice activity detection off writes filler over music: on *Destroy All Planets*, "Let's go" 22 times, "Thank you" 15 times and "The End" 8 times, all with a no-speech probability of 0.0, so the filter kept them. It also mishears key words ("cab" for "camp", "Camera" for "Gamera"). Results are reported as run; testing the transcript stage is the subject of a follow-up.
- **One answering model, one embedder of each kind, one chunking** on the held-out runs.

## 10. Takeaways

1. **Build the floor and the ceiling before the retriever.** An oracle that must score 1.0 catches scorer bugs; a random floor per chunking tells you what "good" means.
2. **Compare against matched dumb baselines.** Scene detection looked fine until it met fixed windows of the same size.
3. **Pre-register what counts as a win**, freeze prompts, and run the eval once.
4. **Budget real human hours for ground truth.** Eight minutes a question is the cost of every number in this post.
5. **Write contracts that encode cross-parameter rules**, not just schemas, and let validation decide stop or proceed.
6. **Believe the held-out set over the dev set.** A dev winner can be a property of the questions you happened to write.
7. If you want to reuse the harness (the oracle and random references, the answer ladder, the contracts), start with the [repository README](https://github.com/sujeet-srivastava/video-rag#readme).

---

[^vrag]: Lee et al., "Rethinking RAG in Long Videos: What to Retrieve and How to Use It?", arXiv preprint [2606.13141](https://arxiv.org/abs/2606.13141), June 2026.
[^moviechat]: Song et al., "MovieChat: From Dense Token to Sparse Memory for Long Video Understanding", CVPR 2024. [arXiv:2307.16449](https://arxiv.org/abs/2307.16449).
[^videomme]: Fu et al., "Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis", CVPR 2025. [arXiv:2405.21075](https://arxiv.org/abs/2405.21075).
