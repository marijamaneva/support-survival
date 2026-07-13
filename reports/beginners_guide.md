# What this project actually does (a beginner's walkthrough)

This is the same project as `model_card.md`, told a different way: no jargon left
unexplained, no numbers dropped in without saying what they mean. If you've never
built a machine learning model before, start here. If you already know the field,
`model_card.md` is the denser, faster version of the same story.

## The question, in one sentence

Given some basic information about a seriously ill hospital patient (age, blood
pressure, whether they have cancer, and so on), can we estimate how likely they are
to die during the hospital stay, and roughly how soon?

That's it. Everything else in this document is about doing that carefully, honestly,
and in a way you could defend if someone pushed back on every number.

## The data, in plain terms

The dataset is called **SUPPORT**. It's real (anonymized) information about 8,873
seriously ill patients from five US hospitals in the 1990s. For each patient we know:

- 14 pieces of clinical information: age, sex, whether they have diabetes, their
  blood pressure, heart rate, and so on.
- Whether they died during the study (`event = 1`) or not (`event = 0`).
- How many days they were tracked for (`time`).

That last point hides something important, so it gets its own section.

### Why "died or not" isn't the whole story

Imagine a study that runs for two years. Patient A dies on day 10. Patient B is
still alive when the study ends on day 730. Patient C moves to another city on
day 200 and the researchers lose track of them.

Patient A: we know exactly what happened. Patients B and C: we don't know if or
when they died after the point we stopped observing them. This is called
**censoring**, and B and C are "censored," not "cured." A common beginner mistake
is to treat censored patients as if they survived forever, which is wrong; all we
actually know is that they were alive up to the last day we saw them.

In this dataset, 68% of patients died during observation and 32% were censored.
Handling that 32% correctly, instead of pretending they're all survivors, is the
central technical challenge of the whole project, and it's why a special kind of
statistics ("survival analysis") shows up later.

## Phase 1: Look before you touch anything

Before building anything, the first step was just looking at the data: how many
patients, how many died, what the typical values look like for each measurement.

One useful trick: look for measurements that are *impossible*, not just unusual.
A blood pressure of exactly 0 doesn't mean "very low blood pressure," it means
"this patient's blood pressure was never recorded," and it got stored as a zero by
mistake somewhere upstream. Fifty-one patients had this. Catching this kind of
thing early matters, because if you don't, the model quietly learns "a blood
pressure of 0 is normal," which is nonsense.

Also in Phase 1: splitting patients by whether they had cancer (none, present, or
metastatic/spread) and comparing how long each group tended to survive. The three
groups looked meaningfully different, which was a hint that "cancer status" would
turn out to matter a lot later. It did.

## Phase 2: Cleaning up, and a real mystery

Real datasets are messy, and this one had two problems worth explaining.

**The zero problem, again.** Beyond blood pressure, three more measurements
(heart rate, breathing rate, white blood cell count) had the same "0 means
missing, not actually zero" issue. Each of these zeros was replaced with a
proper "missing" marker instead of a fake measurement. But *why* a value is
missing can matter too, so instead of throwing that information away, a second
piece of information was kept alongside it: "was this value missing, yes or no."

**The swapped columns.** This one took real detective work. Looking closely at
the actual numbers in the "white blood cell count" column, they didn't look like
white blood cell counts at all. A normal white blood cell count is somewhere
between 4 and 11 (in the units doctors use). This column's numbers ranged from
110 to 181. Meanwhile, the column labeled "sodium level" had numbers that looked
exactly like a textbook white blood cell count. The conclusion: the two columns
had been swapped by whoever originally prepared this data file, years before this
project started. This was confirmed with an outside source (a medical reference
book's default value for a missing white blood cell count matched almost exactly)
and fixed by swapping the labels back, with a permanent automated check added so
this mistake can never silently creep back in.

This is the single most important lesson from the whole project: **never trust a
column name. Check what's actually inside it.**

**Six new columns were added**, each representing a well-known medical rule of
thumb rather than something invented for this project: is the patient over 70
(frailty risk), is their kidney function reduced, is their heart racing, is their
blood pressure dangerously low, do they have an unusually high white blood cell
count, is their sodium level abnormal. These give the model some pre-digested
medical knowledge instead of making it rediscover basic clinical thresholds from
scratch.

**One column was removed entirely: `race`.** The original source data stored a
patient's race as a word ("White," "Black," and so on), not a number. Whoever
converted this dataset into the number-only format used here had to invent their
own numbering scheme, and never wrote down what the numbers mean. Worse, the
numbers in this specific file don't even match the categories the documentation
describes (ten different codes instead of five, and the most common one covers
only a third of patients, too small to plausibly be "White" in a 1990s American
hospital). Rather than guess and risk being wrong about something as sensitive as
a patient's race, the decision was to simply never use that column for
predictions. Checking afterward, this cost almost nothing: the model's accuracy
barely changed without it.

## Phase 3: The first two models

With clean data in hand, two different approaches were tried to answer the
simplest version of the question ("did this patient die: yes or no"):

- **Logistic regression**: a simple, transparent method that's been used in
  medicine for decades, easy to explain to a doctor or a regulator.
- **Gradient boosting**: a more powerful, more flexible method that can pick up
  on complicated patterns, at the cost of being harder to fully explain.

They were compared fairly (same patients used for testing each one, five
different ways) using a score called **AUROC**, which is easiest to understand
like this: pick one patient who died and one who didn't, at random. AUROC is
the probability the model correctly guesses which one was higher risk. A score
of 0.5 means the model is no better than a coin flip; 1.0 means it's always
right.

- Logistic regression: 0.643
- Gradient boosting: 0.698

Gradient boosting wins, but neither number is close to 1.0, and that's honest and
expected: predicting death from a handful of vital signs is genuinely hard, and a
tool claiming near-perfect accuracy here should make you suspicious, not
impressed.

**Calibration** is a second, equally important idea that's easy to overlook.
AUROC only asks "does the model rank patients correctly?" Calibration asks "when
the model says 40% risk, do roughly 40% of those patients actually die?" A model
can rank patients perfectly while still being wildly overconfident or
underconfident about the actual number. Both models here were checked for this,
and both turned out to be reasonably calibrated (predicted risk roughly matches
what actually happened, within about five percentage points).

## Phase 4: Predicting *when*, not just *if*

The two models above only answer "yes or no, during the whole study." A
**Cox proportional hazards model** (usually just called "Cox") is a different
tool that models time directly, and it's the tool that correctly handles the
censoring problem described earlier instead of ignoring it.

Cox produces something called a **hazard ratio** for each factor: a number above
1 means "this raises risk," below 1 means "this lowers risk." Feeding "cancer
status" into this model at first produced a nonsensical result: it looked like
having cancer *lowered* risk. Digging in revealed why: in this dataset, the
"no cancer" group isn't healthy people, it's patients admitted for *other* severe
conditions (like sepsis or organ failure) that are often even more immediately
dangerous than cancer. The fix was to let the model treat "no cancer," "cancer,"
and "metastatic cancer" as three genuinely separate categories instead of a
single sliding scale, which produced sensible, medically plausible numbers.

Cox also comes with a built-in assumption worth understanding: that a given
factor's *effect on risk* stays constant over the entire time period. This was
formally tested rather than assumed, several factors technically violated it (not
surprising with almost 9,000 patients, where even tiny deviations become
statistically detectable), and the violations were investigated rather than
ignored or hidden.

## Phase 5: Grading your own homework honestly

Everything up to this point compared models using a technique called
cross-validation, which is a good way to *choose* between models, but has one
downside: every patient gets used for both training and testing at some point,
just in different rounds. To get a truly honest final grade, the data was split
once, at the very start of this phase, into three separate piles:

- **Train** (used to build the model)
- **Validation** (used to fine-tune small decisions, like whether adjusting the
  model's raw probabilities helps)
- **Test** (touched only at the very end, to report a final score, never used to
  make any decision along the way)

This matters because if you peek at your test data while making decisions, your
final score becomes overly optimistic in a way you can no longer detect. The
final, honest numbers: **AUROC 0.700** and a calibration score (Brier score) of
**0.194**, both reported with a range showing how much they might shift with a
different sample of patients, instead of a single falsely-precise number.

**A fairness check** was also done here: does the model work equally well for
everyone? Splitting patients by age band showed a real gap: the model is
noticeably better at ranking patients under 50 (AUROC 0.744) than patients 85 and
older (AUROC 0.595, barely better than guessing). This is disclosed clearly
rather than hidden, because using this model on the oldest patients without
knowing it's less reliable there would be actively misleading.

## Phase 6: Making it usable

A finished model sitting in a notebook is not a finished project. The last phase
wrapped everything into a small web service (an API) that a program (or a simple
web page) can send a patient's information to and get a risk score back. Two
important engineering ideas here:

- **The exact same cleaning code used to train the model is reused when serving
  predictions.** If those two things ever used different logic, you could end up
  training on one version of "clean data" and predicting on a subtly different
  one without anyone noticing, a classic, hard-to-detect real-world bug.
- **Input validation.** If someone sends a nonsensical value (a negative age, a
  missing field), the service rejects it immediately with a clear error, instead
  of quietly producing a meaningless prediction.

A second, small tool was also built here: a **triage panel** that shows a list of
patients ranked by risk, meant to prompt a conversation among a care team (`"this
patient might need closer attention"`), never to make an automatic decision on
its own. It deliberately uses the Cox model from Phase 4, not just a single risk
number, because it can tell apart "high risk very soon" from "high risk somewhere
down the line," which matters a lot for deciding what to do *right now* versus
later.

## The bugs, because this is the most useful part for a beginner

Nobody gets everything right on the first try, and pretending otherwise would be
the least honest part of this document. Five real mistakes were found while
double-checking this project's own work, not by outside review:

1. **Two columns were swapped** in the raw data file (explained above, Phase 2).
2. **A hazard ratio had the wrong sign** because a category was treated as a
   number when it wasn't one (Phase 4).
3. **A sentence in an earlier draft of this project's own documentation was
   wrong**: it described a rejected experimental setting's poor results as if
   they belonged to the actual, final model. The chart was always right; only
   the words describing it were outdated. Found by recomputing the numbers from
   scratch rather than trusting an old description.
4. **The web service crashed only when packaged into a Docker container**,
   because a piece of code that computed a file path worked fine during
   development but broke under the slightly different setup used for
   deployment. It passed every automated test and still didn't work, because no
   test happened to run in that exact configuration. It was only caught by
   actually building and running the real container.
5. **A second version of the same kind of mistake**: the triage panel's risk
   calculation quietly returned "unknown" for a few patients, because it fed
   data prepared one way into a model that had been trained on data prepared a
   slightly different way. Found by actually reading the output of a real
   request, not just checking that the request succeeded.

The pattern across all five: automated tests are necessary but not sufficient.
Several of these were only caught by manually re-deriving a number, or actually
running the real thing end-to-end, instead of trusting that "it passed" meant
"it's correct."

## A short glossary

- **Censoring**: not knowing the final outcome for someone because you stopped
  watching before it happened (they left the study, or the study ended first).
- **AUROC**: how well a model ranks who's higher risk versus lower risk (0.5 =
  random guessing, 1.0 = perfect).
- **Calibration**: whether a model's predicted percentages match reality (a 40%
  prediction should come true about 40% of the time).
- **Cross-validation**: testing a model on several different slices of the data
  in turn, to get a more reliable score than testing just once.
- **Hazard ratio**: in a Cox model, how much a factor multiplies risk (above 1 =
  riskier, below 1 = protective).
- **Held-out test set**: data set aside and never used until the very last,
  final check, to get an honest final score.
- **Overfitting / leakage**: when a model's reported performance looks better
  than it will actually be in the real world, usually because it was
  accidentally allowed to "see" information it shouldn't have had access to yet.

## Where to go next

- `reports/model_card.md`: the same story, denser, with every exact number and
  the file/function each decision lives in.
- `notebooks/`: the actual step-by-step analysis, one file per phase, meant to
  be read top to bottom.
- The published report artifact (ask if you don't have the link): the same
  content as this file, laid out visually.
