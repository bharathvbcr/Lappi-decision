# The human's answers on Fable's post-F ruling (2026-10-02)

The lead asked these through AskUserQuestion at ~14:05 UTC 2026-10-02, on the items in
`AUDIT/post-f-2026-10-02/fable-post-f-ruling.md` that are the human's call. Each question and the
option the human chose are recorded verbatim.

1. **v5 order.** "F's prose abstention depends on the seed (54/60, 0/60, 1/60), so the open
   question is now abstention. Fable recommends moving v5's abstention-weight arm (3 seeds) ahead
   of J5′. Same runs, same approved $137, only the order changes. Move it?"
   Chosen: **"Arm before J5′ (Recommended)"**, with the text "v5 seeds 0–2, then seeds 3–4 if the
   spread rule fires, then the abstention-weight arm ×3 (if it has room or you pin a yes), then
   J5′ ×3."

2. **J6(a) advance yes.** "F seed 1's needle score of 61/61 means the F′ rule can only print
   'refused' (J6(f) has no room to win), so J6(a) would be skipped and the GPU would idle until
   v5. Give an advance yes for J6(a) to run in that case? It's single-GPU, capped at 32,400 s
   ($20.61), expected about $11–14. It still won't run if any other refusal reason appears or an
   F′ run exists."
   Chosen: **"Yes, run J6(a) then (Recommended)"**, with the text "I put your words in the box's
   pin file. J6(a), then J6(g), use the slot instead of the GPU idling."

3. **F′ policy on a room-only refusal.** "When the F′ rule refuses only because J6(f) has no
   room, and J6(d)-v4 (lr 3e-5, β2 0.95) still wins all three of its targets, what should
   happen?"
   Chosen: **"v5 takes the J6(d) change (Recommended)"**, with the text "No separate F′. v5's
   three seeds use lr 3e-5 and β2 0.95, which saves about 32 GPU-hours and $73. The test is
   confounded with v5's data changes. If J6(d)-v4 doesn't win, v5 keeps F's settings."

4. **Linear-control refit.** "The linear-control fitter bug is fixed: intent.domain never
   converged under the old step size. Fable recommends re-running F's three controls on the Mac
   CPU as report-only rows, leaving F's 'paired margin vs linear' gate as not_run, and using the
   fixed fitter from v5 on. Agree?"
   Chosen: **"Report-only refit (Recommended)"**, with the text "The refit rows are recorded
   beside F's original rows, and no gate reads them. v5 uses the fixed fitter."

## The pin written on the box from answer 2

`/home/ubuntu/queue/j6a-on-room-refusal-yes` holds this text:

    Human (Bharath), 2026-10-02 ~14:05 UTC, AskUserQuestion: "Yes, run J6(a) then (Recommended)" to "Give an advance yes for J6(a) to run in that case [fsucc refused only for room]? It's single-GPU, capped at 32,400 s ($20.61), expected about $11–14. It still won't run if any other refusal reason appears or an F′ run exists." Recorded in AUDIT/post-f-2026-10-02/human-answers-post-f.md.
