# Recording a Scope #2 demo

How to record the demonstrations that teach the agent to encode grades into the
web portal, and how to turn them into a trained model.

Companion to [demonstration_recording_pipeline.md](demonstration_recording_pipeline.md),
which covers the Scope #1 (desktop form + Notepad) version of the same pipeline.

```
1. RECORD   →  2. CLEAN  →  3. OVERSAMPLE  →  4. TRAIN  →  5. CHECK THE CLONE
```

---

## 0. What you are actually making

Not a video. Every time you click or type, the recorder asks the browser *what
is on the page right now* and writes one JSON file holding three things: the
page before your action, the action itself, and the page after.

Training later matches your **click position** against the **rectangles** in
that element list to work out which cell you meant. That is the entire
mechanism, and it is why the setup rules below matter: anything that makes the
rectangles disagree with where your mouse really was corrupts the data without
producing a single error message.

---

## 1. Set up the desktop first

Two windows, side by side, never overlapping. Snap them with `Win`+`←` /
`Win`+`→`.

```
┌───────────────────────────┬───────────────────────────┐
│  Chrome                   │  Excel                    │
│  the portal               │  grade_sheet.xlsx         │
│  tab 1, and only tab 1    │  sheet: SUMMARY           │
└───────────────────────────┴───────────────────────────┘
```

- [ ] Primary monitor only, display scaling at 100%
- [ ] Browser zoom at 100%
- [ ] The portal is the **first** tab — `WebObserver` grabs `contexts[0].pages[0]`
      once when it attaches and never looks again
- [ ] Terminal minimized or fully behind Chrome, covering nothing
- [ ] Notifications off

> **Fails silently.** If Excel sits *on top of* the portal, a click on a
> spreadsheet cell can be recorded as a click on the portal input underneath it.
> Side by side is not a style preference.

**Dual monitor:** the browser window's position is read from Windows directly
(`Chrome_RenderWidgetHostHWND`, the OS window that *is* the viewport), in the
same coordinate space the recorder sees clicks in — so a second monitor is fine,
whichever one is primary. Keep *Chrome and Excel* on the same screen as each
other, and use the second
screen freely for anything else — terminal, docs, music. Only the browser
window's position matters: element rectangles are built from
`window.screenX/screenY` (virtual-desktop space, negative to the left of
primary) while `screen_resolution` reports only the current monitor. On the
primary those agree; off it they do not. Not measured on a real two-monitor rig
— flagged as reasoned, and the safe setup costs nothing.

Clicks and keystrokes on the second screen are dropped as source-side, the same
way your trips to Excel are, so they cannot pollute the demo. Two things still
hold: don't drag Chrome between monitors mid-session (its scale factor changes
underneath the recording), and don't perform the demo itself over there.

---

## 2. Serve the portal

Terminal 1, leave it running:

```bash
python practice_apps/mocksite/serve.py
# → http://127.0.0.1:8765/v0_base/index.html
```

This exists instead of `python -m http.server` because it sends no-cache
headers. A cached `portal.js` has cost real debugging time twice: the page looks
right and the automation fails for a reason that is nowhere on screen.

---

## 3. Start Chrome with a debugging port

Terminal 2. This is **PowerShell** syntax - the shell this project is normally
driven from. `^` line continuations and `%TEMP%` are CMD, and PowerShell rejects
both with `Unexpected token '^'`:

```powershell
Start-Process "C:\Program Files\Google\Chrome\Application\chrome.exe" -ArgumentList "--remote-debugging-port=9222", "--user-data-dir=$env:TEMP\chrome-record", "http://127.0.0.1:8765/v0_base/index.html"
```

Confirm the port is really open before recording anything:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:9222/json/version | Select-Object -Expand Content
```

`-UseBasicParsing` skips the security prompt PowerShell raises when it would
otherwise parse the response as a web page.

JSON back means you are good. An error means Chrome ignored the flag, almost
always because a normal Chrome was already running.

The separate profile matters as much as the port. If your normal Chrome is
already running, the flag is quietly ignored and there is no debug port at all.
A fresh profile also means exactly one window and no "restore 40 tabs" prompt
deciding what tab 1 is.

---

## 4. Open the grade sheet

```powershell
start components\scope2\data\sheets\grade_sheet.xlsx
```

Use the **SUMMARY** sheet, and snap it to the half of the screen Chrome is not
using.

---

## 5. Start the recorder, then read one line

Terminal 3:

```bash
python scripts/record_trace.py --demo --type web
```

Recording starts immediately. Before touching anything, find this line:

```
[DemoRecorder] WebObserver will attach at http://localhost:9222 —
               DOM state active (aria-labelledby resolved).
```

> **Stop if you see this instead:** `No browser answering at http://localhost:9222`
>
> It falls back to reading Chrome through Windows accessibility, which cannot
> resolve the per-row labels. All fifty rows of a column arrive under one name,
> so the demo cannot say which row you filled — and it looks completely fine.
> Fix the browser and start over rather than recording an hour of it.

**The two keys:**

| Key | Does |
|---|---|
| `F9` | pause / resume — use it for anything that isn't the demo |
| `F10` | save and quit |

Function keys are filtered out of the recorded text, so pressing them never
leaks into the data.

---

## 6. Fill the rows, the same way every time

Four columns come from the sheet. One does not.

| Portal column | Grade sheet column | Example |
|---|---|---|
| Course | `PROGRAM` | BS Information Systems |
| Year 1-5 | `YEAR LEVEL` | 2 |
| Grade 0-100 | `FINAL GRADE` | 85 |
| Remarks | **nothing — you decide** | Passed |
| Recommendations | optional | leave blank |

> **Remarks is the interesting one.** No column in the sheet holds it. You work
> it out from the grade — and being consistent about that (the same cutoff,
> every row, all fifty) is what makes the rule learnable instead of arbitrary.

### Per row

1. Click the sheet cell, `Ctrl`+`C`
2. `Alt`+`Tab` to Chrome
3. Click the portal cell, `Ctrl`+`V`

Only the last two are recorded. The recorder checks which window you were
actually in, so the trip to Excel and the copy itself are dropped as
source-side. You will watch it happen:

```
[source-side] click in 'grade_sheet.xlsx - Excel' while observing
              'Grade Encoding Portal - V0 Base' -- not a demonstration step
  [0042] click  Grade 0-100 Abad, Andrea A.
  [0043] paste  "85"
```

That running log is live proof the recording is clean: two lines per value, with
a source-side line between them. If you see portal steps you did not intend,
stop and look before recording another forty rows of it.

### Remarks: use the keyboard, not the dropdown

Click the Remarks cell, then press `P` or `F`. A mouse-picked option from a
native dropdown is drawn by Windows, not by the page — there is no element under
your click, so the step is worth nothing. The first letter selects the option
and records cleanly.

### Rules for the whole session

- **Same column order on every row.** That consistency *is* the thing being
  learned.
- **Do not go back and fix a row.** If you fumble one, finish it and move on;
  corrections teach the model to correct.
- **Do not click any other app.** Not the terminal, not the taskbar.
- **Do not move or resize either window** once you have started.

---

## 7. Finish and check what you got

Press `F10`. It prints the session folder and the step count.

```
data/demos/human/session_20260821_143012/
  live_step_0000.json  live_step_0001.json  ...
```

One full pass of all fifty students should land near **400 steps** — four filled
columns per row, two steps each. Wildly fewer means something was being dropped;
find out what before recording again.

**How many sessions:** at least three full passes. Scope #1's own experience is
the honest reference — it needed thousands of clean steps before navigation
accuracy moved, and no equivalent number has been measured for the portal yet.
More is strictly better; three is where it becomes worth training at all.

---

## 8. Clean, and watch the counts

```bash
python scripts/clean_demos.py data/demos/human data/demos/portal_clean --scope grade_portal
```

`--scope` is **not optional**. Without it the cleaner uses Scope #1's window
markers, decides a window called "Grade Encoding Portal" is not the form, and
throws away every click you recorded. It reports that as a large junk count, not
as an error:

```
window markers: ['grade encoding portal', 'grade portal', 'student rating']
kept 397  |  dropped: dropdown-select=0, junk=6, dupes=2  -> data/demos/portal_clean
```

Read the `kept` number. Near zero means the markers did not match — the single
most likely thing to go wrong here, and one flag away from fixed.

---

## 9. Train

```bash
python scripts/oversample_tails.py data/demos/portal_clean data/demos/portal_final

python scripts/train.py --trace_dir data/demos/portal_final --epochs 80 \
  --d_model 128 --num_layers 4 --dim_feedforward 256 --max_elements 320
```

> **`--max_elements 320` is not optional for this portal.** It raises how many
> elements of the page training may look at. The default is 128; the portal has
> 303, and **29 of the 50 Grade cells sit past that line**. Without it, every
> click on the bottom two-thirds of the grid is dropped and teaches nothing.
> Training warns when your traces are bigger than the cap. Raising it costs zero
> parameters (142,629 either way), and the checkpoint remembers the value, so
> the live run sees the whole grid too.

Oversampling copies the tail of each pass — the last row, then Save — so the
rare "everything is filled, now submit" transition is represented well enough to
learn. That is how Scope #1 learned to click Submit on its own.

Then check whether it cloned *you* rather than learning some order:

```bash
python scripts/test_clone.py data/demos/portal_clean/session_20260821_143012
```

It reports exact-match percentage and an offset distribution — `0` means it
picked your cell, `+1` the next one down. The offsets tell you what it actually
learned, which a single accuracy number never does.

---

## 10. Run it

```bash
python run_scope2.py --records 0-4 --model tasks/grade_portal/model.pt
```

Once the model is good, drop the hand-written helper and run the same shape
Scope #1 uses — transformer for *where*, LLM for *what*:

```bash
python run_scope2.py --records 0-4 --no-plugin
```

That is the finish line: the mapping from `FINAL GRADE` to `Grade 0-100` comes
from what it watched you do, not from `COLUMN_MAP` in
`components/agent/task_plugins/grade_portal_plugin.py`.

**This needs LM Studio running** (`lms server start`, then load a model).
Without it, the LLM call fails quietly and every step returns `wait` — an agent
that looks alive and does nothing. Use `--provider none` if you mean to run
without it.

---

## When something looks wrong

| What you see | What it means |
|---|---|
| Zero steps recorded | The observer fell back to accessibility mode and every browser step was discarded as a noise app. Check the attach line in step 5. |
| Steps marked `[!empty state]` | The snapshot came back with no elements — usually the portal tab was closed or navigated away mid-session. |
| `kept 0` after cleaning | Missing `--scope grade_portal`. |
| Portal steps you did not perform | A window is overlapping the portal. Stop, fix the layout, start a new session. |
| `WARNING: ... more than max_elements` | Pass `--max_elements 320` to `train.py`. |
| Values recorded as `""` | Fixed 2026-08-21. If it persists, the clipboard was empty at paste time. |
| `[!] click at ... matched no element` | The coordinates and the page disagree. Stop — the session is not usable. It means the browser's screen position could not be read and the fallback geometry was wrong (this is what a multi-monitor layout used to do). |
