# UX: Smart Extraction Redesign (Thumbnails First, Extract Selection)

**Problem:** Extracting ALL dives before user review wastes 30-50% of extraction time on dives user will delete.

**Solution:** Generate thumbnails first (fast), let user select which dives to keep, extract ONLY selected dives.

**Impact:**
- User sees gallery in 5 minutes (not 45)
- Only extracts what user wants (50% less extraction)
- 15-min video: 45 min → 30 min (33% faster)
- 1-hour video: 3+ hours → 1.5 hours (50% faster)

---

## New Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ USER RUNS: python -m diveanalyzer process video.mp4         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: DETECTION (30s - 60s)                              │
│ Browser opens: "🔍 Detecting dives... 0/138"                │
│ Shows audio/motion detection progress                        │
└─────────────────────────────────────────────────────────────┘
    ↓ [Detection completes: 138 dives found]
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: THUMBNAIL GENERATION (2-5 min)                     │
│ Gallery loads: "Found 138 dives - generating previews..."   │
│ Shows 138 empty cards with placeholders                      │
│ NO EXTRACTION YET - just generating thumbnails!             │
└─────────────────────────────────────────────────────────────┘
    ↓ [Thumbnails generating in background, 4 workers]
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2B: GALLERY REVIEW MODE (READY TO SELECT)             │
│ ✓ dive_001 [preview thumbnail] [checkbox]                   │
│ ✓ dive_002 [preview thumbnail] [checkbox]                   │
│ ⏳ dive_003 [loading...] [checkbox]                          │
│ ⏳ dive_004 [loading...] [checkbox]                          │
│                                                              │
│ STATUS: "138 dives - 95 previews ready"                     │
│                                                              │
│ USER REVIEWS THUMBNAILS & SELECTS:                          │
│ [✓] Keep dive_001  [✓] Keep dive_002  [✗] Delete dive_003   │
│                                                              │
│ "Selected: 108 dives to keep, 30 to delete"                 │
└─────────────────────────────────────────────────────────────┘
    ↓ [User clicks "Extract Selected Dives"]
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: EXTRACTION (ONLY SELECTED - 15-25 min)             │
│ Browser: "Extracting 108 selected dives..."                 │
│ Cards with extraction progress spinners                      │
│                                                              │
│ ✓ dive_001.mp4 (extracted 2s ago)                           │
│ ✓ dive_002.mp4 (extracted 1s ago)                           │
│ ⏳ dive_003.mp4 (extracting 45%)                             │
│ ⏳ dive_004.mp4 (queued...)                                  │
│ ⏪ dive_005 [NOT SELECTED - skipped]                        │
│ ⏪ dive_006 [NOT SELECTED - skipped]                        │
│                                                              │
│ USER CAN CONTINUE REVIEWING EXTRACTED DIVES                 │
└─────────────────────────────────────────────────────────────┘
    ↓ [All selected dives extracted]
┌─────────────────────────────────────────────────────────────┐
│ FINAL STAGE: READY TO ACCEPT                                │
│ ✓ "All 108 selected dives extracted and ready"              │
│ [Accept & Close]                                             │
└─────────────────────────────────────────────────────────────┘
    ↓
│ Server shutdown, browser closes
│ User has: 108 MP4 dives (only what they wanted)
└─────────────────────────────────────────────────────────────┘
```

---

## Ticket Breakdown

### PHASE 1: Detection (Existing - No Changes)

**[TICKET-1] Show detection placeholder while analyzing**

Status: Already implemented (no changes needed)

---

### PHASE 2: Thumbnail Generation (NEW - CRITICAL)

**[TICKET-201] Generate thumbnails instead of extracting full videos**

**Current:** Extract full video (20-30 sec per dive)
**New:** Generate 3 thumbnail frames (2 sec per dive)

**Workflow:**
- Detection completes
- Don't extract videos yet
- Start thumbnail generation for all dives
- Generate: start frame, middle frame, end frame
- Save as JPEG in memory or temp directory

**Implementation:**
- Use FFmpeg `thumbnail` filter (much faster)
- Generate in parallel (4-8 workers, same as extraction)
- Store thumbnails in memory cache or `/tmp/`
- Don't store full MP4 files yet

**Files:**
- `diveanalyzer/extraction/thumbnails.py` - NEW: Thumbnail generation module
- `diveanalyzer/cli.py` - Skip extraction, go to thumbnails
- `diveanalyzer/server/sse_server.py` - Send thumbnail events

**Performance:**
- 138 dives: 3 thumbnails × 2 sec = 6 sec per dive / 4 workers = ~3 minutes
- 500 dives: 500 × 2 sec / 4 workers = ~5 minutes
- vs. extraction which would take 45+ minutes

**AC:**
```
✓ Detection completes, no video extraction starts
✓ Thumbnails generated from dive time ranges
✓ 138 dives: thumbnails ready in 3-5 minutes
✓ 3 JPEG frames per dive (start, middle, end)
✓ No full MP4 files created until user selects
```

---

**[TICKET-202] Open gallery with thumbnail previews (not video clips)**

**Goal:** Gallery displays dive cards with thumbnail previews instead of waiting for extraction

**Implementation:**
- After detection, generate empty cards for all dives
- As thumbnails complete, populate card with images
- Show placeholder while thumbnail generating

**Files:**
- `diveanalyzer/utils/review_gallery.py` - Thumbnail display template
- `diveanalyzer/server/sse_server.py` - Serve thumbnail images

**AC:**
```
✓ Gallery loads with 138 empty cards (2-3 sec after detection)
✓ Thumbnails appear as they're generated
✓ Each card shows 3-frame preview
✓ No "Waiting for dives..." message
✓ Clear indication: "Generating previews..."
```

---

**[TICKET-203] Show thumbnail progress in gallery header**

**Goal:** Display real-time progress of thumbnail generation

**Implementation:**
- Track: thumbnails_ready, thumbnails_total
- Update header: "138 dives - 95 previews ready"
- Progress bar shows thumbnail generation progress

**Files:**
- `diveanalyzer/server/sse_server.py` - Track thumbnail progress
- `diveanalyzer/utils/review_gallery.py` - Display header

**AC:**
```
✓ Header shows: "138 dives - 0 previews ready"
✓ Updates to: "138 dives - 50 previews ready"
✓ Updates to: "138 dives - 138 previews ready ✓"
```

---

### PHASE 2B: User Selection (NEW - CRITICAL)

**[TICKET-204] Add checkbox selection to each dive card**

**Current:** Cards show "selected" for deletion
**New:** Cards show "keep/delete" for extraction filtering

**Implementation:**
- Each card has checkbox (default: checked/keep)
- User can uncheck to "delete" (don't extract)
- Update counter as user checks/unchecks

**Files:**
- `diveanalyzer/utils/review_gallery.py` - Checkbox template

**AC:**
```
✓ Each dive card has checkbox
✓ Default: checked (will be extracted)
✓ User can uncheck to skip extraction
✓ Counter shows: "Selected: 108/138 dives"
```

---

**[TICKET-205] Implement selection buttons (Select All, Deselect All)**

**Goal:** Quick controls for bulk selection/deselection

**Implementation:**
- "Select All" button - check all dives
- "Deselect All" button - uncheck all dives
- Update counter in real-time

**Files:**
- `diveanalyzer/utils/review_gallery.py` - Button handlers

**AC:**
```
✓ "Select All" button checks all dives
✓ "Deselect All" button unchecks all dives
✓ Counter updates: "Selected: 138/138" or "Selected: 0/138"
```

---

**[TICKET-206] Add "Extract Selected Dives" button**

**Goal:** Transition from selection mode to extraction mode

**Current:** "Accept & Close" deletes selected, keeps rest
**New:** "Extract Selected Dives" button starts extraction only for checked dives

**Implementation:**
- Replace or supplement "Accept & Close" with "Extract Selected Dives"
- On click: POST request with list of selected dive IDs
- Server responds with extraction job
- UI transitions to extraction phase

**Files:**
- `diveanalyzer/utils/review_gallery.py` - Button and handler
- `diveanalyzer/server/sse_server.py` - Handle extraction start request

**AC:**
```
✓ "Extract Selected Dives" button present
✓ On click: sends selected dive IDs to server
✓ Server starts extraction ONLY for selected
✓ Unselected dives are not extracted
✓ UI shows extraction progress
```

---

**[TICKET-207] Show selection summary before extraction**

**Goal:** User confirms selection before spending time on extraction

**Implementation:**
- Show modal/summary: "You selected 108 dives to keep, 30 to delete"
- Show estimated extraction time: "~20 minutes"
- "Confirm & Extract" button
- "Cancel" button to go back and adjust selection

**Files:**
- `diveanalyzer/utils/review_gallery.py` - Confirmation modal

**AC:**
```
✓ Summary shows: "108 selected, 30 will be skipped"
✓ Shows: "Estimated extraction time: ~20 minutes"
✓ "Confirm & Extract" to proceed
✓ "Cancel" to go back and adjust
```

---

### PHASE 3: Smart Extraction (MODIFIED - CRITICAL)

**[TICKET-301] Extract only selected dives (not all dives)**

**Goal:** Only extract dives user selected, skip the rest

**Current:** Extract all 138 dives regardless of user selection
**New:** Extract only 108 dives (or however many user selected)

**Implementation:**
- Receive list of selected dive IDs from gallery
- Filter extraction queue to only selected dives
- Skip unselected dives entirely (don't create MP4 files)
- Start extraction with 4 parallel workers

**Files:**
- `diveanalyzer/extraction/parallel.py` - Filter extraction queue
- `diveanalyzer/cli.py` - Accept selected dives from gallery
- `diveanalyzer/server/sse_server.py` - Receive selection, pass to extraction

**Performance:**
- Before: 138 dives × 20 sec = 45 minutes (even if user deletes 30)
- After: 108 dives × 20 sec / 4 workers = ~15 minutes (only what user wants)

**AC:**
```
✓ Only selected dives are extracted
✓ Unselected dives are not touched
✓ 108 dives × 20 sec / 4 workers = ~15 min
✓ 50% time savings vs. extracting all
```

---

**[TICKET-302] Show extraction progress with selected/skipped indication**

**Goal:** User sees which dives are being extracted vs. skipped

**Implementation:**
- Card shows:
  - ✓ dive_001 "Extracted" (completed)
  - ⏳ dive_003 "Extracting 45%" (in progress)
  - ⏪ dive_005 "Skipped (not selected)" (gray out)
- Progress shows: "Extracted: 5/108 selected"

**Files:**
- `diveanalyzer/utils/review_gallery.py` - Card state rendering
- `diveanalyzer/server/sse_server.py` - Send extraction state

**AC:**
```
✓ Card shows extraction status
✓ Skipped cards clearly marked
✓ Counter shows: "Extracted: 5/108 selected"
✓ Visual clarity on what's being extracted
```

---

**[TICKET-303] Parallel extraction of selected dives (same as before)**

Already covered in original tickets.

**AC:**
```
✓ 4 parallel workers extract dives
✓ Same extraction speed as before
✓ But only extracting ~75% of dives (user's selection)
```

---

### PHASE 4: Completion

**[TICKET-401] "Accept & Close" button for final acceptance**

**Goal:** User reviews extracted dives and closes

**Current:** Deletes selected, accepts rest
**New:** Extraction already done, just accept and close

**Implementation:**
- After all selected dives extracted
- "Accept & Close" button appears
- On click: Server deletes skipped dives, keeps extracted
- Closes browser and stops server

**Files:**
- `diveanalyzer/utils/review_gallery.py` - Finalization logic

**AC:**
```
✓ "Accept & Close" button available after extraction
✓ Deletes skipped dives (never extracted)
✓ Keeps extracted dives
✓ Closes cleanly
```

---

## Data Flow Diagram

```
DETECTION PHASE
┌──────────────┐
│ Detect dives │ → 138 dives found
└──────────────┘
       ↓
THUMBNAIL GENERATION (Parallel)
┌──────────────────────────┐
│ Generate thumbnails (×4) │ → 138 JPEG previews in 3-5 min
│ Store in memory/cache    │
└──────────────────────────┘
       ↓
GALLERY REVIEW (User selects)
┌──────────────────────────┐
│ Display cards with       │
│ thumbnail previews       │
│ User checks/unchecks     │
│ USER SELECTS: 108 dives  │
└──────────────────────────┘
       ↓ [POST: selected_dive_ids]
EXTRACTION (Parallel - ONLY SELECTED)
┌──────────────────────────┐
│ Extract 108 dives (×4)   │ → 108 MP4 files in 15-25 min
│ Skip 30 unselected dives │
└──────────────────────────┘
       ↓
FINAL ACCEPTANCE
┌──────────────────────────┐
│ All selected extracted   │
│ User reviews extracted   │
│ "Accept & Close"         │
│ Delete skipped dives     │
│ Close browser            │
└──────────────────────────┘
```

---

## Ticket Dependencies

```
DETECTION (Existing)
    ↓
TICKET-201 (Generate thumbnails, not videos)
    ↓
TICKET-202 (Show thumbnails in gallery)
    ↓
TICKET-203 (Show progress)
    ↓
TICKET-204 (Checkboxes for selection)
    ↓
TICKET-205 (Select All / Deselect All)
    ↓
TICKET-206 (Extract Selected button)
    ↓
TICKET-207 (Confirmation modal)
    ↓
TICKET-301 (Extract only selected)
    ↓
TICKET-302 (Show extraction progress)
    ↓
TICKET-303 (Parallel extraction)
    ↓
TICKET-401 (Accept & Close)
```

---

## Implementation Phases

### PHASE A: Critical Path (50% time savings)
**Effort: 4-6 hours**

1. TICKET-201 - Generate thumbnails
2. TICKET-202 - Show thumbnails in gallery
3. TICKET-204 - Selection checkboxes
4. TICKET-206 - Extract Selected button
5. TICKET-301 - Extract only selected
6. TICKET-303 - Parallel extraction
7. TICKET-401 - Accept & Close

**Result:** User sees gallery in 5min, extracts only selected dives, saves 50% time

### PHASE B: Polish (better UX)
**Effort: 2-3 hours**

1. TICKET-203 - Progress display
2. TICKET-205 - Select All/Deselect All
3. TICKET-207 - Confirmation modal
4. TICKET-302 - Extraction progress details

**Result:** Professional workflow with clear feedback

---

## Comparison: Old vs. New

### Old Workflow (45 min for 138 dives)
```
Detection (0.5 min)
  ↓
Extract ALL 138 (45 min) ← User doesn't know if they want these yet!
  ↓
Gallery Review (5 min)
  ↓
User deletes 30 dives ← 20 min of extraction was wasted!
  ↓
Result: 50.5 minutes, 30 dives extracted and deleted

Timeline:
0:00 ──────────────────────────────────── 45:00 ──── 50:00
Detect ............ Extract 138 dives .... Review & delete
```

### New Workflow (30 min for 138 dives)
```
Detection (0.5 min)
  ↓
Generate Thumbnails (3 min) ← Fast! Shows preview
  ↓
Gallery Review & Selection (2 min) ← User decides what to keep
  ↓
Extract ONLY 108 Selected (15 min) ← No wasted extraction!
  ↓
Final Accept (1 min)
  ↓
Result: 21.5 minutes, only 108 dives extracted

Timeline:
0:00 ─ 0:30 ─── 3:30 ────────── 5:30 ────────────────────── 20:30 ─ 21:30
Detect Thumbnails Gallery Review Extract 108 Selected Accept
```

**Savings: 29 minutes (58% faster)!**

---

## Performance Analysis

### 138 Dives (15-min video)
| Stage | Old | New | Savings |
|-------|-----|-----|---------|
| Detection | 0.5 min | 0.5 min | - |
| Extraction | 45 min | 20 min (only 108) | **25 min (56%)** |
| Thumbnails | - | 3 min | - |
| Review | 5 min | 2 min | 3 min |
| **TOTAL** | **50.5 min** | **25.5 min** | **50% faster** |

### 500 Dives (1-hour video)
| Stage | Old | New | Savings |
|-------|-----|-----|---------|
| Detection | 1 min | 1 min | - |
| Extraction | 3+ hours | 1.5 hours (only 300) | **1.5+ hours (50%)** |
| Thumbnails | - | 8 min | - |
| Review | 10 min | 3 min | 7 min |
| **TOTAL** | **3+ hours** | **1.5 hours** | **50% faster** |

---

## New Button Flow

### Old Gallery Buttons
```
[Select All] [Deselect All] [Watch Selected] [Delete Selected] [Accept All & Close]
```
Purpose: Review extracted dives, delete unwanted, accept rest

### New Gallery Buttons
```
SELECTION MODE (Thumbnails visible, no extraction yet):
[Select All] [Deselect All] [Extract Selected Dives]

EXTRACTION MODE (Dives being extracted):
(No selection changes allowed - extraction in progress)

FINAL MODE (All selected extracted):
[Accept & Close]
```

---

## State Machine

```
STATE: THUMBNAIL_GENERATION
  ├─ Show placeholder cards
  ├─ Generate thumbnails in background
  └─ On completion → SELECTION_MODE

STATE: SELECTION_MODE
  ├─ Show cards with thumbnails
  ├─ User can check/uncheck
  ├─ "Extract Selected" button active
  └─ On click → CONFIRMATION_MODE

STATE: CONFIRMATION_MODE
  ├─ Show: "Extract 108 selected, skip 30, ~20 min"
  ├─ "Confirm & Extract" button
  ├─ "Cancel" button
  └─ On confirm → EXTRACTION_MODE

STATE: EXTRACTION_MODE
  ├─ Show extraction progress
  ├─ No user selection changes allowed
  ├─ User can review extracted dives
  └─ On completion → FINAL_MODE

STATE: FINAL_MODE
  ├─ Show: "All 108 selected dives extracted"
  ├─ "Accept & Close" button active
  └─ On click → close browser, stop server
```

---

## Questions for User

1. **Default selection:** Should all dives start checked (selected to extract)?
   - Pro: User extracts everything if they don't uncheck
   - Con: Large videos will still extract everything

2. **Confirmation step:** Should we require confirmation before extracting?
   - Pro: User knows exactly what will be extracted
   - Con: Extra click, slightly annoying

3. **Unselected dive handling:** What to do with dives user didn't select?
   - Option A: Delete them after extraction complete
   - Option B: Keep them as temporary, user can delete manually
   - Option C: Never create them at all (just skip)

4. **Selection persistence:** If user leaves and comes back, remember selection?
   - Or: Start fresh every time?

5. **Extract during review:** Can user also delete extracted dives while extraction ongoing?
   - Or: Lock down UI during extraction?

---

## Files to Create/Modify

**New Files:**
- `diveanalyzer/extraction/thumbnails.py` - Thumbnail generation
- `diveanalyzer/extraction/selection.py` - Selection management (optional)

**Modified Files:**
- `diveanalyzer/cli.py` - Flow: Detect → Thumbnails → Selection → Extract Selected
- `diveanalyzer/utils/review_gallery.py` - Gallery with selection UI
- `diveanalyzer/server/sse_server.py` - Handle selection, start extraction
- `diveanalyzer/extraction/parallel.py` - Filter to selected dives only

---

## Summary

### This redesign achieves:
✅ **50% time savings** (30 min instead of 60+ min)
✅ **No wasted extraction** (only extract what user wants)
✅ **Faster feedback** (thumbnails in 5 min instead of videos in 45 min)
✅ **Better UX** (user decides before big extraction)
✅ **Scalable** (works for 500-dive videos)

### For user's use case:
- **15-min video (138 dives):** 50 min → 25 min ✅
- **1-hour video (~500 dives):** 3+ hours → 1.5 hours ✅

### Implementation effort:
- **Critical path:** 4-6 hours
- **Full solution:** 6-8 hours
- **Payoff:** 50% time savings on every video ✅

---

## Comparison to Original Tickets

**Original 16 tickets** focused on:
- Opening gallery during extraction
- Parallel extraction
- Real-time updates
- User interaction during extraction

**New design** provides better approach:
- Don't extract until user decides
- Thumbnails provide preview (fast)
- Parallel extraction only for selected
- **50% extraction time saved**

This is actually **more efficient** than original design!

---

## Recommendation

**I strongly recommend this approach over the original 16 tickets because:**

1. **Bigger payoff:** 50% time savings vs. 25% with parallel extraction
2. **Better UX:** User reviews before big time commitment
3. **Simpler logic:** Clear separation of phases
4. **Scales better:** Doesn't matter if user has 138 or 500 dives - they control what's extracted
5. **Same implementation time:** Still 4-6 hours for critical path

**Shall I consolidate this into a cleaner ticket list for implementation?**
