# UX: Parallel Extraction & Live Gallery Population

**Problem:** User waits for ALL dives to extract before seeing gallery. For 500-dive videos, this means 30+ minutes waiting.

**Solution:** Show gallery after detection with live population as dives are extracted in parallel.

**Impact:**
- User sees results immediately after detection
- Can start reviewing while extraction happens
- 138-dive videos: see gallery in 30s (not 45+ minutes)
- 500-dive videos: see gallery in 60s (not 3+ hours)

---

## PHASE 1: Detection → Gallery (Immediate Display)

**[TICKET-1] Open gallery after detection completes (before extraction)**

**Current:**
- Detection → Extraction → Gallery loads

**Goal:**
- Detection → Gallery opens (empty) → Extraction (background)

**Changes:**
- Move gallery opening from after extraction to after detection
- Pass final dive count to gallery
- Gallery renders empty cards for all 138 dives
- Show "extracting..." state for each card

**Files:**
- `diveanalyzer/cli.py` - Control flow
- `diveanalyzer/utils/review_gallery.py` - Accept dive count parameter
- `diveanalyzer/server/sse_server.py` - Send dive count in initial gallery load

**AC:**
```
✓ User sees gallery with 138 empty cards within 30s (detection time)
✓ No "Waiting for dives..." message
✓ Each card shows "⏳ Extracting..." spinner
✓ Card count displayed: "Found 138 dives - 0 extracted"
```

---

**[TICKET-2] Display final dive count in gallery placeholder**

**Goal:**
- Show "Found 138 dives - 0 extracted" header
- Real-time counter: "Found 138 dives - X extracted"
- Progress indicator showing extraction state

**Files:**
- `diveanalyzer/utils/review_gallery.py` - Header template
- `diveanalyzer/server/sse_server.py` - Send extraction progress

**AC:**
```
✓ Header shows: "Found 138 dives - 0 extracted"
✓ Updates to: "Found 138 dives - 5 extracted" (in real-time)
✓ Styling matches existing UI
```

---

**[TICKET-3] Hide "Waiting for dives..." message when gallery opens with known dive count**

**Goal:**
- Remove confusing message when user sees empty but populated card grid
- Message only appears if no dives detected

**Files:**
- `diveanalyzer/utils/review_gallery.py` - Condition for hiding message

**AC:**
```
✓ Message hidden when gallery loads with dives
✓ Message shown only if dive_count === 0
✓ No console errors
```

---

## PHASE 2: Parallel Extraction

**[TICKET-4] Refactor extraction to non-blocking parallel task**

**Current:**
- `extract_and_save_dive()` called sequentially in main thread
- Blocks gallery interaction

**Goal:**
- Launch extraction in background thread pool
- Main thread handles user interactions
- Return immediately from extraction phase

**Changes:**
- Create `ExtractionWorker` thread pool (4-8 workers based on CPU)
- Queue dive extraction tasks
- Each task returns upon completion
- Dives extracted in parallel, not sequentially

**Files:**
- `diveanalyzer/cli.py` - Launch extraction in background
- `diveanalyzer/extraction/parallel.py` - NEW: Parallel extraction orchestrator

**AC:**
```
✓ Extraction starts immediately after detection
✓ Multiple dives extracting simultaneously
✓ Gallery loads and is interactive while extraction happens
✓ No blocking of main thread
```

---

**[TICKET-5] Extract multiple dives in parallel (configurable workers)**

**Goal:**
- Use multiple CPU cores for parallel extraction
- Configurable workers: `--workers 4` (default: CPU count)
- FFmpeg handles concurrent stream copies (can be parallelized)

**Details:**
- Default: 2-4 workers (conservative, allows user interaction)
- Max: CPU cores / 2 (don't overwhelm system)
- Can be overridden: `--workers 8`

**Files:**
- `diveanalyzer/extraction/parallel.py` - Worker pool implementation
- `diveanalyzer/cli.py` - Accept `--workers` argument

**AC:**
```
✓ With --workers 4: 4 dives extract simultaneously
✓ 138 dives extract in ~35 dives/worker = ~9 extraction rounds
✓ Each round takes ~20s (avg dive duration), so ~3 minutes total
✓ Faster than sequential (which took 45+ minutes)
✓ System stays responsive
```

---

**[TICKET-6] Send SSE event for each completed dive extraction**

**Goal:**
- Gallery receives `dive_extracted` event after each FFmpeg completes
- Allows real-time UI updates

**SSE Event Format:**
```javascript
{
  event: 'dive_extracted',
  data: {
    dive_id: 1,
    filename: 'dive_001.mp4',
    timestamp: '2026-01-23T14:30:45Z',
    extracted_count: 5,
    total_count: 138,
    size_mb: 25.3
  }
}
```

**Files:**
- `diveanalyzer/extraction/parallel.py` - Emit event after extraction
- `diveanalyzer/server/sse_server.py` - Forward event to clients

**AC:**
```
✓ Each dive extraction triggers SSE event
✓ Event includes dive_id, count, progress
✓ Browser console shows events flowing in
✓ One event per 15-20 seconds (extraction pace)
```

---

## PHASE 3: Real-time Gallery Population

**[TICKET-7] Create dive card on SSE `dive_extracted` event**

**Goal:**
- Listen for `dive_extracted` event in browser
- Update corresponding dive card UI
- Change from "⏳ Extracting" to "✓ Ready to review"

**Implementation:**
- Add `eventSource.addEventListener('dive_extracted', ...)`
- Find card by dive_id
- Update state: spinner → checkmark
- Show file size
- Enable selection

**Files:**
- `diveanalyzer/utils/review_gallery.py` - Gallery JavaScript (lines 3000+)

**AC:**
```
✓ Dive card changes from spinner to checkmark
✓ Card becomes interactive (can be selected)
✓ Event fires once per dive as extracted
✓ No duplicates or errors
```

---

**[TICKET-8] Generate and queue thumbnails as dives are detected (not after extraction)**

**Goal:**
- Start thumbnail generation immediately after detection
- Generate in parallel with extraction
- Don't block extraction on thumbnail generation

**Current:** Thumbnails generated during extraction (blocking)
**New:** Thumbnails queued separately and generated in parallel

**Implementation:**
- After detection, create thumbnail generation queue
- Separate worker pool for thumbnails (2-4 workers)
- Extract dive without waiting for thumbnails
- Send dive_extracted event (without thumbnails initially)
- Update card when thumbnails ready

**Files:**
- `diveanalyzer/extraction/parallel.py` - Add thumbnail queue
- `diveanalyzer/utils/review_gallery.py` - Generate thumbnails
- `diveanalyzer/server/sse_server.py` - Handle thumbnail events

**AC:**
```
✓ Dives extract quickly without thumbnail wait
✓ Thumbnails generated in background
✓ Dive cards show without thumbnails first
✓ Thumbnails appear as they're ready
✓ User can review/select while thumbnails generating
```

---

**[TICKET-9] Stream extracted dive info to gallery in real-time**

**Goal:**
- Gallery header updates in real-time
- Shows: "Found 138 dives - X extracted - Y thumbnails ready"
- Progress bar shows extraction/thumbnail completion

**Implementation:**
- Track extraction progress: extracted_count, thumbnail_count
- Send progress every extraction or every 5 dives
- Gallery updates counter and progress bar

**Files:**
- `diveanalyzer/server/sse_server.py` - Track & send progress
- `diveanalyzer/utils/review_gallery.py` - Update progress display

**AC:**
```
✓ Header shows: "Found 138 dives - 0 extracted"
✓ Updates to: "Found 138 dives - 5 extracted - 3 thumbnails"
✓ Updates every 15-30 seconds (as dives complete)
✓ Progress bar fills as extraction progresses
```

---

**[TICKET-10] Add thumbnail preview to dive cards as they're generated**

**Goal:**
- Dive cards show thumbnail preview once thumbnails are ready
- Don't block card display on thumbnail generation

**Implementation:**
- Cards initially show placeholder (gray + spinner)
- On `thumbnail_ready` event, fetch and display thumbnail
- Smooth transition from placeholder to thumbnail

**Files:**
- `diveanalyzer/utils/review_gallery.py` - Thumbnail event handling
- `diveanalyzer/server/sse_server.py` - Send thumbnail_ready event

**AC:**
```
✓ Card shows placeholder initially
✓ Thumbnail appears when ready
✓ No blocking of card creation
✓ User can select dives with or without thumbnails
```

---

## PHASE 4: User Interaction During Extraction

**[TICKET-11] Allow user to accept/delete dives while extraction ongoing**

**Goal:**
- User doesn't have to wait for all dives to extract before deciding what to keep
- Can delete dives as soon as they appear
- Can start accepting (keeping) dives immediately

**Implementation:**
- Make gallery fully interactive from the start
- User can select/deselect/delete any dive at any time
- Extraction continues in background regardless of user actions
- Track which dives user selected vs. still extracting

**Files:**
- `diveanalyzer/utils/review_gallery.py` - Already has delete functionality
- `diveanalyzer/server/sse_server.py` - Handle delete while extraction ongoing

**AC:**
```
✓ User can click "Delete" on any dive (extracted or not)
✓ Deleted dives skip extraction
✓ Extracted dives deleted from disk when "Accept & Close"
✓ User can delete dives 5-50 while dives 51-138 still extracting
```

---

**[TICKET-12] Handle state consistency (user actions during extraction)**

**Goal:**
- Prevent race conditions when user deletes dive while it's being extracted
- Prevent user accepting while extraction still ongoing
- Show warning: "X dives still extracting, close early?"

**Implementation:**
- Track state of each dive: queued, extracting, extracted, deleted, selected
- If user deletes a queued dive: remove from queue
- If user deletes an extracting dive: let it finish, then mark as deleted
- On "Accept & Close": show count of dives still extracting

**Files:**
- `diveanalyzer/server/sse_server.py` - Track dive states
- `diveanalyzer/utils/review_gallery.py` - State-aware UI

**AC:**
```
✓ Delete queued dive: removed from queue immediately
✓ Delete extracting dive: extraction continues, marked as deleted
✓ Delete extracted dive: normal delete
✓ Accept & Close with ongoing extraction: show warning/count
✓ No race conditions or corrupt files
```

---

**[TICKET-13] Show extraction progress per-dive in gallery cards**

**Goal:**
- Each card shows progress: "⏳ Extracting..." → "✓ Ready" or "⏳ Queued (5 ahead)"
- User understands what's happening with each dive

**Implementation:**
- Card states: queued, extracting (with %), extracted, deleted, selected
- Show extraction progress bar per-dive
- Show queue position for queued dives

**Files:**
- `diveanalyzer/utils/review_gallery.py` - Card state rendering
- `diveanalyzer/server/sse_server.py` - Send per-dive progress

**AC:**
```
✓ Card shows: "⏳ Queued (5 ahead)"
✓ Card shows: "⏳ Extracting (45%)"
✓ Card shows: "✓ Extracted"
✓ Shows extraction speed (Mbps, file size)
```

---

## PHASE 5: Optimization (Optional Enhancement)

**[TICKET-14] Smart thumbnail generation strategy**

**Goal:**
- Generate preview thumbnail first (fast), then full timeline (slower)
- User sees card with preview while timeline generating
- Don't block review process

**Implementation:**
- Phase 1: Extract dive + generate 1 preview frame (3 thumbnails)
- Phase 2: Generate 10-frame timeline in background
- Send preview immediately, timeline when ready

**Files:**
- `diveanalyzer/extraction/thumbnails.py` - Multi-phase thumbnail generation

---

**[TICKET-15] Prioritize extraction of selected/early dives**

**Goal:**
- If user selects a dive, prioritize its extraction
- Extract dives 1-20 first (user sees them first)
- Extract later dives only after early ones done

**Implementation:**
- User selects a queued dive → move to front of queue
- Or: extract first 20 dives, then others

**Files:**
- `diveanalyzer/extraction/parallel.py` - Priority queue

---

**[TICKET-16] Adaptive parallel workers based on system resources**

**Goal:**
- Detect available CPU/RAM/disk speed
- Auto-set optimal worker count
- Don't overwhelm system or run out of disk I/O

**Implementation:**
- Check: CPU cores, RAM available, disk speed
- Set workers = CPU_cores / 2 (default)
- Monitor: if system > 90% CPU, reduce workers
- Monitor: if disk I/O saturated, reduce workers

**Files:**
- `diveanalyzer/extraction/parallel.py` - Adaptive worker scaling

---

## Implementation Order

### CRITICAL PATH (Must do first for UX improvement):
1. **TICKET-1** - Open gallery after detection ⭐ (BIGGEST UX impact)
2. **TICKET-4** - Parallel extraction infrastructure
3. **TICKET-5** - Multiple workers (4-8)
4. **TICKET-6** - SSE events for extraction complete
5. **TICKET-7** - Update gallery cards on event

### IMPORTANT (Complete the loop):
6. **TICKET-2** - Show extraction progress counter
7. **TICKET-3** - Hide "Waiting..." message
8. **TICKET-11** - User can delete while extracting

### NICE TO HAVE (Enhancement):
9. **TICKET-8** - Parallel thumbnail generation
10. **TICKET-12** - State consistency handling
11. **TICKET-13** - Per-dive progress display
12. **TICKET-9** - Progress bar updates
13. **TICKET-10** - Thumbnail preview streaming

### OPTIONAL (Performance):
14. **TICKET-14** - Preview + timeline thumbnails
15. **TICKET-15** - Priority queue for selection
16. **TICKET-16** - Adaptive worker scaling

---

## Estimated Impact

### Current Workflow (Sequential):
- 15-min video, 138 dives: ~45 minutes total wait
- 1-hour video, 500 dives: ~2.5+ hours total wait ❌

### With Critical Path (TICKET-1 to 7):
- 15-min video, 138 dives:
  - Gallery appears in 30s (detection)
  - Extraction happens in background (~3-5 minutes, 4 parallel workers)
  - User can review while extraction happens ✅
  - Total wait time: **30s to start reviewing** (not 45 min)

- 1-hour video, 500 dives:
  - Gallery appears in 60s (detection)
  - Extraction: 500 dives ÷ 4 workers × 20s/dive = ~25 minutes
  - User can review in **60s**, full set in 25 min ✅
  - vs. sequential which would take 3+ hours

### With All Tickets:
- Thumbnails don't block
- User interaction fully fluid
- Adaptive scaling prevents system overload
- **User experience is professional and fast** ✅

---

## Testing Strategy

### Unit Tests:
- Parallel worker pool behavior
- Event emission and reception
- State transitions

### Integration Tests:
- Detect 20 dives → Gallery opens → Extract in parallel
- User deletes dive while extracting → Handled correctly
- SSE events flow through correctly

### Load Tests:
- 500 dives: extraction doesn't overwhelm system
- Real-time UI update responsiveness
- Memory usage stays reasonable

### Manual Tests:
- 15-min video with 138 dives
- 1-hour video (if available)
- Delete dives while extracting
- Accept & Close while extraction ongoing

---

## Questions for User

1. **Worker count:** Default 4? Or dynamic (CPU_cores / 2)?
2. **Thumbnails:** Generate preview first (fast) or full timeline (slower)?
3. **Early closing:** Allow "Accept & Close" while dives still extracting?
4. **Progress display:** Show per-dive progress or just counter?

---

## Summary

**This refactoring solves the critical UX issue:**
- ✅ Gallery opens after detection (not after extraction)
- ✅ Dives extract in parallel (not sequentially)
- ✅ Gallery populates in real-time as dives are extracted
- ✅ User can review while extraction happens
- ✅ 15-min video: see gallery in 30s (not 45 min)
- ✅ 1-hour video: see gallery in 60s (not 3+ hours)

**16 actionable tickets organized in 5 phases with clear dependencies.**
