# Smart Redesign - Implementation Roadmap

**Goal:** Extract only selected dives after thumbnail review (50% time savings)

**Status:** Ready to implement
**Estimated effort:** 4-6 hours for critical path

---

## Phase 1: Thumbnail Generation (2 hours)

**Impact:** Gallery appears in 5 min (not 45 min)

### TICKET-201: Generate thumbnails instead of full videos ⭐
- Create `diveanalyzer/extraction/thumbnails.py`
- Extract 3 frames per dive (start, middle, end) as JPEG
- Use FFmpeg thumbnail filter (fast)
- Parallel generation with 4 workers
- Store in temp directory or memory cache

**Files to create:**
- `diveanalyzer/extraction/thumbnails.py`

**Files to modify:**
- `diveanalyzer/cli.py` - Skip extraction, call thumbnail generation

### TICKET-202: Display thumbnails in gallery
- Update gallery HTML to show thumbnail previews
- Show placeholder while generating
- Update card as thumbnails complete

**Files to modify:**
- `diveanalyzer/utils/review_gallery.py`
- `diveanalyzer/server/sse_server.py` - Serve thumbnails

### TICKET-203: Show thumbnail progress
- Track thumbnails_ready / thumbnails_total
- Update header: "138 dives - 95 previews ready"
- Progress bar for thumbnail generation

**Files to modify:**
- `diveanalyzer/server/sse_server.py`
- `diveanalyzer/utils/review_gallery.py`

---

## Phase 2: Selection UI (1-2 hours)

**Impact:** User controls what gets extracted

### TICKET-204: Add selection checkboxes
- Each card has checkbox (default: checked)
- User can uncheck to skip extraction
- Counter: "Selected: 108/138 dives"

**Files to modify:**
- `diveanalyzer/utils/review_gallery.py`

### TICKET-205: Select All / Deselect All buttons
- Quick bulk selection controls
- Update counter in real-time

**Files to modify:**
- `diveanalyzer/utils/review_gallery.py`

### TICKET-206: "Extract Selected Dives" button
- Replace "Accept & Close" during selection phase
- POST selected dive IDs to server
- Transition to extraction phase

**Files to modify:**
- `diveanalyzer/utils/review_gallery.py`
- `diveanalyzer/server/sse_server.py`

---

## Phase 3: Smart Extraction (1-2 hours)

**Impact:** Extract only selected dives (50% time savings)

### TICKET-301: Extract only selected dives
- Receive list of selected dive IDs
- Filter extraction queue
- Skip unselected dives entirely
- Start parallel extraction

**Files to modify:**
- `diveanalyzer/cli.py`
- `diveanalyzer/extraction/parallel.py` (NEW)
- `diveanalyzer/server/sse_server.py`

### TICKET-302: Show extraction progress
- Mark extracted/extracting/skipped cards
- Progress: "Extracted: 5/108 selected"
- Visual differentiation

**Files to modify:**
- `diveanalyzer/utils/review_gallery.py`
- `diveanalyzer/server/sse_server.py`

### TICKET-303: Parallel extraction
- 4 worker threads
- Extract in parallel (not sequential)
- Send SSE event per completion

**Files to create:**
- `diveanalyzer/extraction/parallel.py`

---

## Phase 4: Finalization (0.5 hours)

### TICKET-401: Final "Accept & Close"
- Show after extraction complete
- Delete skipped dives (if any temp files)
- Close browser and stop server

**Files to modify:**
- `diveanalyzer/utils/review_gallery.py`

---

## Implementation Order

```
START
  ↓
1. Create thumbnails.py module (TICKET-201)
  ↓
2. Modify cli.py to use thumbnails (TICKET-201)
  ↓
3. Update gallery to show thumbnails (TICKET-202)
  ↓
4. Add progress tracking (TICKET-203)
  ↓
5. Add checkboxes to cards (TICKET-204)
  ↓
6. Add "Extract Selected" button (TICKET-206)
  ↓
7. Create parallel extraction (TICKET-301, 303)
  ↓
8. Filter to selected dives (TICKET-301)
  ↓
9. Show extraction progress (TICKET-302)
  ↓
10. Final "Accept & Close" (TICKET-401)
  ↓
DONE
```

---

## Testing Plan

### Unit Tests
- Thumbnail generation (3 frames per dive)
- Selection state management
- Extraction filtering (only selected)

### Integration Tests
- Full workflow: detect → thumbnails → select → extract
- 10-dive video (quick test)
- 138-dive video (user's test case)

### Manual Tests
- Select some dives, not all
- Deselect all, reselect some
- Extract during review
- Progress display accuracy

---

## Success Criteria

✅ Gallery appears in 5 minutes (not 45)
✅ User can review thumbnails and select
✅ Only selected dives are extracted
✅ 50% time savings vs. current
✅ No breaking changes to existing functionality
✅ Professional UI/UX

---

## Files Summary

**New files:**
- `diveanalyzer/extraction/thumbnails.py` - Thumbnail generation
- `diveanalyzer/extraction/parallel.py` - Parallel extraction orchestrator

**Modified files:**
- `diveanalyzer/cli.py` - Flow control
- `diveanalyzer/utils/review_gallery.py` - Gallery UI
- `diveanalyzer/server/sse_server.py` - Event handling

**Test files:**
- `tests/integration/test_smart_extraction.py` - NEW
- `tests/test_thumbnail_generation.py` - NEW

---

## Current Status

- [x] Design complete
- [x] Tickets created
- [x] Roadmap written
- [ ] Implementation started

**Next step:** Create thumbnail generation module (TICKET-201)

---

Ready to begin implementation! 🚀
