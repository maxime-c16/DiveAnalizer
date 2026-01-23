# Detailed Dive Review Modal - Implementation Tickets

## Overview
Add interactive modal for detailed dive review with rich timeline, quick actions, and seamless auto-advance workflow.

**User Story**: User clicks dive card → modal opens → sees 8-frame timeline → can quickly delete/keep → next dive auto-loads for efficient batch review.

---

## Ticket: FEAT-01 - Modal HTML/CSS Structure

**Description**: Create modal overlay, container, and basic styling.

**Requirements**:
- Modal overlay (semi-transparent dark background)
- Modal container centered on screen (max-width 900px)
- Smooth fade-in/out animations (300ms)
- Close button (X) in top-right corner
- Modal sections:
  - Header: dive title, close button
  - Timeline area: placeholder for 8 frames
  - Info panel: dive details
  - Action buttons: Keep, Delete, Cancel

**Files to modify**: `diveanalyzer/utils/review_gallery.py` (generate_html method)

**Acceptance Criteria**:
- [ ] Modal HTML renders without errors
- [ ] Modal hidden by default (display: none)
- [ ] Smooth fade animations on show/hide
- [ ] Mobile responsive (90vw on small screens)
- [ ] Z-index stacking correct (overlay visible)

**Size**: Small (CSS styling)

---

## Ticket: FEAT-02 - Detailed Timeline Frame Extraction

**Description**: Extract 8 evenly-spaced frames per dive at higher resolution.

**Requirements**:
- Frame positions: 0%, 12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5% of dive duration
- Resolution: 480x360 (vs 320x240 for grid)
- Quality: JPEG quality 4 (vs 5 for grid)
- Output: Base64 data URLs
- Storage: Add `timeline_thumbnails` array to dive data structure

**Technical**:
- Modify `extract_thumbnail()` to accept percentage parameter
- Create new method `extract_timeline_frames()` that calls extract_thumbnail 8 times
- Cache in `self.dives[i]['timeline_thumbnails']` during `scan_dives()`

**Files to modify**: `diveanalyzer/utils/review_gallery.py`

**Acceptance Criteria**:
- [ ] 8 frames extracted per dive
- [ ] Base64 encoded and embedded
- [ ] No external file dependencies
- [ ] Resolution visible/clear (480x360)
- [ ] Generation time <500ms per dive

**Size**: Small (utility function)

---

## Ticket: FEAT-03 - Modal Timeline Display

**Description**: Render 8-frame timeline in modal with visual scrubber.

**Requirements**:
- Horizontal frame strip layout
- Each frame: 100px × 75px (480x360 aspect ratio)
- Frames in flex container with gaps
- Optional: visual indicator showing which frame is "current"
- Hover effect: slight zoom on frames
- Click on frame: show frame number/timestamp (optional)

**JavaScript**:
- Function `renderTimelineFrames(diveData)` - populates modal timeline
- Updates when modal opened with new dive

**Files to modify**: `diveanalyzer/utils/review_gallery.py` (HTML generation + JavaScript)

**Acceptance Criteria**:
- [ ] 8 frames display horizontally in modal
- [ ] All frames visible without horizontal scroll
- [ ] Images load and display correctly
- [ ] Hover effects work
- [ ] Responsive on mobile (smaller frames if needed)

**Size**: Small (HTML + CSS)

---

## Ticket: FEAT-04 - Quick Action Buttons & Handlers

**Description**: Add Keep/Delete/Cancel buttons with event handlers.

**Requirements**:
- Button layout in modal footer
- Keep button: green, closes modal, returns to gallery (no action on dive)
- Delete button: red, marks dive selected, closes modal, auto-loads next (see FEAT-05)
- Cancel button: gray, closes modal, no changes
- Keyboard shortcuts: `K` (keep), `D` (delete), `Esc` (cancel)
- Button handlers disabled during transition (prevent double-clicks)

**JavaScript**:
- `handleKeep()` - close modal, proceed to next
- `handleDelete()` - mark checkbox, trigger next
- `handleCancel()` - close modal
- Modal-aware keyboard handlers

**Files to modify**: `diveanalyzer/utils/review_gallery.py` (JavaScript)

**Acceptance Criteria**:
- [ ] All 3 buttons clickable and functional
- [ ] Keyboard shortcuts work
- [ ] No double-click issues
- [ ] Safari compatibility verified
- [ ] Visual feedback (button hover/active states)

**Size**: Medium (event handling)

---

## Ticket: FEAT-05 - Auto-Advance on Delete

**Description**: Seamless workflow - delete dive → auto-load next dive in modal.

**Requirements**:
- After delete button clicked:
  - Current dive marked for deletion (checkbox checked)
  - Current card removed from gallery (fade out)
  - Next dive's modal immediately opens
  - Stats updated (keep count decreases)
- If last dive was deleted:
  - Modal closes
  - Show completion message
  - Optional: show delete confirmation count
- Transition smooth (no jarring UI changes)
- Data consistency: gallery and modal in sync

**JavaScript**:
- `openDiveModal(diveIndex)` - opens modal for given dive
- `deleteAndAdvance()` - delete current + open next
- `getNextUndeleted(currentIndex)` - find next keepable dive
- Handle edge case: all dives deleted

**Files to modify**: `diveanalyzer/utils/review_gallery.py` (JavaScript)

**Acceptance Criteria**:
- [ ] Delete → next dive loads in modal immediately
- [ ] Gallery card animates out smoothly
- [ ] Stats update correctly
- [ ] Last dive → completion state
- [ ] No console errors
- [ ] Workflow feels fast/responsive

**Size**: Medium (state management)

---

## Ticket: FEAT-06 - Modal Dive Info Panel

**Description**: Display detailed dive metadata in modal.

**Requirements**:
- Info sections:
  - Dive number & title (e.g., "Dive #005")
  - Duration (e.g., "1.23s")
  - Confidence level (HIGH/MEDIUM/LOW with color badge)
  - Detection metadata (optional: audio peak strength, motion score)
  - File size & bitrate (optional)
- Layout: Right sidebar or below timeline
- Responsive: stack vertically on mobile

**Data source**: Existing dive object + optionally enriched metadata

**Files to modify**: `diveanalyzer/utils/review_gallery.py` (HTML generation)

**Acceptance Criteria**:
- [ ] All info displays correctly
- [ ] Confidence badges styled correctly
- [ ] Mobile responsive
- [ ] Info reads clearly (good contrast)
- [ ] No overflow/layout issues

**Size**: Small (display only)

---

## Ticket: FEAT-07 - Keyboard Navigation in Modal

**Description**: Full keyboard control for efficient review workflow.

**Requirements**:
- `K` = Keep current dive, open next
- `D` = Delete current dive, open next (auto-advance)
- `←` / `→` = Manual navigate prev/next without action
- `Esc` = Close modal
- `?` = Show help overlay with all shortcuts
- Focus management: modal captures keyboard when open
- Context aware: shortcuts only work when modal is active

**JavaScript**:
- Extend existing keyboard handler to detect modal state
- Add help modal/popup showing all shortcuts
- Prevent conflicts with gallery-level shortcuts

**Files to modify**: `diveanalyzer/utils/review_gallery.py` (JavaScript)

**Acceptance Criteria**:
- [ ] All shortcuts work as specified
- [ ] Keyboard events don't bubble to gallery
- [ ] Help menu displays and is readable
- [ ] Safari keyboard handling verified
- [ ] No conflicts with browser shortcuts

**Size**: Medium (event handling)

---

## Ticket: FEAT-08 - Testing & Performance Validation

**Description**: Integration testing and performance optimization.

**Requirements**:
- Test scenarios:
  1. Open gallery → click card → modal opens with timeline
  2. Press `D` → auto-advance to next dive
  3. Repeat until last dive
  4. Delete last dive → completion state
  5. Use mouse + keyboard mixed
  6. Test on Safari, Chrome, Firefox
- Performance targets:
  - Modal open time: <200ms
  - Auto-advance transition: <300ms
  - Thumbnail generation for 61 dives: <60s total
- Browser compatibility check

**Deliverables**:
- [ ] Full workflow tested end-to-end
- [ ] All browsers passing
- [ ] Performance metrics captured
- [ ] Bug report (if any issues found)

**Size**: Medium (testing & QA)

---

## Implementation Order

1. **FEAT-01** → Modal structure (foundation)
2. **FEAT-02** → Timeline extraction (data layer)
3. **FEAT-03** → Timeline display (UI)
4. **FEAT-06** → Info panel (UI)
5. **FEAT-04** → Action buttons (interaction)
6. **FEAT-07** → Keyboard shortcuts (interaction)
7. **FEAT-05** → Auto-advance (workflow) ← depends on all above
8. **FEAT-08** → Testing (validation)

---

## File Structure After Implementation

```
diveanalyzer/utils/review_gallery.py
├── DiveGalleryGenerator
│   ├── extract_thumbnail() [modified - existing]
│   ├── extract_timeline_frames() [NEW]
│   ├── generate_html() [modified - add modal HTML/CSS]
│   └── scan_dives() [modified - call extract_timeline_frames()]
├── HTML Output
│   ├── <style> [add modal CSS]
│   └── <script> [add modal JS]
```

---

## Success Criteria (All Tickets Complete)

- ✅ User clicks any dive card
- ✅ Detailed modal opens with 8-frame timeline
- ✅ Modal shows dive info (number, duration, confidence)
- ✅ User can press `D` to delete
- ✅ Next dive auto-loads in modal immediately
- ✅ Workflow is smooth and feels responsive
- ✅ All keyboard shortcuts work in Safari
- ✅ Stats update correctly as user deletes
- ✅ Last dive deletion shows completion

---

## Notes

- All thumbnails embedded as base64 (no external files)
- No backend API calls needed (pure frontend)
- Stats update in real-time as user makes decisions
- Modal design inspired by photo gallery UX patterns
- Accessibility: keyboard navigation ensures no mouse dependency

