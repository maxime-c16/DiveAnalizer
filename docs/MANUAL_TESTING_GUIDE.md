# Manual Testing Guide - Modal Implementation

**Purpose:** Final verification steps to test the modal in real browsers before production
**Status:** Code review complete ✅ → Manual browser testing needed

---

## Getting Started

### 1. Generate Test Gallery
```bash
cd /Users/mcauchy/workflow/DiveAnalizer
python3 diveanalyzer/utils/review_gallery.py extracted_test IMG_6497.MOV
```

### 2. Open in Browser
```bash
# macOS
open extracted_test/review_gallery.html

# Or manually open the file:
# /Users/mcauchy/workflow/DiveAnalizer/extracted_test/review_gallery.html
```

---

## Test Suite 1: Basic Modal Functionality

### Test 1.1: Double-Click Opens Modal
**Steps:**
1. Gallery page loads with 61 dive cards
2. Double-click any dive card
3. Modal overlay appears with fade-in animation

**Expected Results:**
- ✅ Modal overlays appear smoothly
- ✅ Background darkens with semi-transparent overlay
- ✅ Modal container centered on screen
- ✅ Fade animation is smooth (not jarring)

**Acceptance:** Pass if all 4 criteria met

---

### Test 1.2: Modal Contains Expected Sections
**Steps:**
1. Modal is open (from Test 1.1)
2. Visually inspect the modal

**Expected Results:**
- ✅ Modal header visible with dive title (e.g., "Dive #005")
- ✅ Close button (X) in top-right corner
- ✅ Timeline section showing 8 frames horizontally
- ✅ Info panel on right or below showing:
  - Duration (e.g., "1.23s")
  - Confidence level (HIGH/MEDIUM/LOW with color badge)
  - Filename (e.g., "dive_001.mp4")
- ✅ Three buttons at bottom: Keep, Delete, Cancel

**Acceptance:** Pass if all sections present and readable

---

### Test 1.3: Timeline Frames Display
**Steps:**
1. Modal is open
2. Look at the timeline section

**Expected Results:**
- ✅ Exactly 8 thumbnail frames visible
- ✅ Frames are arranged horizontally
- ✅ All frames show clear images (not blank/broken)
- ✅ Frames progress from start to end of dive
- ✅ Frames fit without requiring horizontal scrolling

**Acceptance:** Pass if all frames visible and clear

---

### Test 1.4: Hover Effects
**Steps:**
1. Modal is open
2. Hover mouse over timeline frames
3. Hover mouse over Keep/Delete/Cancel buttons

**Expected Results:**
- ✅ Timeline frames zoom slightly on hover
- ✅ Buttons change color/appearance on hover
- ✅ Effects smooth (not jumpy)

**Acceptance:** Pass if hover effects smooth

---

## Test Suite 2: Keyboard Navigation

### Test 2.1: D Key (Delete)
**Steps:**
1. Modal is open
2. Press D key on keyboard
3. Observe modal behavior

**Expected Results:**
- ✅ No console errors (check DevTools)
- ✅ Current dive card fades out in gallery
- ✅ Next dive loads in modal (< 1 second)
- ✅ Timeline updates to show new dive's 8 frames
- ✅ Info panel updates to new dive's data
- ✅ Keep/delete counts update

**Acceptance:** Pass if next dive loads smoothly

---

### Test 2.2: K Key (Keep)
**Steps:**
1. Modal is open with a different dive
2. Press K key on keyboard

**Expected Results:**
- ✅ No console errors
- ✅ Current dive card does NOT fade out
- ✅ Next dive loads in modal
- ✅ Can continue with K and D keys repeatedly
- ✅ Keep count increases when using K

**Acceptance:** Pass if workflow smooth

---

### Test 2.3: Esc Key (Close Modal)
**Steps:**
1. Modal is open
2. Press Esc key

**Expected Results:**
- ✅ Modal closes smoothly with fade-out
- ✅ Return to gallery view
- ✅ No changes to any dives

**Acceptance:** Pass if modal closes

---

### Test 2.4: Arrow Keys (Navigate)
**Steps:**
1. Modal is open
2. Press Right arrow key
3. Press Left arrow key

**Expected Results:**
- ✅ Right arrow: Loads next undeleted dive
- ✅ Left arrow: Loads previous undeleted dive
- ✅ No dives are marked as deleted
- ✅ Navigate is smooth and fast
- ✅ Can't go past first/last dive

**Acceptance:** Pass if navigation works

---

### Test 2.5: Help Menu (?)
**Steps:**
1. Modal is open
2. Press ? key

**Expected Results:**
- ✅ Help popup or alert appears
- ✅ Shows list of keyboard shortcuts
- ✅ Shortcuts are readable
- ✅ Can close help and continue

**Acceptance:** Pass if help displays

---

## Test Suite 3: Mouse Interaction

### Test 3.1: Keep Button Click
**Steps:**
1. Modal is open
2. Click "Keep" button with mouse

**Expected Results:**
- ✅ Button shows active/pressed state
- ✅ Next dive loads in modal
- ✅ Current dive card not marked for deletion

**Acceptance:** Pass if next dive loads

---

### Test 3.2: Delete Button Click
**Steps:**
1. Modal is open
2. Click "Delete" button with mouse

**Expected Results:**
- ✅ Button shows active/pressed state
- ✅ Current dive card fades out
- ✅ Next dive loads in modal
- ✅ Delete count increases

**Acceptance:** Pass if next dive loads with animation

---

### Test 3.3: Cancel Button Click
**Steps:**
1. Modal is open
2. Click "Cancel" button

**Expected Results:**
- ✅ Modal closes
- ✅ No changes to dives
- ✅ Return to gallery

**Acceptance:** Pass if modal closes

---

### Test 3.4: Close Button (X) Click
**Steps:**
1. Modal is open
2. Click X button in top-right corner

**Expected Results:**
- ✅ Modal closes
- ✅ Same result as Cancel button

**Acceptance:** Pass if modal closes

---

### Test 3.5: Click Outside Modal
**Steps:**
1. Modal is open
2. Click on dark overlay (outside modal box)

**Expected Results:**
- ✅ Modal closes
- ✅ Same result as Cancel button

**Acceptance:** Pass if modal closes

---

## Test Suite 4: End-to-End Workflow

### Test 4.1: Delete All Dives
**Steps:**
1. Open gallery
2. Open first dive modal (double-click)
3. Repeatedly press D to delete dives
4. Continue until reaching last dive
5. Delete last dive

**Expected Results:**
- ✅ Each delete smooth and responsive (< 500ms per dive)
- ✅ Transitions smooth (no lag)
- ✅ All cards fade out properly
- ✅ Stats update correctly throughout
- ✅ After last dive deleted:
  - Modal closes or shows completion message
  - Stats show all dives deleted
  - Keep count shows 0

**Acceptance:** Pass if workflow smooth throughout

---

### Test 4.2: Keep All Dives
**Steps:**
1. Open gallery
2. Open first dive modal
3. Repeatedly press K to keep all dives
4. Continue until all 61 dives reviewed

**Expected Results:**
- ✅ Each dive advances smoothly
- ✅ No dives marked for deletion
- ✅ Keep count increases with each K press
- ✅ At end: All dives still in gallery

**Acceptance:** Pass if all dives marked as keep

---

### Test 4.3: Mixed Keep/Delete
**Steps:**
1. Open gallery
2. Open first dive modal
3. Use mix of:
   - Press D to delete some
   - Press K to keep others
   - Click buttons instead of keyboard
   - Use Esc to close and re-open
4. Delete ~30 of 61 dives randomly

**Expected Results:**
- ✅ All actions work correctly
- ✅ Stats accurate (kept vs. deleted count)
- ✅ Deleted cards fade out
- ✅ Kept cards remain visible
- ✅ Workflow feels natural

**Acceptance:** Pass if all interactions smooth

---

## Test Suite 5: Mobile Responsiveness

### Test 5.1: Mobile View (Portrait)
**Steps:**
1. Open gallery in browser on mobile phone (portrait)
2. Double-click dive card (or tap)
3. Observe modal

**Expected Results:**
- ✅ Gallery cards visible in portrait layout
- ✅ Modal opens and fits screen (90vw width)
- ✅ Timeline frames visible (may need horizontal scroll)
- ✅ Info panel visible
- ✅ Buttons tappable (not too small)
- ✅ No text cut off
- ✅ Can use D/K keys on mobile keyboard

**Acceptance:** Pass if modal usable on mobile

---

### Test 5.2: Mobile View (Landscape)
**Steps:**
1. Rotate mobile to landscape
2. Modal should adapt

**Expected Results:**
- ✅ Modal adapts to landscape width
- ✅ All elements still visible
- ✅ Timeline fits better in landscape
- ✅ All buttons still accessible

**Acceptance:** Pass if layout adapts

---

## Test Suite 6: Performance & Stability

### Test 6.1: Rapid Deletion
**Steps:**
1. Open modal
2. Press D key repeatedly very fast (10+ times)
3. Observe behavior

**Expected Results:**
- ✅ Each delete completes
- ✅ No double-deletion issues
- ✅ No modal gets "stuck"
- ✅ Transition guard prevents race conditions
- ✅ All dives processed correctly

**Acceptance:** Pass if no crashes or hangs

---

### Test 6.2: Long Session (All 61 Dives)
**Steps:**
1. Review all 61 dives by pressing D repeatedly
2. Monitor performance throughout
3. Check browser memory usage if possible (DevTools)

**Expected Results:**
- ✅ No performance degradation as you progress
- ✅ Modal responsive at end (not at dive #50)
- ✅ No memory leaks (RAM stable)
- ✅ Smooth transitions throughout
- ✅ Completion message at the end

**Acceptance:** Pass if performance stable throughout

---

### Test 6.3: Rapid Open/Close
**Steps:**
1. Open modal (double-click)
2. Close modal (Esc)
3. Open again (double-click)
4. Repeat 10+ times

**Expected Results:**
- ✅ No console errors
- ✅ Modal opens every time
- ✅ Content correct on each open
- ✅ Smooth animations each time
- ✅ No stuck modals or overlays

**Acceptance:** Pass if modal stable

---

## Test Suite 7: Browser-Specific Tests

### Test 7.1: Safari
**Steps:**
1. Open gallery in Safari (macOS or iOS)
2. Run full Test Suite 1-4

**Expected Results:**
- ✅ All features work
- ✅ No JavaScript errors
- ✅ Keyboard shortcuts work (especially D, K, Esc)
- ✅ No conflicts with Safari browser shortcuts
- ✅ Animations smooth

**Acceptance:** Pass if all features work

---

### Test 7.2: Chrome
**Steps:**
1. Open gallery in Chrome
2. Run full Test Suite 1-4
3. Open DevTools (F12) and check Console for errors

**Expected Results:**
- ✅ All features work
- ✅ Console shows no errors (only debug logs)
- ✅ Performance metrics good
- ✅ DevTools shows no warnings

**Acceptance:** Pass if no console errors

---

### Test 7.3: Firefox
**Steps:**
1. Open gallery in Firefox
2. Run full Test Suite 1-4
3. Open DevTools (F12) and check Console

**Expected Results:**
- ✅ All features work
- ✅ No console errors
- ✅ Smooth animations

**Acceptance:** Pass if all features work

---

## Test Suite 8: Data Integrity

### Test 8.1: Gallery-Modal Sync
**Steps:**
1. Open modal for dive #10
2. Press D to delete
3. Check gallery: Card #10 should fade out
4. Check modal: Should show dive #11
5. Close modal
6. Reopen dive #10 (try to find it in gallery)

**Expected Results:**
- ✅ Card #10 marked as deleted in gallery
- ✅ Can't reopen deleted dive
- ✅ Gallery and modal stay in sync
- ✅ Stats show correct deleted count

**Acceptance:** Pass if data consistent

---

### Test 8.2: Thumbnail Consistency
**Steps:**
1. In gallery: Note the 3 thumbnails for a dive
2. Open modal for that dive
3. Compare the 8-frame timeline

**Expected Results:**
- ✅ The 8-frame timeline shows progression
- ✅ Gallery thumbnails match (start, middle, end)
- ✅ Frames are consistent (no wrong dive shown)

**Acceptance:** Pass if data consistent

---

## Test Suite 9: Accessibility

### Test 9.1: Keyboard-Only Navigation
**Steps:**
1. Use ONLY keyboard (no mouse)
2. Tab through gallery
3. Open modal with Enter/Double-click
4. Use D/K keys to review all dives
5. Never touch mouse

**Expected Results:**
- ✅ All operations possible with keyboard
- ✅ Focus visible on interactive elements
- ✅ No mouse needed
- ✅ Workflow complete without mouse

**Acceptance:** Pass if keyboard-only works

---

### Test 9.2: Screen Reader (Optional)
**Steps:**
1. Enable VoiceOver (macOS) or NVDA (Windows)
2. Navigate modal

**Expected Results:**
- ✅ Dive number announced
- ✅ Info panel values announced
- ✅ Buttons labeled clearly
- ✅ No screen reader confusion

**Acceptance:** Nice to have but not required

---

## Test Suite 10: Edge Cases

### Test 10.1: All Dives Same Duration
**Steps:**
1. Open multiple dives
2. Check if info panel updates properly

**Expected Results:**
- ✅ Each dive's duration displayed correctly
- ✅ No confusion between dives

---

### Test 10.2: Confidence Level Display
**Steps:**
1. Open dives with different confidence levels
2. Check badge colors

**Expected Results:**
- ✅ HIGH: Green badge
- ✅ MEDIUM: Yellow/orange badge
- ✅ LOW: Red/orange badge
- ✅ Colors clearly distinguished

---

### Test 10.3: Long Filenames
**Steps:**
1. Check modal filename field
2. If filename is very long

**Expected Results:**
- ✅ Text doesn't overflow
- ✅ Text is readable (not cut off)
- ✅ Layout stays intact

---

## Reporting Issues

### If You Find a Problem

**Document the following:**

1. **Browser & Version**
   - e.g., "Safari 17.2 on macOS 14.2"

2. **Steps to Reproduce**
   - Exact steps to recreate issue

3. **Expected Result**
   - What should happen

4. **Actual Result**
   - What actually happened

5. **Screenshot/Video** (if possible)
   - Visual proof of issue

6. **Console Errors** (if any)
   - Copy JavaScript errors from DevTools Console

7. **Severity**
   - Critical (blocks workflow)
   - Major (significant issue)
   - Minor (cosmetic)

### Example Issue Report
```
Browser: Safari 17.2 on macOS 14.2
Steps:
  1. Open gallery
  2. Double-click dive #3
  3. Press D key 5 times
  4. Modal content shows wrong dive

Expected: Each D press should show next dive
Actual: Modal shows dive #5 instead of #4
Console errors: None visible

Severity: Major
```

---

## Success Criteria

✅ **All tests pass** = Ready for production

⚠️ **Minor issues only** = Can deploy with workarounds documented

❌ **Blocking issues found** = Requires fixes before deployment

---

## Quick Checklist

Print this and check off as you test:

```
BASIC MODAL
☐ Modal opens on double-click
☐ Modal closes on Esc
☐ Timeline shows 8 frames
☐ Info panel shows data
☐ Buttons are clickable

KEYBOARD
☐ D key deletes and advances
☐ K key keeps and advances
☐ Esc closes modal
☐ Arrows navigate
☐ ? shows help

MOUSE
☐ Keep button works
☐ Delete button works
☐ Cancel button works
☐ X button closes

WORKFLOW
☐ Delete all 61 dives smoothly
☐ Keep all 61 dives smoothly
☐ Mixed keep/delete works
☐ Stats update correctly

PERFORMANCE
☐ Modal opens instantly
☐ Auto-advance smooth (< 1s)
☐ No lag on keyboard input
☐ No memory leaks

MOBILE
☐ Responsive on iPhone portrait
☐ Responsive on iPhone landscape
☐ Buttons tappable
☐ Timeline visible

BROWSERS
☐ Safari works
☐ Chrome works
☐ Firefox works

DATA INTEGRITY
☐ Gallery-modal in sync
☐ Stats accurate
☐ No duplicate dives
☐ All 61 dives accounted for

ACCESSIBILITY
☐ Keyboard-only workflow works
☐ No mouse required
☐ Focus visible

COMPLETION
☐ No console errors
☐ Smooth throughout
☐ Professional appearance
☐ Ready for production
```

---

## Time Estimates

| Test Suite | Estimated Time |
|------------|-----------------|
| Suite 1 (Basic) | 5 minutes |
| Suite 2 (Keyboard) | 10 minutes |
| Suite 3 (Mouse) | 5 minutes |
| Suite 4 (End-to-End) | 15 minutes |
| Suite 5 (Mobile) | 10 minutes |
| Suite 6 (Performance) | 10 minutes |
| Suite 7 (Browsers) | 15 minutes |
| Suite 8 (Data) | 5 minutes |
| Suite 9 (Accessibility) | 5 minutes |
| Suite 10 (Edge Cases) | 5 minutes |
| **Total** | **~85 minutes** |

---

## Next Actions

1. **Before Testing:**
   - Generate fresh gallery
   - Clear browser cache
   - Have DevTools ready

2. **During Testing:**
   - Take notes on any issues
   - Keep console visible
   - Test all browsers

3. **After Testing:**
   - Report any issues found
   - Document workarounds
   - Get sign-off before production

---

**Document Version:** 1.0
**Date:** 2026-01-21
**Status:** Ready for Manual Testing
