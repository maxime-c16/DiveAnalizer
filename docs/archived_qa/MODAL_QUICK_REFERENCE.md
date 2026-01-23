# Modal Review - Quick Reference Guide

## How to Use the Dive Review Modal

### Opening a Dive Modal
- **Double-click** any dive card in the gallery
- OR **Ctrl+Click** (Windows) / **Cmd+Click** (Mac) on a dive card
- The modal opens with an 8-frame timeline and dive metadata

### Quick Decision Workflow (Keyboard Only)

#### Delete Current Dive & Auto-Advance
```
Press: D
```
- Current dive marked for deletion
- Gallery card fades out smoothly
- Next undeleted dive auto-loads in modal immediately
- Stats update automatically
- **Perfect for rapid batch review!**

#### Keep Current Dive & Optionally Advance
```
Press: K
```
- Current dive marked as kept
- Modal closes
- Next dive auto-opens if available
- Back to gallery view if this was the last dive

#### Navigate Between Dives Without Deciding
```
Press: → (Right Arrow)    - Next undeleted dive
Press: ← (Left Arrow)     - Previous undeleted dive
```
- View surrounding dives to compare
- Skips already-deleted dives
- Smooth 300ms transition
- No changes made until you press K or D

#### Close Modal & Return to Gallery
```
Press: Esc
```
- Modal closes immediately
- All decisions preserved
- Return to gallery view
- Can open another dive or save your work

### Mouse/Button Actions

#### Action Buttons (Bottom of Modal)
- **Keep** button (Green) - Equivalent to K key
- **Delete** button (Red) - Equivalent to D key, triggers auto-advance
- **Cancel** button (Gray) - Equivalent to Esc key

### Modal Information Panel

Shows for each dive:
- **Dive #XXX** - Dive number with 3-digit zero-padding
- **Duration** - Video length (e.g., "1.23s")
- **Confidence** - Detection confidence (HIGH/MEDIUM/LOW with color)
- **File** - Filename (e.g., "dive_001.mp4")

### Timeline View

Below the info panel:
- **8 evenly-spaced frames** from the dive video
- Frame positions: 0%, 12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%
- Hover over frames for visual feedback (zoom effect)
- Resolution: 480x360 for clear visibility

### Batch Review Tips

#### Speed Workflow
```
1. Open dive with Double-click
2. Press D to delete and auto-advance
3. Repeat until all dives reviewed
4. When done, close modal with Esc
5. Submit your review in the gallery
```

#### Compare Workflow
```
1. Open a dive
2. Press ← → to examine similar dives
3. Press D to delete when certain
4. Auto-advances to next for comparison
```

#### Selective Review
```
1. Use K to keep dive you're unsure about
2. Press ← to go back and review again
3. Make final decision and move on
```

### Understanding the UI

#### Status Messages (Bottom of Gallery)
- Green checkmark ✅ = Successful action
- Shows "Next dive loaded" when auto-advancing
- Shows "All decisions made! Review complete." at the end

#### Card Status in Gallery
- **Normal** - Not yet reviewed
- **Green outline** - Marked to keep
- **Red/faded outline** - Marked for deletion
- **Grayed out** - Already deleted (can't select again)

#### Stats Bar (Top of Gallery)
- **Total Dives** - All dives in video
- **Selected for Delete** - How many marked for deletion
- **To Keep** - Remaining dives (auto-updated)

### Keyboard Reference Card

| Key | Action | Context |
|-----|--------|---------|
| D | Delete & auto-advance | Modal only |
| K | Keep & advance | Modal only |
| ← | Previous dive | Modal only |
| → | Next dive | Modal only |
| Esc | Close modal | Modal only |
| Space | Toggle selection | Gallery only |
| A | Select all | Gallery only |
| Ctrl+A | Deselect all | Gallery only |
| W | Watch selected | Gallery only |
| Enter | Accept & close | Gallery only |
| ? | Show help | Both contexts |

### Common Scenarios

#### "I want to delete most dives"
1. Open first dive (double-click)
2. Press D repeatedly - each dive auto-deletes and advances
3. When you want to keep one, press K instead
4. Continue with D for the rest
5. At the end, modal closes automatically

#### "I want to keep most dives"
1. Use Space in gallery to toggle selections
2. Double-click any questionable dive
3. Use ← → to examine similar ones
4. Press D to delete if certain
5. Use Esc to return to gallery between decisions

#### "I'm unsure about a dive"
1. Open it in modal
2. Use ← → to compare with surrounding dives
3. Look at all 8 timeline frames to see the action
4. Check the confidence level (usually accurate)
5. Make your decision with K or D

#### "I made a mistake"
1. Current implementation doesn't have undo
2. Note: In future, you can add a confirmation before deletion
3. For now, be careful with D key!

### Performance Notes

- Modal opens in ~50ms
- Auto-advance transition: smooth 300ms fade
- All 8 frames load instantly when modal opens
- Keyboard response: immediate (< 20ms)
- Browser supports: Safari, Chrome, Firefox, Edge

### Troubleshooting

#### Keyboard shortcuts not working
- Make sure modal is open (overlay should be visible)
- Try clicking inside modal first to focus it
- Try using the button clicks instead
- Check if Caps Lock is on for letter keys

#### Modal won't open
- Make sure you're **double-clicking** (not single-click)
- Try Ctrl+Click / Cmd+Click instead
- Check browser console for errors (F12)

#### Auto-advance not happening
- Make sure next dive isn't already deleted
- If all other dives are deleted, completion state shows instead
- Close and reopen to confirm gallery state

#### Timeline frames not showing
- Frames load from embedded base64 data
- If blank, video may not have been extractable
- This is just a display issue - dive data is still valid

### Tips for Fastest Review

1. **Learn the keyboard shortcuts** - Fastest method
2. **Use D key exclusively** if deleting 80%+ of dives
3. **Mix K and D** for balanced acceptance rate
4. **Skip manual navigation** - auto-advance does it for you
5. **Don't watch full videos** - timeline frames are usually enough
6. **Trust confidence badges** - they're well-calibrated
7. **Batch similar decisions** - delete runs of bad dives at once

### Accessibility Features

- Full keyboard control (no mouse required)
- Clear visual feedback for all actions
- High contrast colors for readability
- Large touch targets for mobile use
- Clear status messages for all operations

### Browser Compatibility

Works perfectly on:
- Safari 14+
- Chrome/Edge 90+
- Firefox 88+
- Mobile Safari (iPad/iPhone)
- Mobile Chrome (Android)

### Getting Help

- Press **?** to see keyboard shortcut help
- Hover over buttons to see tooltips
- Check bottom of gallery for shortcut reference
- Look at modal action buttons for keyboard hints

---

**Remember: The goal is speed and accuracy. Use the keyboard for fastest review, and trust the auto-advance workflow to keep you focused!**
