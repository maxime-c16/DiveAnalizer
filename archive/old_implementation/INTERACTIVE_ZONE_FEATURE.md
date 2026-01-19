# Interactive Splash Zone Selection Feature

## 🎯 Overview

I've successfully added interactive splash zone selection to the splash-only detector, similar to the functionality in `slAIcer.py`. This feature allows users to visually select the splash detection zone using mouse click and drag operations.

## ✨ New Features Added

### 1. Interactive Zone Selection Function
- **Function**: `get_splash_zone_interactive(video_path)`
- **Visual interface** with mouse click and drag
- **Real-time preview** of selected zone
- **Smart scaling** for large videos
- **Automatic coordinate conversion** to normalized values

### 2. Command Line Integration
- **New argument**: `--interactive-zone`
- **Backward compatible** with existing `--zone TOP BOTTOM` option
- **Graceful fallback** to default zone if selection fails

### 3. Enhanced User Experience
- **Visual feedback** during selection
- **Zone preview** with statistics
- **Clear instructions** and controls
- **ESC key** for default zone

## 🎮 How to Use

### Interactive Zone Selection
```bash
# Basic interactive selection
python3 splash_only_detector.py video.mp4 --interactive-zone

# Interactive with debug mode
python3 splash_only_detector.py video.mp4 --interactive-zone --debug

# Interactive with custom parameters
python3 splash_only_detector.py video.mp4 --interactive-zone --method combined --threshold 15.0
```

### Manual Zone Selection (Existing)
```bash
# Manual coordinates
python3 splash_only_detector.py video.mp4 --zone 0.75 0.9

# Default zone (no arguments)
python3 splash_only_detector.py video.mp4
```

## 🖱️ Interactive Controls

### Mouse Operations
- **Click and drag**: Select splash detection zone
- **Real-time preview**: See zone as you drag
- **Visual feedback**: Highlighted zone with statistics

### Keyboard Controls
- **ESC**: Use default zone (0.7 - 0.95)
- **Window close**: Complete selection

### Visual Feedback
- **Gray line**: Current mouse position
- **Purple rectangle**: Selected zone during drag
- **Zone statistics**: Height and coordinates displayed
- **Instructions**: On-screen guidance

## 📊 Zone Selection Output

### Console Information
```
✅ Splash zone selected: 974-1173 pixels
📊 Normalized coordinates: 0.761 - 0.916
📏 Zone height: 199 pixels (15.5% of frame)
🔍 Showing preview of selected zone for 3 seconds...
```

### Zone Preview
- **3-second preview** of selected zone
- **Yellow overlay** highlighting the area
- **Statistics display** on preview
- **Coordinate information** shown

## 🔧 Technical Implementation

### Core Components

1. **Mouse Event Handling**
```python
def on_mouse(event, x, y, _flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing['active'] = True
        drawing['start_y'] = actual_y
    elif event == cv2.EVENT_LBUTTONUP and drawing['active']:
        # Complete selection
```

2. **Real-time Display Updates**
```python
while clicked['top_y'] is None or clicked['bottom_y'] is None:
    # Only redraw if something changed
    if mouse_pos['changed'] or last_state != current_state:
        # Update display with current selection
```

3. **Coordinate Normalization**
```python
# Convert pixel coordinates to normalized values
top_norm = clicked['top_y'] / h
bottom_norm = clicked['bottom_y'] / h
```

### Smart Scaling
- **Automatic resizing** for large videos (>1080p)
- **Coordinate scaling** between display and original resolution
- **Maintains precision** regardless of display size

## 🎯 Benefits

### For New Videos
- **Visual verification** of zone placement
- **Precise selection** based on actual content
- **Immediate feedback** on zone appropriateness

### For Different Camera Angles
- **Adaptable** to various viewing angles
- **Custom zones** for unique setups
- **Optimized detection** for specific scenarios

### For Batch Processing
- **Test interactively** first to find optimal zone
- **Use coordinates** for batch processing later
- **Consistent results** across similar videos

## 📋 Usage Examples

### Quick Testing
```bash
# Test detection with interactive zone selection
python3 splash_only_detector.py dive_video.mp4 --interactive-zone --no-extract --debug
```

### Production Processing
```bash
# After finding optimal zone interactively, use coordinates for batch
python3 splash_only_detector.py dive_video.mp4 --zone 0.761 0.916 --method motion_intensity
```

### Comparison Testing
```bash
# Test different zones
python3 splash_only_detector.py dive_video.mp4 --interactive-zone --debug
python3 splash_only_detector.py dive_video.mp4 --zone 0.7 0.95 --debug
```

## 🎪 Advanced Features

### Zone Validation
- **Minimum height**: Prevents too-small zones
- **Coordinate validation**: Ensures top < bottom
- **Error handling**: Graceful fallback to defaults

### Visual Enhancements
- **Real-time statistics**: Zone height and percentage
- **Color coding**: Purple for selection, yellow for preview
- **On-screen instructions**: Clear user guidance

### Integration
- **Seamless integration** with existing parameters
- **Preserves all functionality** of original system
- **Enhanced debugging** with zone information

## 🔄 Workflow Integration

### Recommended Workflow
1. **Start with interactive selection** for new videos
2. **Note the optimal coordinates** from console output
3. **Use coordinates** for batch processing similar videos
4. **Adjust thresholds** based on debug plots
5. **Fine-tune parameters** for optimal detection

This feature significantly enhances the usability of the splash-only detection system by providing visual feedback and precise control over the detection zone, similar to the original `slAIcer.py` implementation but integrated into the new pure splash detection paradigm.
