# TransBloom Android App

This app runs on-device flower classification from camera frames using:
- CameraX (live preview + frame analysis)
- ONNX Runtime Android
- `flowernet.onnx` bundled in app assets

## Project Location

- Android project root: `android-app/`
- Model asset: `android-app/app/src/main/assets/flowernet.onnx`
- Class map asset: `android-app/app/src/main/assets/class_to_idx.json`

## Open and Build APK

1. Open `android-app/` in Android Studio.
2. Let Gradle sync complete.
3. Connect Android device (or start emulator).
4. Build APK:
   - Android Studio: `Build > Build APK(s)`
   - output: `android-app/app/build/outputs/apk/debug/app-debug.apk`

## Install on Device

From terminal:

```bash
adb install -r android-app/app/build/outputs/apk/debug/app-debug.apk
```

Or copy the APK to the phone and install manually.

## Update the Model

If you retrain and export a new ONNX file, replace:

`android-app/app/src/main/assets/flowernet.onnx`

Then rebuild the APK.

## Runtime Behavior

- App shows live camera preview.
- It computes frame-to-frame motion.
- When motion remains low for several frames (still scene), it runs ONNX inference.
- Overlay shows predicted class and confidence.
