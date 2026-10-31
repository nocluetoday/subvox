# Subvox

A macOS prototype that tests whether AirPods Pro head-motion data (`CMHeadphoneMotionManager`) carries enough information to turn **silently mouthed words into text**. The front camera can add lip and jaw landmarks, which are fused with the motion stream.

It doesn't use a language model. It's a small-vocabulary recognizer (DTW templates, or a Core ML sequence model trained in the app on your own takes) plus a statistical test that tells you whether the signal is real. Decoded words can be spoken in your Personal Voice, with a lip-synced avatar, for silent calls.

## Build & run

```sh
brew install xcodegen   # once
xcodegen generate
open Subvox.xcodeproj   # ⌘R
```

Or from the command line:

```sh
xcodebuild -project Subvox.xcodeproj -scheme Subvox -derivedDataPath build -allowProvisioningUpdates build
open build/Build/Products/Debug/Subvox.app
xcodebuild -project Subvox.xcodeproj -scheme Subvox -derivedDataPath build test
```

Requires macOS 14+, AirPods Pro or another headphone that supports head tracking, connected as the Mac's audio output. On first launch, grant **Motion**. The camera, mic and speech prompts only appear when you start the camera or mic.

## Screens

| Screen | What it does |
|---|---|
| **Sensors** | Live plots of user acceleration, rotation rate and attitude, the measured sample rate, a mic input picker and level meter, the camera preview with the tracked lip outline and lip features, and a recorder for experiment sessions. |
| **Train** | Pick a word, hold **Space**, mouth it, release. Each take is stored as a template, with lip landmarks if the camera is on (camera icon on the tile). **Evaluate** runs leave-one-out classification and a 500× label-permutation test, then shows accuracy against chance, a p-value and a confusion matrix. The feature checkboxes (acceleration / rotation / attitude / lips) let you test which channels carry the signal. **Train Model** fits the Core ML sequence model on the same channels and reports accuracy on held-out takes. |
| **Decode** | The motion-to-text screen. Pick the decoder (DTW templates or Core ML model). Hold Space, or use auto-detect after calibrating the noise floor. Each segment is classified and added to the transcript. A live mic transcript runs alongside it for comparison. The **Silent call** panel speaks decoded words and drives the avatar. |

## How decoding works

1. **Segmentation** (`CaptureController`). Space-hold, or an energy trigger above a calibrated noise floor with pre-roll and hangover.
2. **Features** (`MotionFeatures`). Per frame: user acceleration (3), rotation rate (3), and attitude change since the start of the segment (3). Each channel is mean-centered, and quiet lead-in and lead-out frames are trimmed.
3. **Matching** (`DTW`, `TemplateClassifier`). Each channel is z-scaled across the training set. Matching is 1-nearest-neighbour per word, using dynamic time warping with a Sakoe–Chiba band. A segment is rejected if the best distance is above *max distance* or the margin to the runner-up is below *min confidence*.
4. **Lip fusion** (`FaceCameraService`, `LipGeometry`). Vision face landmarks give 7 lip and jaw measures per frame (width, outer and inner opening, upper and lower lip, jaw drop, inner area). They are in face units: divided by the eye distance and measured along the eye line. Frame times are converted to the host clock, so they share the AirPods clock. Lip values are interpolated at each motion timestamp and appended to the motion channels. Lips are used only when at least two words have camera takes; otherwise the decoder falls back to motion only.
5. **Sequence model** (`SequenceModel`). A Create ML activity classifier (convolution + LSTM), trained in the app. Each take is resampled to 32 frames, so one prediction window is one word. Every fifth take of words with 5+ takes is held out for the validation score. The model is saved to `~/Library/Application Support/Subvox/WordModel.mlmodelc`.
6. **Is it real?** (`Evaluation`). The leave-one-out accuracy is compared with the same procedure run on shuffled labels. If p < 0.05, the motion stream distinguishes your words better than chance.

## Experiment protocol

Use **Record Session** in the toolbar or on the Sensors screen. Recordings go to `~/Documents/Subvox/Sessions/<timestamp>-<condition>/`:

- `motion.csv`: every motion sample. `t` is seconds from session start.
- `lips.csv`: every camera lip sample, if the camera is running. Same `t` clock as `motion.csv`.
- `audio.caf`: microphone audio, if the mic is running.
- `events.csv`: every Space-held segment, with its label or prediction.
- `meta.json`: condition, prompt, and the audio start offset for aligning audio with motion.

Suggested runs: the same 5-word script under **Normal speech**, **Whisper** and **Silent mouthing**, 10 reps each. Then, in Train, compare the Evaluate results across feature sets.

The training set lives at `~/Library/Application Support/Subvox/dataset.json`.

## Phrases

1. In **Train › Phrases**, type a phrase made of words you train, for example "call you back". New words are added to the vocabulary. You can also type an output, for example "I'll call you back in five minutes".
2. Train each word as usual (10 takes per word is a good start). A green check shows when every word of a phrase has takes.
3. In **Decode**, choose **Phrases** and mouth the whole phrase in one hold of Space. Auto-detect waits 0.7 s of stillness before it ends a phrase.

`PhraseDecoder` runs one-pass connected-word DTW. It matches each phrase's words in order against all takes of each word, and allows short pauses between words. Only phrases on the list can be decoded. Channels are locally centered over about 1 s, so a word looks the same alone and inside a phrase. New phrases need no retraining.

## Silent calls

1. Create a Personal Voice in **System Settings › Accessibility › Personal Voice** (optional; a system voice is used otherwise). Then click **Use My Personal Voice…** in the Decode screen.
2. To speak into a call app, install a virtual audio cable (for example BlackHole). Choose it as **Output** in the Silent call panel and as the microphone in the call app.
3. Turn on **Speak each decoded word**, or use **Speak Transcript**.
4. Click **Open Avatar Window** and share that window as your video. Use the green background to key it out in OBS.

The avatar's mouth follows the speech audio: loudness sets the opening, and the zero-crossing rate sets lip spread.

## Known limits

- Headphone motion is processed and low-rate (roughly 25 Hz on AirPods). The speech-detecting accelerometer and the inward mic are not exposed. Expect jaw and head movement, not vocal vibration.
- Templates are user- and session-specific. Re-seating the AirPods can shift the signal.
- Choosing AirPods as the mic input switches them to the Bluetooth headset (HFP) profile.
- Lip features are 2D. Lip protrusion (rounding) is only seen indirectly, through width. Lighting and face angle affect landmark quality.
- The sequence model needs many takes. With fewer than about 10 takes per word, DTW is often as good.
- The avatar is a stylized face, not a photo-real likeness. There is no virtual camera; share the avatar window instead.
