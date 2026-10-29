# Viva preparation: Focus Coach

All numbers below are generated from `results/` by `report/build_report.py`. Do not edit them here; rerun the script.

## 3-minute demo script

Check before you start: the room is lit, the camera sees your face and the desk, the phone is on the desk, and the model file `models/best.pt` is in place. If the webcam misbehaves, use the backup: `python -m app.focus_coach --source demo.mp4` (see `app/make_demo_clip.md`).

1. **0:00 Start.** Run `python -m app.focus_coach`. Say: "This watches my webcam, finds my head and any phone, and counts pickups." Point at the tip on screen (camera must see face and desk; nothing is saved but event times). Press **D** to show the boxes. Press **S**. The banner shows FOCUSED.
2. **0:25 Quick glance.** Look at the phone on the desk for about half a second without picking it up. Say: "Nothing happens. A state must hold for about 0.9 of a 1.5-second window, so brief glances are ignored." The pickup counter stays at 0.
3. **0:45 Pickup.** Pick the phone up and look at it for about 5 seconds. The banner turns amber (ON PHONE) and Pickups becomes 1. Put it down. Say: "One event, with a start and a duration."
4. **1:15 Call.** Hold the phone to your ear for about 5 seconds. The banner turns red (ON CALL). Say: "The rules check the phone against the head box grown by 0.1; at the ear it overlaps."
5. **1:45 Low light.** Turn off the room light, or cover most of the camera with a cloth. Say: "When mean brightness drops below 29, the app applies CLAHE + blur; the footer shows it." Press **C** to switch enhancement off and on and point at the footer. If it goes fully dark, say: "Below a certain level nothing works, which my low-light experiment showed."
6. **2:15 End and report.** Turn the light on. Press **E**. Walk through the report: percent focused, phone time, pickups, longest focus streak, the coloured timeline. Say: "Only these numbers and times are saved, not video."
7. **2:45 Close.** Say one result: "On 2,855 held-out images the rules reach 0.873 accuracy; fine-tuning raised phone mAP50 from 0.313 to 0.869." Press **Q**.

## 12 likely questions

**1. What is transfer learning, and why fine-tune instead of using COCO as it is?**
Transfer learning starts from a network that already learned general visual features on a big dataset and adapts it to a new task. COCO knows "cell phone" but has no "head" class, and its phones look different from the FPI-Det photos. Fine-tuning raised phone mAP50 from 0.313 to 0.869, gave a head detector (0.954 mAP50), and took only 1.5 hours on one GPU.

**2. What do mAP and IoU measure? What are your numbers?**
IoU is how much a predicted box overlaps the true box: overlap area divided by combined area. mAP50 is the average precision when a detection counts as correct at IoU 0.5 or more; mAP50-95 averages over stricter thresholds. On the test split the fine-tuned phone detector has mAP50 0.869, mAP50-95 0.604, and the mean IoU of its correct detections is 0.844.

**3. Why rules after detection instead of one classifier?**
I tried the classifier: a MobileNetV2 that looks at the whole image reached 0.736 accuracy against 0.873 for detection plus rules. Shrinking the whole picture to 224 pixels makes small phones almost vanish, and the classifier saw only 979 training images with one label each. The detector also tells where the phone is, which the app needs for ON_CALL and ON_PHONE.

**4. What does CLAHE do differently from global histogram equalization? Did it help?**
Global equalization stretches one histogram over the whole image, which boosts noise and can wash out bright regions. CLAHE does it in small tiles and limits how much any bin can be amplified, so noise grows less. On dim images CLAHE with a light blur raised phone mAP50 from 0.699 to 0.737, while global equalization lowered it to 0.600. On near-black images nothing worked (0.000 at best), and on real dark photos it was not needed.

**5. How does YOLO detect objects in one pass? Why doesn't YOLO26 need an NMS step?**
YOLO runs the image through one network that outputs, for every region of a grid, boxes with class scores, instead of first proposing regions and then classifying them. Older YOLO versions output many overlapping boxes for one object, so non-maximum suppression (NMS) removes duplicates. YOLO26 also has a one-to-one head trained to give one box per object, which needs no NMS. Honest detail: in this project I used the default Ultralytics path, which uses the other head plus NMS, and I did not evaluate the NMS-free mode.

**6. Why does the 1.5-second smoothing exist?**
A single frame can be wrong, and a half-second glance is not a pickup. A state is shown only after it fills 60% of a 1.5-second window, so flicker and short glances create no events. The cost is a delay of about 0.9 seconds at both ends of an event, which mostly cancels in durations. I demonstrated this on a scripted clip but did not measure it on labelled video.

**7. What fails, and why?**
There were 246 false alarms and 116 misses on the held-out images. Of the misses, 80% had no phone detected at all (small or partly hidden phones); the others are phones held low or beside the person, outside the rule zones. Many false alarms and misses are visually the same scene with opposite labels, so some are label noise. I can show the failure grids.

**8. How was data leakage avoided?**
The usage labels were split once, with a fixed seed, into a tune part (1,224 images) and a held-out part (2,855 images). Thresholds and the MobileNet were trained on the tune part only, and results are reported on the held-out part. One small leak remains: about 42% of these images were also used to pick the best detector checkpoint, so the numbers are slightly optimistic.

**9. Your rules did not beat "any phone found". Is the rule step pointless?**
On accuracy they are level (0.873 against 0.876), and I report that honestly. On these photos a visible phone already predicts the label (97% of "using" images have a phone box, 12% of "not using" images). The rules still give what the app needs: separate ON_CALL, ON_PHONE and FOCUSED states, and ignoring phones lying far from the user, which "any phone found" cannot do.

**10. The dataset photos are not from a laptop webcam. Does your result apply?**
Not directly. FPI-Det is mostly surveillance-style photos from above or the side, and I tuned all thresholds on them. The app uses a frontal webcam, where the geometry may differ. This is listed as the main limitation, and the next step would be a small labelled webcam video set to re-tune and to measure the smoothing.

**11. Can it run live on a laptop?**
The whole app pipeline (enhancement check, detector, rules, smoothing, drawing) takes about 26.5 ms per frame, which is 38 frames per second on 13th Gen Intel Core i7-13700HX without a GPU; at the smaller 480-pixel setting it is 45. The camera usually delivers fewer frames than that, so the live rate depends on the camera and the light. I did not measure live rates for the report.

**12. What about privacy and ethics?**
It is for self-monitoring with the user's consent. Frames are processed in memory and not stored; only event times and summary numbers go to disk, and saving annotated video needs an explicit option. A camera watching other people without their knowledge would be a misuse, and I say so in the report.

## Extra questions if you show the Colab demo (Phase 7, optional)

**13. Why is the Colab demo slower than the desktop app?**
Every webcam frame travels from the browser over the internet to the Colab session and the result travels back, and the stream is limited to about five frames per second. The desktop app reads the camera directly and processes the whole pipeline in about 26.5 ms per frame. That is why the demo has a Video tab: an uploaded clip is processed without the live network delay.

**14. Where does the video go?**
To the user's own temporary Colab session, where it is processed in memory and not stored; the session and the public link disappear when the Colab session ends. The desktop app keeps everything on the laptop.

**15. Why Gradio and not Streamlit?**
Gradio has built-in webcam streaming and creates a shareable link from Colab with one setting, with no tunnel to set up. The demo is only a new front end: it calls the same model, rules, smoothing and report code as the desktop app.

## Out-of-syllabus items, and what to say about each

| Item | One line to say out loud |
|---|---|
| Time smoothing (sliding window) | "It keeps the last 1.5 seconds of frame states and only changes the shown state when a new one fills 60% of that window, like a simple vote over time." |
| Precision, recall, F1 | "Precision: of what I called 'using', how much was right. Recall: of what really was 'using', how much I found. F1 combines them." |
| Non-maximum suppression and the one-to-one head | "NMS removes duplicate overlapping boxes for the same object; YOLO26 has an alternative head that gives one box per object." |
| mAP with torchmetrics (and its pycocotools backend) | "A ready-made library that computes mAP the standard way; I only use it for the number." |
| Grid search for thresholds | "I tried every combination from a small list of values on the tune images and kept the best F1." |
| Stratified split (scikit-learn `train_test_split`) | "A random split that keeps the share of 'using' images the same in both parts." |
| Class weights and early stopping (MobileNet) | "Class weights count the rarer class more; early stopping ends training when validation loss stops improving." |
| Gamma correction and noise (to darken images) | "Darkening isn't just multiplying brightness; gamma and noise make the image look like a real dark camera frame." |
| Gradio (Phase 7 demo only) | "A library that turns Python functions into a web page, so the demo runs in a browser; it does not change any of the computer-vision code." |
| Threads (capture) | "One thread reads the camera and keeps only the latest frame, so the display never lags behind." |
