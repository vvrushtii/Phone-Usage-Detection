# Demo examples

Made by `python demo/make_examples.py` from real FPI-Det photos (held-out part of `label.csv`). Faces in the photos are masked by the dataset.

| File | What it shows |
|---|---|
| `images/01_focused.jpg` | one person, no phone: FOCUSED |
| `images/02_on_call.jpg` | phone at the ear: ON_CALL |
| `images/03_on_phone.jpg` | phone held in front: ON_PHONE |
| `images/04_tricky_labelled_not_using.jpg` | the dataset label says "not using", but a phone is in front of the person, so the rules say "using". One of the label-noise cases from the report |
| `scripted_clip.mp4` | a scripted 38 s clip made from three of these photos: focused, 5 s phone, 0.5 s glance (ignored), 5 s call, covered camera (AWAY). Not a recording of a person |

To use your own clip in the Video tab, record it as described in `app/make_demo_clip.md` and save it here as `demo.mp4`.
