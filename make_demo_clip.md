# Recording a backup demo clip

If the webcam misbehaves during the viva, the app can run from a recording instead:
`python -m app.focus_coach --source demo.mp4`. Record one 60-second clip of yourself at your desk, with the same camera
and lighting you will demo in. Use the Windows Camera app (or any recorder that saves .mp4), sit where you normally study,
and do this in order: study quietly for about 15 seconds (focused), glance at your phone for half a second without picking
it up, then pick the phone up and look at it for about 10 seconds, put it down and study for 10 seconds, hold the phone to
your ear for about 10 seconds (a call), and put it down and study for the last few seconds. Save the file as `demo.mp4` in
the repo root. When you play it, press S to start the session, and E at the end to see the report: you should see
two pickups (the phone, then the call), no event for the half-second glance, and a focus streak between them.
