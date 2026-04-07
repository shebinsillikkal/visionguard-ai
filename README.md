# VisionGuard AI

> Real-time computer vision security system that detects unauthorized access, fire/smoke, and behavioral anomalies across live CCTV feeds — alerts within 2 seconds.

## The Origin
Started as a demo I built to test TensorFlow on live CCTV footage. Trained it to spot unauthorized entry and fire. Showed a client, they wanted it installed. It caught two real incidents in the first year.

## Features
- 🔍 Real-time object detection at 30 FPS on edge hardware (Jetson Nano)
- 🔥 Fire & smoke detection with sub-2-second alerting
- 🚨 Unauthorized access detection with zone-based rules
- 📺 Multi-camera dashboard with event timeline and clip playback
- 📲 SMS + WhatsApp + email alerts

## Stack
```
TensorFlow 2.x | OpenCV | Python | FastAPI | MQTT | React (dashboard)
```

## Performance
- 97.3% detection accuracy on test set
- <2 second alert latency from detection to notification
- Runs at 30 FPS on NVIDIA Jetson Nano

## Contact
Built by **Shebin S Illikkal** — [Shebinsillikkal@gmail.com](mailto:Shebinsillikkal@gmail.com)
