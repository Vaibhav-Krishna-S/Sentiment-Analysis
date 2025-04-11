# Sentiment-Analysis
Real-time facial sentiment analysis using OpenCV and MediaPipe. This project detects facial landmarks through the webcam and analyzes expressions like happiness, surprise, and neutrality by calculating facial feature ratios such as mouth openness, smile width, and eyebrow distance.

# 😊 Facial Sentiment Analysis using OpenCV & MediaPipe

This project performs **real-time sentiment analysis** based on facial expressions using computer vision. It uses **OpenCV** for video capture and **MediaPipe** to detect and track facial landmarks — interpreting emotions like *Happy*, *Neutral*, or *Surprised* from facial geometry.

---

## 📸 How It Works

- Uses your webcam to capture live video.
- Detects facial landmarks using MediaPipe's Face Mesh.
- Analyzes the **mouth-to-eye distance ratio** to predict sentiment.
- Displays the detected mood (Happy, Neutral, Surprised) in real time.

---

## 🧠 Tech Stack

- Python 🐍
- OpenCV
- MediaPipe

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone git@github.com:Vaibhav-Krishna-S/Sentiment-Analysis.git
cd Sentiment-Analysis

