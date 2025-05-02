# Vigia: Intelligent Camera Monitoring System

Vigia is an intelligent camera monitoring system that uses YOLOv8 for object detection and provides features such as motion detection, linger detection, and alert notifications via email or Google Home devices.

## Features
- **Object Detection**: Detect specific objects (e.g., people) using YOLOv8.
- **Motion Detection**: Detect significant motion in the camera feed.
- **Linger Detection**: Identify and alert when a person stays in a defined region of interest (ROI) for too long.
- **Notifications**: Send alerts via email or Google Home devices.
- **Customizable Configuration**: Configure cameras, detection parameters, and alerting options via `config.yaml`.

## Requirements
- Python 3.10 or higher
- Required Python libraries (see `requirements` file):
  - `torch`, `torchvision`, `torchaudio`
  - `opencv-python`
  - `ultralytics`
  - `pychromecast`

## Setup

### 1. Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements
```

### 2. Configure the System
Edit the `config.yaml` file to define your cameras, detection parameters, and alerting options. Example:
```yaml
cameras:
  - name: "Door cam"
    url: rtsp://rtsp:password@ip/av_stream/ch0
    linger_detection:
      enabled: true
      roi: [700, 400, 1050, 1000]
      linger_time_seconds: 5
      tracking_distance_threshold: 75
detection:
  model: "yolov8n.pt"
  classes_to_detect: ["person", "dog", "cat"]
  confidence_threshold: 0.5
alerting:
  cooldown_seconds: 60
  save_directory: "detections"
  google_home:
    enabled: true
    device_name: "kitchen"
    sound_server_url: "http://ip:port"
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    sender_email: "your_email@gmail.com"
    sender_password: "password"
    recipient_email: "destination@mail.com"
```

### 3. Prepare Audio Files
Place the audio files (e.g., `person.mp3`) in a directory and serve them using an HTTP server. For example:
```bash
cd audios
python -m http.server 8000
```
Ensure the `sound_server_url` in `config.yaml` points to this server.

### 4. Run the Application
Run the main monitoring script:
```bash
python main_monitor.py
```

## Docker Support
Build and run the application using Docker:
1. Build the Docker image:
   ```bash
   docker build -t vigia .
   ```
2. Run the container:
   ```bash
   docker run -it --rm -v $(pwd)/detections:/app/detections vigia
   ```

## Project Structure
- `app.py`: Example script for testing YOLOv8 and Chromecast integration.
- `main_monitor.py`: Main monitoring script.
- `camera_processor.py`: Handles camera streams and detection logic.
- `notifications.py`: Notification handlers (e.g., email, Google Home).
- `config_loader.py`: Loads and validates the configuration file.
- `config.yaml`: Configuration file for cameras, detection, and alerts.
- `requirements`: List of required Python libraries.
- `Dockerfile`: Docker configuration for the project.

## Notes
- Ensure the YOLOv8 model file (`yolov8n.pt`) is available or can be downloaded by the `ultralytics` library.
- The `detections` directory will store images of detected objects.
- Use `Ctrl+C` to gracefully stop the monitoring system.

## License
This project is licensed under the MIT License.
