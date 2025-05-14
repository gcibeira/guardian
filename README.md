# Guardian: Intelligent Camera Monitoring System

Guardian is an intelligent camera monitoring system that uses YOLOv8 for object detection and provides advanced features such as motion detection, linger detection, persistent object tracking, and alert notifications via email or Google Home devices.

## Features

- **Object Detection**: Detect specific objects (e.g., people) using YOLOv8 with configurable classes and confidence thresholds.
- **Motion Detection**: Detect significant motion in the camera feed using frame differencing and contour analysis.
- **Linger Detection**: Identify and alert when a person or object stays in a defined region of interest (ROI) for too long.
- **Persistent Object Tracking**: Assigns persistent IDs to detected objects, allowing for robust tracking even with brief occlusions or missed detections.
- **Notifications**: Send alerts via email (with image attachment) or Google Home devices (audio playback).
- **Customizable Configuration**: Configure cameras, detection parameters, tracking, and alerting options via a YAML file.
- **Multi-camera Support**: Process multiple camera streams in parallel, each with its own configuration.
- **Automatic Reconnection**: Intelligent reconnection to camera streams on failure.
- **Snapshot Saving**: Save annotated images of linger events to a configurable directory.
- **Docker Support**: Easily deploy the system in a containerized environment.
- **Graceful Shutdown**: Handles termination signals for clean shutdown of all threads and resources.

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

Create the `config.yaml` file to define your cameras, detection parameters, and alerting options. Example:
```yaml
cameras:
  - name: "Door Camera"
    url: rtsp://rtsp:password@cam_ip:port/av_stream/ch0
    confidence_threshold: 0.5
    classes_to_detect: ["person"]
    motion_detection:
      enabled: true
      min_area: 2500
      threshold: 15
      blur_kernel: [21, 21]
    linger_detection:
      enabled: true
      roi: [800, 400, 1450, 1000]
      linger_time_seconds: 5
      tracking_distance_threshold: 150
      max_missing_frames: 5
    alert_cooldown_seconds: 60
    save_directory: "detections"

detection:
  model: "yolov8n.pt"
  classes_to_detect: ["person"]
  confidence_threshold: 0.5
  skip_frames: 4
  force_interval: 20
  motion_detection:
    enabled: true
    min_area: 2500
    threshold: 15
    blur_kernel: [21, 21]

alerting:
  cooldown_seconds: 60
  save_directory: "detections"
  google_home:
    enabled: true
    device_name: "Assistant"
    sound_server_url: "http://ip:port"
  email:
    enabled: false
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    sender_email: "your_email@gmail.com"
    sender_password: "your_password"
    recipient_email: "destination@example.com"
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
python main.py --config config.yaml --display --log INFO
```

#### Command-line Arguments

- `-c`, `--config`: Path to the YAML configuration file. Default: `config.yaml`
- `--display`: Show a video window for each camera with overlays. Omit for headless operation.
- `--log`: Logging level. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`. Default: `INFO`

Example:
```bash
python main.py --config config.yaml --display --log DEBUG
```

## Docker Support

Build and run the application using Docker:

1. Build the Docker image:
   ```bash
   docker build -t guardian .
   ```
2. Run the container:
   ```bash
   docker run -it --rm -v $(pwd)/detections:/app/detections guardian
   ```

## Project Structure

- `main.py`: Main orchestration script and entry point.
- `camera_processor.py`: Handles per-camera processing, detection, tracking, and notifications.
- `camera_manager.py`: Manages camera stream connections and reconnections.
- `motion_detector.py`: Detects significant motion in video frames.
- `object_detector.py`: YOLOv8 wrapper for object detection.
- `tracker.py`: Assigns persistent IDs to detected objects and manages object tracking.
- `linger_detector.py`: Detects and emits linger events for objects in ROI.
- `overlay_renderer.py`: Draws bounding boxes, IDs, and linger timers on frames.
- `notifications.py`: Notification handlers (email, Google Home, or no-op).
- `config_loader.py`: Loads and validates the configuration file.
- `requirements`: List of required Python libraries.
- `Dockerfile`: Docker configuration for the project.
- `detections/`: Directory for saving detection snapshots.
- `audios/`: Directory for audio files to be played on Google Home.

## Notes

- Ensure the YOLOv8 model file (`yolov8n.pt`) is available or can be downloaded by the `ultralytics` library.
- The `detections` directory will store images of detected linger events.
- Use `Ctrl+C` to gracefully stop the monitoring system.
- Each camera can have its own detection, motion, and linger configuration, or inherit from global settings.

## License

This project is licensed under the MIT License.
