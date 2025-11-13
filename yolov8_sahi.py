# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from ultralytics.utils.files import increment_path


class SAHIInference:
    """Runs Ultralytics YOLO11 and SAHI for object detection on video with options to view, save, and track results.

    This class integrates SAHI (Slicing Aided Hyper Inference) with YOLO11 models to perform efficient object detection
    on large images by slicing them into smaller pieces, running inference on each slice, and then merging the results.

    Attributes:
        detection_model (AutoDetectionModel): The loaded YOLO11 model wrapped with SAHI functionality.

    Methods:
        load_model: Load a YOLO11 model with specified weights for object detection using SAHI.
        inference: Run object detection on a video using YOLO11 and SAHI.
        parse_opt: Parse command line arguments for the inference process.

    Examples:
        Initialize and run SAHI inference on a video
        >>> sahi_inference = SAHIInference()
        >>> sahi_inference.inference(weights="yolo11n.pt", source="video.mp4", view_img=True)
    """

    def __init__(self):
        """Initialize the SAHIInference class for performing sliced inference using SAHI with YOLO11 models."""
        self.detection_model = None

    def load_model(self, weights: str, device: str) -> None:
        """Load a YOLO11 model with specified weights for object detection using SAHI.

        Args:
            weights (str): Path to the model weights file.
            device (str): CUDA device, i.e., '0' or '0,1,2,3' or 'cpu'.
        """
        from ultralytics.utils.torch_utils import select_device

        # FIX: Use the provided path directly to resolve FileNotFoundError
        yolo11_model_path = weights  
        # download_model_weights(yolo11_model_path)  # Download model if not present
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=yolo11_model_path, device=select_device(device)
        )

    # --- OpenCV Drawing Function ---
    def draw_bboxes(self, frame, results, hide_conf, bbox_thickness, hide_labels):
        """Draws bounding boxes and optional labels/confidences on the frame."""
        # Use a consistent color for all boxes (e.g., green: BGR 0, 255, 0)
        color = (0, 255, 0) 
        annotated_frame = frame.copy()

        for prediction in results.object_prediction_list:
            # Get coordinates [x_min, y_min, x_max, y_max]
            bbox = prediction.bbox.to_xyxy()
            x_min, y_min, x_max, y_max = [int(p) for p in bbox]

            # 1. Draw Bounding Box (Rectangle)
            cv2.rectangle(
                annotated_frame, 
                (x_min, y_min), 
                (x_max, y_max), 
                color, 
                thickness=bbox_thickness # Use custom thickness
            )

            # 2. Add Label/Confidence (Only if not hidden)
            if not hide_labels or not hide_conf:
                label_parts = []
                if not hide_labels:
                    label_parts.append(prediction.category.name)
                if not hide_conf:
                    label_parts.append(f"{prediction.score.value:.2f}")

                label = " ".join(label_parts)
                
                # Setup label background
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_thickness)[0]
                text_w, text_h = text_size
                margin = 5
                
                # Draw filled rectangle for background
                cv2.rectangle(
                    annotated_frame, 
                    (x_min, y_min - text_h - margin), 
                    (x_min + text_w + margin, y_min), 
                    color, 
                    -1 # -1 fills the rectangle
                )
                
                # Draw text
                cv2.putText(
                    annotated_frame, 
                    label, 
                    (x_min + margin // 2, y_min - margin // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 0, 0), # Black text
                    bbox_thickness
                )

        return annotated_frame
    # -------------------------------

    def inference(
        self,
        weights: str = "yolo11n.pt",
        source: str = "test.mp4",
        view_img: bool = False,
        save_img: bool = False,
        exist_ok: bool = False,
        device: str = "",
        hide_conf: bool = False,
        slice_width: int = 512,
        slice_height: int = 512,
        bbox_thickness: int = 1,  # New argument
        hide_labels: bool = True, # New argument
    ) -> None:
        """Run object detection on a video using YOLO11 and SAHI.

        The function processes each frame of the video, applies sliced inference using SAHI, and optionally displays
        and/or saves the results with bounding boxes and labels.
        """
        # Video setup
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), "Error reading video file"

        # Output setup
        save_dir = increment_path("runs/detect/predict", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        self.load_model(weights, device)
        idx = 0  # Index for image frame writing
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Perform sliced prediction using SAHI
            results = get_sliced_prediction(
                frame[..., ::-1],  # Convert BGR to RGB
                self.detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
            )

            # --- CUSTOM VISUALIZATION: Draw boxes directly using OpenCV ---
            annotated_frame = self.draw_bboxes(
                frame, 
                results, 
                hide_conf, 
                bbox_thickness, 
                hide_labels
            )
            # --------------------------------------------------------------

            # Display results if requested
            if view_img:
                cv2.imshow("Ultralytics YOLO Inference", annotated_frame)

            # Save results if requested
            if save_img:
                idx += 1
                # Save the frame that was manually annotated
                cv2.imwrite(str(save_dir / f"img_{idx}.jpg"), annotated_frame) 

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def parse_opt() -> argparse.Namespace:
        """Parse command line arguments for the inference process.

        Returns:
            (argparse.Namespace): Parsed command line arguments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--weights", type=str, default="yolo11n.pt", help="initial weights path")
        parser.add_argument("--source", type=str, required=True, help="video file path")
        parser.add_argument("--view-img", action="store_true", help="show results")
        parser.add_argument("--save-img", action="store_true", help="save results")
        parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
        parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
        parser.add_argument("--hide-conf", default=False, action="store_true", help="display or hide confidences")
        parser.add_argument("--slice-width", default=512, type=int, help="Slice width for inference")
        parser.add_argument("--slice-height", default=512, type=int, help="Slice height for inference")
        
        # New arguments for manual drawing control
        parser.add_argument("--bbox-thickness", type=int, default=1, help="Bounding box line thickness")
        parser.add_argument("--hide-labels", action="store_true", help="Hide labels and scores from visualization")

        return parser.parse_args()


if __name__ == "__main__":
    inference = SAHIInference()
    inference.inference(**vars(inference.parse_opt()))