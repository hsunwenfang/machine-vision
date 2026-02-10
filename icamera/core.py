import cv2
import time

class IPhoneCamera:
    def __init__(self, camera_index=None):
        """
        Initialize the camera.
        
        Args:
            camera_index (int, optional): The specific camera index to open. 
                                          If None, it attempts to find a working camera, 
                                          preferring external ones (often the iPhone).
        """
        self.cap = None
        self.camera_index = camera_index
        
        if self.camera_index is not None:
            self.open(self.camera_index)
        else:
            # Try to find the iPhone automatically or default to 0
            self._find_and_open_camera()

    def _find_and_open_camera(self):
        """
        Attempts to find available cameras. 
        On macOS with Continuity Camera, the iPhone often appears at index 1 or 2 
        if the built-in webcam is 0.
        """
        # Check a range of indices. 
        # Usually 0 is built-in, 1 is often iPhone or external.
        print("Searching for cameras...")
        found = False
        
        # We'll try indices 0 through 3. 
        # If the user specifically wants the iPhone, they might need to check the output.
        for idx in range(4):
            temp_cap = cv2.VideoCapture(idx)
            if temp_cap.isOpened():
                ret, _ = temp_cap.read()
                if ret:
                    print(f"Found working camera at index {idx}")
                    # If we haven't selected one yet, or if this is index 1 (likely iPhone), pick it.
                    # This is a heuristic: prefer index 1 over 0 if both exist, as 0 is usually the low-res built-in webcam.
                    if not found:
                        self.camera_index = idx
                        found = True
                    elif idx == 1: 
                        # If we found 0 previously but now found 1, switch to 1 (likely iPhone)
                        self.camera_index = idx
                temp_cap.release()
        
        if found:
            print(f"Selecting camera index {self.camera_index}")
            self.open(self.camera_index)
        else:
            raise RuntimeError("No working cameras found.")

    def open(self, index):
        """Opens the video capture device."""
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {index}")
        
        # Optional: Set high resolution (iPhone cameras support high res)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def get_frame(self):
        """
        Reads a frame from the camera.
        
        Returns:
            frame (numpy.ndarray): The image frame.
            ret (bool): True if frame is read correctly, False otherwise.
        """
        if self.cap is None or not self.cap.isOpened():
            return None, False
        return self.cap.read()

    def stream(self):
        """
        Simple generator to yield frames.
        """
        while True:
            ret, frame = self.get_frame()
            if not ret:
                break
            yield frame

    def release(self):
        """Releases the camera resource."""
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

def list_cameras(max_indices=5):
    """
    Helper function to list available camera indices and their resolutions.
    Useful for debugging which index corresponds to the iPhone.
    """
    available = []
    for i in range(max_indices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"Index {i}: Open (Resolution: {w}x{h})")
                available.append(i)
            else:
                print(f"Index {i}: Open but cannot read")
            cap.release()
        else:
            print(f"Index {i}: Failed to open")
    return available

if __name__ == "__main__":
    # Simple demo
    print("Starting iPhone Camera Demo...")
    
    # You can pass a specific index if you know it, e.g., IPhoneCamera(1)
    try:
        with IPhoneCamera() as cam:
            print("Press 'q' to quit.")
            for frame in cam.stream():
                cv2.imshow('iPhone Camera Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")
