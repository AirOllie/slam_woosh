import requests
import json
from pynput import keyboard
import time
import threading

class RobotController:
    def __init__(self, base_url="http://192.168.1.87:5480", control_frequency=10):
        self.base_url = base_url
        self.headers = {'Content-Type': 'application/json'}
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.speed_increment = 0.1
        self.max_speed = 0.5
        self.pressed_keys = set()
        
        # Control loop parameters
        self.control_frequency = control_frequency  # Hz
        self.control_period = 1.0 / control_frequency
        self.is_running = False
        self.control_thread = None

    def initialize_robot(self):
        url = f"{self.base_url}/woosh/robot/InitRobot"
        payload = json.dumps({"isRecord": True})
        try:
            response = requests.post(url, headers=self.headers, data=payload)
            print("Robot initialized" if response.ok else "Initialization failed")
            return response.ok
        except Exception as e:
            print(f"Error initializing robot: {e}")
            return False

    def send_movement_command(self):
        url = f"{self.base_url}/woosh/robot/Twist"
        payload = json.dumps({
            "linear": self.linear_speed,
            "angular": self.angular_speed
        })
        try:
            print(f"Linear speed: {self.linear_speed}, Angular speed: {self.angular_speed}")
            response = requests.post(url, headers=self.headers, data=payload)
            return response.ok
        except Exception as e:
            print(f"Error sending movement command: {e}")
            return False

    def control_loop(self):
        """Main control loop that runs in a separate thread"""
        last_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            dt = current_time - last_time
            
            if dt >= self.control_period:
                self.send_movement_command()
                last_time = current_time
            else:
                # Sleep for a small amount of time to prevent CPU hogging
                time.sleep(max(0, self.control_period - dt) / 2)

    def start_control_thread(self):
        """Start the control thread"""
        self.is_running = True
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True  # Thread will close when main program exits
        self.control_thread.start()

    def stop_control_thread(self):
        """Stop the control thread"""
        self.is_running = False
        if self.control_thread:
            self.control_thread.join()

    def stop(self):
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.send_movement_command()

    def on_press(self, key):
        try:
            key_char = key.char
        except AttributeError:
            # Special keys
            if key == keyboard.Key.space:
                self.stop()
                return
            elif key == keyboard.Key.esc:
                self.stop()
                self.stop_control_thread()
                # Stop listener
                return False
            return

        if key_char not in self.pressed_keys:
            self.pressed_keys.add(key_char)
            
        if key_char == 'w':  # Forward
            self.linear_speed = min(self.linear_speed + self.speed_increment, self.max_speed)
        elif key_char == 's':  # Backward
            self.linear_speed = max(self.linear_speed - self.speed_increment, -self.max_speed)
        elif key_char == 'a':  # Turn left
            self.angular_speed = min(self.angular_speed + self.speed_increment, self.max_speed*2)
        elif key_char == 'd':  # Turn right
            self.angular_speed = max(self.angular_speed - self.speed_increment, -self.max_speed*2)

    def on_release(self, key):
        try:
            key_char = key.char
        except AttributeError:
            return

        if key_char in self.pressed_keys:
            self.pressed_keys.remove(key_char)

        if key_char in ['w', 's']:  # Stop linear movement
            self.linear_speed = 0.0
        elif key_char in ['a', 'd']:  # Stop angular movement
            self.angular_speed = 0.0

        print(f"Pressed keys: {self.pressed_keys}")

def main():
    robot = RobotController(control_frequency=10)  # 10 Hz control frequency
    
    if not robot.initialize_robot():
        print("Failed to initialize robot. Exiting...")
        return

    # Start the control thread
    robot.start_control_thread()

    print("""
Robot Keyboard Controller
------------------------
Controls:
W - Move forward
S - Move backward
A - Turn left
D - Turn right
Space - Emergency stop
ESC - Exit
------------------------
Running control loop at 10 Hz
""")

    # Set up keyboard listener
    with keyboard.Listener(
        on_press=robot.on_press,
        on_release=robot.on_release) as listener:
        listener.join()

    print("\nController stopped.")

if __name__ == "__main__":
    main()