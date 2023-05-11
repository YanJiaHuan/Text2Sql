import time

try:
    while True:
        print("Running...")
        time.sleep(1)  # Simulate some processing with a delay
except KeyboardInterrupt:
    print("KeyboardInterrupt detected. Exiting the program...")
