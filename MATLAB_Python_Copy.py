import time
from selenium import webdriver
import subprocess
import json
import threading as thr
from os import path
import sys

# Kill the server if a dash app is already running
def kill_server():
    subprocess.run("lsof -t -i tcp:8080 | xargs kill -9", shell=True)

# Start Dash app.
def start_dash_app_frozen():
    path_dir = str(path.dirname(sys.executable))
    subprocess.Popen(path_dir+"/dash_app", shell=False)

# Start the driver
def start_driver():
    driver = webdriver.Chrome()
    time.sleep(5) # give dash app time to start running
    driver.get("http://0.0.0.0:8080/") # go to the local server
    save_browser_session(driver) # save the browser identity info for giving future instructions to the browser (for instance opening up a new browser tab).

# Function to save browser session info
def save_browser_session(input_driver):
    driver = input_driver
    executor_url = driver.command_executor._url
    session_id = driver.session_id
    browser_file = path_dir+"/browsersession.txt"
    with open(browser_file, "w") as f:
        f.write(executor_url)
        f.write("\n")
        f.write(session_id)
    print("DRIVER SAVED IN TEXT FILE browsersession.txt")

# Infinite while loop to keep server running
def keep_server_running():
    while True:
        time.sleep(60)
        print("Next run for 60 seconds")

# Putting everything together in a function
def main():
    kill_server() # kill open server on port
    thread = thr.Thread(target=start_dash_app_frozen)
    thread.start() # start dash app on port
    start_driver() # start selenium controled chrome browser and go to port
    keep_server_running() # keep the main file running with a loop

if __name__ == '__main__':
    main()