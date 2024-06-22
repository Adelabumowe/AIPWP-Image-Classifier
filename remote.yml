import os

def execute_command(command):
    # Vulnerable: directly executing user input
    os.system(command)

if __name__ == "__main__":
    user_input = input("Enter a command to execute: ")
    execute_command(user_input)
