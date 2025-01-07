#!/bin/bash

# Detect OS type
detect_os() {
  OS="$(uname -s)"
  case "$OS" in
    Linux*)     OS_TYPE="Linux";;
    Darwin*)    OS_TYPE="Mac";;
    CYGWIN*|MINGW*|MSYS*) OS_TYPE="Windows";;
    *)          OS_TYPE="Unknown";;
  esac
}

# Adjust file permissions
adjust_permissions() {
  if [ "$OS_TYPE" == "Linux" ] || [ "$OS_TYPE" == "Mac" ]; then
    echo "Adjusting permissions for $OS_TYPE..."
    chmod -R 755 "$APP_PATH"
    chmod 644 "$APP_PATH"/App/www/*.png
  elif [ "$OS_TYPE" == "Windows" ]; then
    echo "Adjusting permissions for Windows..."
    icacls "$APP_PATH" /grant Everyone:\(RX,W\)
    icacls "$APP_PATH"/App/www/*.png /grant Everyone:\(RX\)
  else
    echo "Unsupported OS type: $OS_TYPE"
    exit 1
  fi
}

# Install python3-venv package for Linux/Mac if not installed
install_python3_venv() {
  if [ "$OS_TYPE" == "Linux" ] || [ "$OS_TYPE" == "Mac" ]; then
    if ! dpkg -l | grep -q python3.10-venv; then
      echo "The python3-venv package is not installed. Installing it now..."
      sudo apt update
      sudo apt install -y python3.10-venv || {
        echo "Failed to install python3-venv. Please install it manually."
        exit 1
      }
    fi
  fi
}

# Main setup function
setup_app() {
  # Define app path
  APP_PATH=$(pwd)

  if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found. Run this script from the project root."
    exit 1
  fi

  # Install python3-venv on Linux/Mac
  install_python3_venv

  echo "Installing dependencies..."
  if [ "$OS_TYPE" == "Linux" ] || [ "$OS_TYPE" == "Mac" ]; then
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
  elif [ "$OS_TYPE" == "Windows" ]; then
    python -m venv env
    source env/Scripts/activate
    pip install -r requirements.txt
  else
    echo "Unsupported OS for dependency installation."
    exit 1
  fi

  echo "Setting up permissions..."
  adjust_permissions

  # Start the app
  echo "Starting the Shiny app..."
  if [ "$OS_TYPE" == "Linux" ] || [ "$OS_TYPE" == "Mac" ]; then
    shiny run app.py
  elif [ "$OS_TYPE" == "Windows" ]; then
    shiny.exe run app.py
  else
    echo "Unsupported OS for running the app."
    exit 1
  fi

  echo "Setup complete!"
}

# Detect OS and run setup
detect_os
echo "Detected OS: $OS_TYPE"
setup_app
