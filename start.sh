#!/bin/bash

# Script to start the Flask app
APP_NAME="app.py"           # Flask 앱 메인 파일 이름
HOST="127.0.0.1"            # 호스트 주소
PORT="22"                 # 포트 번호
WORKERS=4                   # Gunicorn 워커 수 (프로덕션 환경에서 사용)

# Check if a production flag is provided
if [ "$1" == "production" ]; then
    echo "Starting Flask app in PRODUCTION mode using Gunicorn..."
    gunicorn -w $WORKERS -b $HOST:$PORT app:app
else
    echo "Starting Flask app in DEVELOPMENT mode using Flask's built-in server..."
    python3 $APP_NAME
fi
