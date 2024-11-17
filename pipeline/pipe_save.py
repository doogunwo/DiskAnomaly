import csv
from blktrace import run_blktrace
from fio import run_fio, select_fio
import time
import threading
import random


def pipe(device: str, output_csv: str, stop_event: threading.Event):
    
    data = []
    
    try:
        for line in run_blktrace(device):
            parts = line.split()
            if len(parts) >= 11:  # 필요한 필드가 충분한지 확인
                try:
                    timestamp = float(parts[3])  # 타임스탬프
                    io_type = parts[5]          # I/O 타입 (I/D/C 등)
                    sector = int(parts[7])      # 섹터 시작 번호
                    size = int(parts[9]) * 512  # 크기 (섹터 크기 기준으로 계산)
                    data.append([timestamp, io_type, sector, size])
                except (ValueError, IndexError) as e:
                    print(f"Skipping line due to error: {e}")  # 디버깅: 오류 메시지 출력
                    continue
                
    except KeyboardInterrupt:
        print("process interrrupted. Saving data")

    finally:
    # Save parsed data to CSV
        if data:
            with open(output_csv, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Timestamp", "IO_Type", "Sector", "Size"])
                writer.writerows(data)
            print(f"Blktrace data saved to {output_csv}")
        else:
            print("No valid data was captured.")

def process_blktrace(device: str, output_csv: str, stop_event: threading.Event):
   
    data = []
    try:
        for line in run_blktrace(device):
            parts = line.split()
            if len(parts) >= 11:  # 필요한 필드가 충분한지 확인
                try:
                    timestamp = float(parts[3])  # 타임스탬프
                    io_type = parts[5]          # I/O 타입 (I/D/C 등)
                    sector = int(parts[7])      # 섹터 시작 번호
                    size = int(parts[9]) * 512  # 크기 (섹터 크기 기준으로 계산)
                    data.append([timestamp, io_type, sector, size])
                except (ValueError, IndexError) as e:
                    print(f"Skipping line due to error: {e}")  # 디버깅: 오류 메시지 출력
                    continue
                
    except KeyboardInterrupt:
        print("process interrrupted. Saving data")

    finally:
    # Save parsed data to CSV
        if data:
            with open(output_csv, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Timestamp", "IO_Type", "Sector", "Size"])
                writer.writerows(data)
            print(f"Blktrace data saved to {output_csv}")
        else:
            print("No valid data was captured.")

def process_fio(folder: str,stop_event: threading.Event, pause_time: int = 10):
    jobfile = select_fio(folder)
    print(f"Selected Fio job : {jobfile}")
    run_fio(jobfile)
    time.sleep(pause_time)
    
if __name__ == "__main__":
    device = "/dev/sda6"
    output_csv = "blktrace_data.csv"
    fio_folder = "./fio"
    pause_time = random.randint(0, 9)
    # Event to signal the end of operations
    stop_event = threading.Event()

    # Start the blktrace thread
    blktrace_thread = threading.Thread(
        target=process_blktrace, args=(device, output_csv, stop_event)
    )
    blktrace_thread.start()

    # Start the Fio processing thread
    fio_thread = threading.Thread(
        target=process_fio, args=(fio_folder, stop_event, pause_time)
    )
    fio_thread.start()

    try:
        while True:
            fio_thread = threading.Thread(
                target=process_fio, args=(fio_folder, stop_event)
            )
            fio_thread.start()
            fio_thread.join()  # Ensure one job finishes before the next
    except KeyboardInterrupt:
            print("Interrupt received, stopping processes...")

    stop_event.set()
    blktrace_thread.join()
    fio_thread.join()

    print("end")
    