import ffmpeg

# Tên file đầu vào và đầu ra
input_file = "myc.mp4"
output_file = "mycaudio.flac"

# Gọi ffmpeg để trích âm thanh
ffmpeg.input(input_file).output(output_file, acodec='flac', vn=None).run()

print("Đã xuất âm thanh dưới dạng FLAC.")
