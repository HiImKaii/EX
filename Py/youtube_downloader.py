import os
import re
import sys
from pytube import YouTube, Playlist
from pytube.exceptions import PytubeError, RegexMatchError

def validate_url(url):
    """Kiểm tra tính hợp lệ của URL YouTube"""
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/.+'
    
    if re.match(youtube_regex, url):
        return True
    return False

def get_video_info(url):
    """Lấy thông tin của video"""
    try:
        yt = YouTube(url)
        
        info = {
            "title": yt.title,
            "author": yt.author,
            "length": yt.length,  # Độ dài tính bằng giây
            "views": yt.views,
            "thumbnail_url": yt.thumbnail_url,
            "description": yt.description[:200] + "..." if len(yt.description) > 200 else yt.description,
        }
        
        return info, yt
    
    except PytubeError as e:
        print(f"Lỗi khi lấy thông tin video: {e}")
        return None, None

def format_duration(seconds):
    """Chuyển đổi thời gian từ giây sang định dạng HH:MM:SS"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"

def format_filesize(bytes):
    """Chuyển đổi kích thước từ byte sang MB/GB"""
    mb = bytes / (1024 * 1024)
    
    if mb > 1024:
        return f"{mb/1024:.2f} GB"
    else:
        return f"{mb:.2f} MB"

def get_streams_info(yt):
    """Lấy thông tin về các luồng có sẵn"""
    if not yt:
        return None
    
    # Lấy tất cả các luồng video
    streams = yt.streams.filter(progressive=True).order_by('resolution').desc()
    
    result = []
    for i, stream in enumerate(streams):
        result.append({
            "index": i + 1,
            "resolution": stream.resolution,
            "mime_type": stream.mime_type,
            "fps": stream.fps,
            "filesize": stream.filesize,
            "stream": stream
        })
    
    # Thêm các luồng âm thanh
    audio_streams = yt.streams.filter(only_audio=True).order_by('abr').desc()
    
    for i, stream in enumerate(audio_streams):
        result.append({
            "index": len(result) + 1,
            "resolution": "Audio only",
            "mime_type": stream.mime_type,
            "abr": stream.abr,
            "filesize": stream.filesize,
            "stream": stream
        })
    
    return result

def download_video(stream, output_path=None, filename=None):
    """Tải xuống video từ luồng đã chọn"""
    try:
        # Nếu không chỉ định đường dẫn đầu ra, sử dụng thư mục hiện tại
        if not output_path:
            output_path = os.getcwd()
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(output_path, exist_ok=True)
        
        # Thực hiện tải xuống
        print("\nĐang tải xuống...")
        
        # Nếu chỉ định tên file, sử dụng tên đó
        if filename:
            file_path = stream.download(output_path=output_path, filename=filename)
        else:
            file_path = stream.download(output_path=output_path)
        
        return file_path
    
    except Exception as e:
        print(f"Lỗi khi tải xuống: {e}")
        return None

def download_playlist(playlist_url, output_path=None, resolution="highest"):
    """Tải xuống tất cả video trong danh sách phát"""
    try:
        playlist = Playlist(playlist_url)
        
        # Nếu không chỉ định đường dẫn đầu ra, sử dụng thư mục hiện tại
        if not output_path:
            output_path = os.getcwd()
        
        # Tạo thư mục với tên của danh sách phát
        playlist_dir = os.path.join(output_path, playlist.title)
        os.makedirs(playlist_dir, exist_ok=True)
        
        print(f"Danh sách phát: {playlist.title}")
        print(f"Số lượng video: {len(playlist.video_urls)}")
        
        successful = 0
        failed = 0
        
        for i, url in enumerate(playlist.video_urls):
            try:
                yt = YouTube(url)
                print(f"\n[{i+1}/{len(playlist.video_urls)}] Đang tải: {yt.title}")
                
                # Chọn luồng phù hợp dựa trên độ phân giải được yêu cầu
                if resolution == "highest":
                    stream = yt.streams.get_highest_resolution()
                elif resolution == "lowest":
                    stream = yt.streams.get_lowest_resolution()
                elif resolution == "audio":
                    stream = yt.streams.get_audio_only()
                else:
                    # Tìm luồng với độ phân giải cụ thể
                    stream = yt.streams.filter(res=resolution, progressive=True).first()
                    if not stream:
                        print(f"Không tìm thấy độ phân giải {resolution}, sử dụng độ phân giải cao nhất")
                        stream = yt.streams.get_highest_resolution()
                
                # Tải xuống video
                file_path = download_video(stream, playlist_dir)
                
                if file_path:
                    print(f"Đã tải xuống: {os.path.basename(file_path)}")
                    successful += 1
                else:
                    print(f"Không thể tải xuống video {i+1}")
                    failed += 1
            
            except Exception as e:
                print(f"Lỗi khi tải video {i+1}: {e}")
                failed += 1
        
        print(f"\nĐã tải xuống {successful} video, thất bại {failed} video")
        print(f"Thư mục đầu ra: {playlist_dir}")
        
        return playlist_dir
    
    except Exception as e:
        print(f"Lỗi khi tải danh sách phát: {e}")
        return None

def main():
    print("===== YOUTUBE DOWNLOADER =====")
    print("1. Tải xuống một video")
    print("2. Tải xuống danh sách phát")
    
    choice = input("\nNhập lựa chọn của bạn: ")
    
    if choice == "1":
        # Tải xuống một video
        url = input("\nNhập URL của video YouTube: ")
        
        if not validate_url(url):
            print("URL không hợp lệ!")
            return
        
        print("\nĐang lấy thông tin video...")
        info, yt = get_video_info(url)
        
        if not info:
            print("Không thể lấy thông tin video!")
            return
        
        print(f"\nTiêu đề: {info['title']}")
        print(f"Tác giả: {info['author']}")
        print(f"Thời lượng: {format_duration(info['length'])}")
        print(f"Lượt xem: {info['views']:,}")
        
        # Lấy thông tin về các luồng có sẵn
        streams_info = get_streams_info(yt)
        
        if not streams_info:
            print("Không tìm thấy luồng nào có sẵn!")
            return
        
        print("\nCác luồng có sẵn:")
        for stream in streams_info:
            if "resolution" in stream and stream["resolution"] != "Audio only":
                print(f"{stream['index']}. {stream['resolution']} ({stream['mime_type']}, {stream['fps']}fps) - {format_filesize(stream['filesize'])}")
            elif "resolution" in stream and stream["resolution"] == "Audio only":
                print(f"{stream['index']}. Audio only ({stream['mime_type']}, {stream['abr']}) - {format_filesize(stream['filesize'])}")
        
        # Chọn luồng để tải xuống
        stream_choice = input("\nNhập số thứ tự của luồng muốn tải xuống (mặc định: 1): ")
        
        try:
            stream_idx = int(stream_choice) - 1 if stream_choice else 0
            
            if 0 <= stream_idx < len(streams_info):
                selected_stream = streams_info[stream_idx]
                
                # Chọn thư mục đầu ra
                output_path = input("\nNhập đường dẫn đầu ra (để trống nếu muốn sử dụng thư mục hiện tại): ")
                
                # Tải xuống video
                file_path = download_video(selected_stream["stream"], output_path)
                
                if file_path:
                    print(f"\nĐã tải xuống thành công: {os.path.basename(file_path)}")
                    print(f"Đường dẫn: {file_path}")
                else:
                    print("\nTải xuống thất bại!")
            else:
                print("Lựa chọn không hợp lệ!")
        
        except ValueError:
            print("Số thứ tự không hợp lệ!")
    
    elif choice == "2":
        # Tải xuống danh sách phát
        url = input("\nNhập URL của danh sách phát YouTube: ")
        
        if not validate_url(url):
            print("URL không hợp lệ!")
            return
        
        print("\nĐang lấy thông tin danh sách phát...")
        
        output_path = input("\nNhập đường dẫn đầu ra (để trống nếu muốn sử dụng thư mục hiện tại): ")
        
        print("\nChọn độ phân giải:")
        print("1. Cao nhất")
        print("2. Thấp nhất")
        print("3. Chỉ âm thanh")
        print("4. 720p")
        print("5. 480p")
        print("6. 360p")
        
        res_choice = input("\nNhập lựa chọn của bạn (mặc định: 1): ")
        
        resolution_map = {
            "1": "highest",
            "2": "lowest",
            "3": "audio",
            "4": "720p",
            "5": "480p",
            "6": "360p"
        }
        
        resolution = resolution_map.get(res_choice, "highest")
        
        # Tải xuống danh sách phát
        output_dir = download_playlist(url, output_path, resolution)
        
        if output_dir:
            print("\nĐã tải xuống danh sách phát thành công!")
        else:
            print("\nTải xuống danh sách phát thất bại!")
    
    else:
        print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã hủy tải xuống!")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi: {e}")
    
    print("\nCảm ơn bạn đã sử dụng YouTube Downloader!") 