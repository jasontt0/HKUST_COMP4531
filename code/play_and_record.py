import time
import numpy as np
import sounddevice as sd
import threading
import matplotlib.pyplot as plt
import argparse
import soundfile as sf

recording = None
accu_frames = 0  # Initialize a counter for accumulated frames
start_idx = 0
mic1_channel = None
# Ignore Frames collected in first 500ms
ignore_frame = 44100 / 1000 * 500

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text
        

def play_and_record(channels, input_device, output_device, samplerate):

    def audio_callback(indata, frames, time):
        global accu_frames, recording, mic1_channel, ignore_frame

        if mic1_channel is None:
            threshold = 0.75
            max_amplitudes = np.max(np.abs(indata), axis=0)
            exceeding_signals = np.where(max_amplitudes > threshold)[0]
            if len(exceeding_signals) > 0:
                mic1_channel = exceeding_signals[0]
                print(f"Self-Checking Complete, Mic1 on {mic1_channel}")
            return
        else:
            if ignore_frame > 0:
                ignore_frame -= frames
                return
        
        # Determine the desired end index
        end_index = accu_frames + frames

        # Check if we need to expand the recording array
        if end_index > recording.shape[0]:
            # Calculate the new size of the recording array
            new_size = max(end_index, recording.shape[0] * 2)  # Double the size or at least reach end_index
            new_recording = np.zeros((new_size, channels))  # Create a new larger array
            new_recording[:recording.shape[0]] = recording  # Copy old data to the new array
            recording = new_recording  # Point `recording` to the new array

        # Accumulate data into the recording array
        shifted_indata = np.roll(indata, -mic1_channel, axis=1)
        recording[accu_frames:end_index] = shifted_indata[:frames]
        
        # Update the number of accumulated frames
        accu_frames += frames
    
    def output_callback(outdata, frames, time):
        global start_idx, mic1_channel
        if mic1_channel is None or ignore_frame > 0:
            return
        t = (start_idx + np.arange(frames)) / samplerate
        t = t.reshape(-1, 1)
        outdata[:] = args.amplitude * np.sin(2 * np.pi * args.frequency * t)
        start_idx += frames

    def callback(indata, outdata, frames, time, status):
        audio_callback(indata, frames, time)
        output_callback(outdata, frames, time)
        
    def finalize_recording():
        print("Recording Stop, generating plots")
        global recording, accu_frames
        # Resize recording to remove unused zeros
        if accu_frames < recording.shape[0]:
            recording = recording[:accu_frames]

    with sd.Stream(device=(args.input_device, args.output_device),
                   samplerate=args.samplerate,
                   channels=args.channels, callback=callback):
        print(f"Start Recording, press ENTER to stop (if self-checking is enabled, tap MIC1 to complete self-checking)")
        input()

    finalize_recording()
    
    return recording

# def save_plots(recording):
#     for channel in range(recording.shape[1]):
#         plt.figure()
#         plt.plot(recording[:, channel])
#         plt.title(f'Channel {channel + 1} Recording')
#         plt.xlabel('Samples')
#         plt.ylabel('Amplitude')
#         plt.grid()
#         plt.savefig(f'channel_{channel + 1}.png')
#         plt.close()
        
        
# def save_plot_and_audio(recording, output_filename='average_channels_1_to_6.wav', samplerate=44100):
#     # 确保录音数据至少有6个通道
#     if recording.shape[1] >= 6:
#         # 提取前六个通道的数据
#         first_six_channels = recording[:, :6]

#         # 计算每个时间点的平均值
#         avg_data = np.mean(first_six_channels, axis=1)

#         # 创建新的图形窗口
#         plt.figure()

#         # 绘制平均值的波形
#         plt.plot(avg_data)
#         plt.title('Average of Channels 1 to 6 Recording')
#         plt.xlabel('Samples')
#         plt.ylabel('Amplitude')
#         plt.grid()

#         # 保存图像
#         plt.savefig('average_channels_1_to_6.png')
#         plt.close()
        
#         # 保存为音频文件
#         sf.write(output_filename, avg_data, samplerate)
#         print(f"Audio saved to {output_filename}")

#     else:
#         print("Recording does not have 6 channels.")
        
        
def save_plot_and_audio(recording, output_filename='channel_1.wav', plot_filename='channel_1.png', samplerate=44100):
    # 提取第一个通道的数据
    first_channel_data = recording[:, 0]

    # 创建新的图形窗口
    plt.figure()

    # 绘制第一个通道的波形
    plt.plot(first_channel_data)
    plt.title('Channel 1 Recording')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid()

    # 保存图像
    plt.savefig(plot_filename)
    plt.close()

    # 确保音频数据在合理范围内（-1.0 到 1.0）
    first_channel_data = np.clip(first_channel_data, -1.0, 1.0)

    # 保存为音频文件
    sf.write(output_filename, first_channel_data, samplerate)
    print(f"Audio saved to {output_filename}")
    
    print(len(first_channel_data))
    

# def calculate_amplitude(first_channel_data, window_size=512, step_size=512):
#     """
#     计算每个窗口的幅值（max - min），并绘制幅值图。

#     参数:
#     - first_channel_data: 第一个通道的数据（numpy 数组）
#     - window_size: 窗口大小（默认: 1024）
#     - step_size: 每次滑动的步长（默认: 512）
    
#     返回:
#     - amplitudes: 每个窗口的幅值（max - min）数组
#     """
#     amplitudes = []  # 用于存储每个窗口的幅值
#     for start in range(0, len(first_channel_data) - window_size + 1, step_size):
#         # 取窗口的数据
#         window_data = first_channel_data[start:start + window_size]

#         # 计算该窗口的最大值和最小值
#         window_max = np.max(window_data)
#         window_min = np.min(window_data)

#         # 计算幅值
#         amplitude = window_max - window_min
#         amplitudes.append(amplitude)

#     # 转换为 numpy 数组
#     amplitudes = np.array(amplitudes)

#     # 绘制幅值图
#     plt.figure(figsize=(10, 6))
#     plt.plot(amplitudes)
#     plt.title('Amplitude')
#     plt.xlabel('Window Index')
#     plt.ylabel('Amplitude')
#     plt.grid()
#     plt.savefig('amplitude.png')
#     plt.close()

#     return amplitudes
        


def smooth_amplitude(amplitudes, window_size=10):
    """
    使用滑动平均法对幅值进行平滑。

    参数:
    - amplitudes: 原始幅值数据
    - window_size: 平滑窗口大小，默认值为 5

    返回:
    - smoothed_amplitudes: 平滑后的幅值数据
    """
    smoothed_amplitudes = np.convolve(amplitudes, np.ones(window_size)/window_size, mode='valid')
    return smoothed_amplitudes

def calculate_amplitude_and_smooth(first_channel_data, window_size=512, step_size=512):
    """
    计算每个窗口的幅值并进行平滑，绘制原始和平滑后的幅值图。

    参数:
    - first_channel_data: 第一个通道的数据（numpy 数组）
    - window_size: 窗口大小（默认: 1024）
    - step_size: 每次滑动的步长（默认: 512）
    - smooth_window_size: 平滑窗口大小（默认: 5）

    返回:
    - amplitudes: 原始幅值数据
    - smoothed_amplitudes: 平滑后的幅值数据
    """
    amplitudes = []  # 用于存储每个窗口的幅值
    for start in range(0, len(first_channel_data) - window_size + 1, step_size):
        # 取窗口的数据
        window_data = first_channel_data[start:start + window_size]

        # 计算该窗口的最大值和最小值
        window_max = np.max(window_data)
        window_min = np.min(window_data)

        # 计算幅值
        amplitude = window_max - window_min
        amplitudes.append(amplitude)

    # 转换为 numpy 数组
    amplitudes = np.array(amplitudes)

    # 对幅值进行平滑
    smoothed_amplitudes = smooth_amplitude(amplitudes, window_size=30)

    # 绘制原始幅值图
    plt.figure(figsize=(10, 6))
    plt.plot(amplitudes, label='Original Amplitude')
    plt.title('Original Amplitude per Window')
    plt.xlabel('Window Index')
    plt.ylabel('Original Amplitude')
    plt.grid()
    plt.savefig('OriginalAmplitude.png')

    # 绘制平滑后的幅值图
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_amplitudes, label='Smoothed Amplitude')
    plt.title('Smoothed Amplitude per Window')
    plt.xlabel('Window Index')
    plt.ylabel('Smoothed Amplitude')
    plt.grid()
    plt.savefig('SmoothedAmplitude.png')
    
    plt.close()

    return amplitudes, smoothed_amplitudes


def detect_pattern_and_output(amplitudes, threshold=0.5):
    """
    检测 amplitudes 中是否有 '先增大后减小' 的模式，并输出特定的数字。
    
    参数:
    - amplitudes: 幅值数组（例如经过平滑后的结果）
    - threshold: 用于判断增大和减小的阈值，默认值为 0.5
    
    返回:
    - 输出数字，符合模式时返回 10，否则返回 None
    """
    # 计算差分（导数）来检测增大和减小
    diff = np.diff(amplitudes)
    
    plt.figure(figsize=(10, 6))
    plt.plot(diff, label='Smoothed Amplitude Diff')
    plt.title('Smoothed Amplitude Diff per Window')
    plt.xlabel('Window Index')
    plt.ylabel('Smoothed Amplitude Diff')
    plt.grid()
    plt.savefig('DiffSmoothedAmplitude.png')
    
    plt.close()
    
    # 定义状态：先增大后减小
    increasing = False
    peak_detected = False
    for i in range(1, len(diff)):
        # 检查增大趋势（差分 > threshold）
        if not increasing and diff[i-1] < 0 and diff[i] > threshold:
            increasing = True
        
        # 检查是否出现峰值（先增大后减小）
        if increasing and diff[i] < -threshold:
            peak_detected = True
            break

    # 如果符合“先增大后减小”的模式，则输出 10
    if peak_detected:
         print(10)
    else:
        print("None")

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play a sine wave and record audio.')
    parser.add_argument('-c', '--channels', type=int, default=8, help='Number of recording channels (default: 8)')
    parser.add_argument('--list-devices', action='store_true', help='List audio devices and exit')
    parser.add_argument('--self-checking', action='store_true', help='Run the self-checking mechanism to reorder your channels')
    parser.add_argument('--input-device', type=int_or_str, help='Input device (numeric ID or substring)')
    parser.add_argument('--output-device', type=int_or_str, help='Output device (numeric ID or substring)')
    parser.add_argument('--samplerate', type=int, default=44100, help='Sample rate (default: 44100 Hz)')
    parser.add_argument('--frequency', nargs='?', metavar='FREQUENCY', type=float, default=500, help='frequency in Hz (default: %(default)s)')
    parser.add_argument('-a', '--amplitude', type=float, default=0.2, help='amplitude (default: %(default)s)')
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        exit()

    if not args.self_checking:
        mic1_channel = 0
    
    # global recording
    # Intialize a recording space for 10s (we will expand it if needed)
    recording = np.zeros((args.samplerate * 10, args.channels))

    # Play the audio file and record
    recording = play_and_record(args.channels, args.input_device, args.output_device, args.samplerate)

    # Save the recording as plots
    save_plot_and_audio(recording, output_filename='channel_1.wav', samplerate=44100)
    print("Plots saved for each channel.")
    
    _ , smoothed_amplitudes = calculate_amplitude_and_smooth(first_channel_data=recording[:, 0], window_size=512, step_size=512)
    
    detect_pattern_and_output(amplitudes = smoothed_amplitudes, threshold=0.5)
    
