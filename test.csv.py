import librosa
import numpy as np
import pandas as pd
import os

def load_audio(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")
    
    try:
        y, sr = librosa.load(filepath, sr=None)
        print(f"✓ 読み込み成功: {filepath} (sr={sr}Hz)")
        return y, sr
    except Exception as e:
        raise Exception(f"読み込めませんでした: {filepath}")

audio_file = 'sound.wav'
y, sr = load_audio(audio_file)

# CQT計算
print("CQT分析中...")
CQT = librosa.cqt(y, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C2'))
CQT_db = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)

print(f"CQTの形状: {CQT_db.shape}")
print(f"音声の長さ: {len(y)/sr:.2f}秒")

# 時間軸とフリー軸の作成
times = librosa.frames_to_time(np.arange(CQT_db.shape[1]), sr=sr, hop_length=512)
freqs = librosa.cqt_frequencies(CQT_db.shape[0], fmin=librosa.note_to_hz('C2'))

# 間引きして保存（例：10フレームごと = 約0.1秒間隔）
downsample_rate = 10
downsampled_indices = np.arange(0, CQT_db.shape[1], downsample_rate)
cqt_downsampled = CQT_db[:, downsampled_indices]
times_downsampled = times[downsampled_indices]

print(f"間引き後の形状: {cqt_downsampled.shape}")
print(f"間引き後の時間範囲: {times_downsampled[0]:.3f}s - {times_downsampled[-1]:.3f}s")

# 1. 完全なCQTデータ（間引き版）を保存
cqt_full_df = pd.DataFrame(cqt_downsampled.T)
cqt_full_df.columns = [f'{freqs[i]:.1f}Hz' for i in range(len(freqs))]
cqt_full_df.index = times_downsampled
cqt_full_df.to_csv('cqt_data_full_downsampled.csv', encoding='utf-8-sig')

# 2. 各時刻での最大振幅周波数を抽出
max_freq_indices = np.argmax(cqt_downsampled, axis=0)
max_freqs = freqs[max_freq_indices]
max_amplitudes = np.max(cqt_downsampled, axis=0)

dominant_freq_df = pd.DataFrame({
    '時間(秒)': times_downsampled,
    '最大振幅周波数(Hz)': max_freqs,
    '音名': [librosa.hz_to_note(f) for f in max_freqs],
    '最大振幅(dB)': max_amplitudes
})
dominant_freq_df.to_csv('dominant_frequencies.csv', index=False, encoding='utf-8-sig')

# 3. 時系列での統計データ
stats_over_time = []
window_size = 20  # 20フレーム単位で統計

for i in range(0, cqt_downsampled.shape[1], window_size):
    end_idx = min(i + window_size, cqt_downsampled.shape[1])
    window_data = cqt_downsampled[:, i:end_idx]
    
    stats_over_time.append({
        '開始時間(秒)': times_downsampled[i],
        '終了時間(秒)': times_downsampled[end_idx-1],
        '平均振幅(dB)': np.mean(window_data),
        '最大振幅(dB)': np.max(window_data),
        '標準偏差(dB)': np.std(window_data)
    })

stats_df = pd.DataFrame(stats_over_time)
stats_df.to_csv('time_series_stats.csv', index=False, encoding='utf-8-sig')

# 4. 周波数帯域ごとの時系列データ
freq_bands = {
    '低域(65-130Hz)': (0, 12),
    '中低域(130-260Hz)': (12, 24), 
    '中域(260-520Hz)': (24, 36),
    '中高域(520-1040Hz)': (36, 48),
    '高域(1040Hz以上)': (48, len(freqs))
}

band_data = {}
for band_name, (start_idx, end_idx) in freq_bands.items():
    band_amplitudes = np.mean(cqt_downsampled[start_idx:end_idx, :], axis=0)
    band_data[band_name] = band_amplitudes

band_df = pd.DataFrame(band_data)
band_df.index = times_downsampled
band_df.to_csv('frequency_bands_over_time.csv', encoding='utf-8-sig')

# 5. データ概要
summary = {
    '項目': [
        '総フレーム数', '間引き後フレーム数', '間引き率', 
        '音声長(秒)', '周波数bins数', '最低周波数(Hz)', 
        '最高周波数(Hz)', 'データ範囲最小(dB)', 'データ範囲最大(dB)'
    ],
    '値': [
        CQT_db.shape[1], cqt_downsampled.shape[1], downsample_rate,
        len(y)/sr, len(freqs), freqs[0], freqs[-1],
        np.min(CQT_db), np.max(CQT_db)
    ]
}

summary_df = pd.DataFrame(summary)
summary_df.to_csv('data_summary.csv', index=False, encoding='utf-8-sig')

# 6. 最後の10秒分だけ詳細データ
last_seconds = 10
last_frames = int(last_seconds * sr / 512)
if CQT_db.shape[1] >= last_frames:
    last_data = CQT_db[:, -last_frames:]
    last_times = times[-last_frames:]
    
    last_df = pd.DataFrame(last_data.T)
    last_df.columns = [f'{freqs[i]:.1f}Hz' for i in range(len(freqs))]
    last_df.index = last_times
    last_df.to_csv('cqt_last_10_seconds.csv', encoding='utf-8-sig')

print("\n=== 出力されたCSVファイル ===")
print("1. cqt_data_full_downsampled.csv - 間引き版全データ")
print("2. dominant_frequencies.csv - 各時刻の最大振幅周波数")
print("3. time_series_stats.csv - 時系列統計")
print("4. frequency_bands_over_time.csv - 周波数帯域別時系列")
print("5. data_summary.csv - データ概要")
print("6. cqt_last_10_seconds.csv - 最後の10秒詳細")

print(f"\n=== 処理結果 ===")
print(f"元データ: {CQT_db.shape}")
print(f"間引き後: {cqt_downsampled.shape}")
print(f"時間範囲: 0秒 - {times[-1]:.2f}秒")
print(f"間引き率: {downsample_rate}フレームごと（約{downsample_rate*512/sr:.3f}秒間隔）")