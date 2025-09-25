
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# MoviePyのインポート（バージョン互換性対応）
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
except ImportError:
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
    except ImportError:
        print("MoviePyがインストールされていません。")
        print("pip install moviepy を実行してください。")
        import sys
        sys.exit(1)
import os
import argparse

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'

class CQTVisualizer:
    def __init__(self, input_file, output_file='output.mp4'):
        """
        CQTビジュアライザー初期化
        
        Parameters:
        -----------
        input_file : str
            入力ファイル（音声または動画）
        output_file : str
            出力ファイル名
        """
        self.input_file = input_file
        self.output_file = output_file
        self.is_video_input = False
        self.base_video = None
        
        # 表示フラグ（デフォルト値）
        self.show_spectrogram = True
        self.show_bars = True
        self.show_grid = True
        self.show_time_marker = True
        
        # オーバーレイ設定
        self.overlay_opacity = 0.7  # 0.0（完全透明）～1.0（完全不透明）
        self.background_opacity = 0.3  # 背景の不透明度
        
        # 表示設定
        self.target_HOP = 512
        self.target_fps = 25
        self.time_window = 10.0  # 表示する時間幅（秒）
        self.freq_range = (65, 2000)  # 表示周波数範囲
        
        # カラー設定
        self.colormap = 'plasma'
        self.bar_color = '#00ff00'
        self.grid_color = 'white'
        self.marker_color = '#ff0000'
        
    def detect_input_type(self):
        """入力ファイルタイプを判定"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg']
        
        ext = os.path.splitext(self.input_file)[1].lower()
        
        if ext in video_extensions:
            self.is_video_input = True
            print(f"✓ 動画ファイルを検出: {self.input_file}")
            return 'video'
        elif ext in audio_extensions:
            self.is_video_input = False
            print(f"✓ 音声ファイルを検出: {self.input_file}")
            return 'audio'
        else:
            # 拡張子で判定できない場合は、moviepyで読み込み試行
            try:
                clip = VideoFileClip(self.input_file)
                if clip.audio is not None:
                    self.is_video_input = True
                    clip.close()
                    print(f"✓ 動画ファイルとして検出: {self.input_file}")
                    return 'video'
            except:
                pass
            
            # 音声として試行
            try:
                y, sr = librosa.load(self.input_file, sr=None, duration=0.1)
                self.is_video_input = False
                print(f"✓ 音声ファイルとして検出: {self.input_file}")
                return 'audio'
            except:
                raise ValueError(f"ファイルタイプを判定できません: {self.input_file}")
    
    def load_audio(self):
        """音声データ読み込み"""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"ファイルが見つかりません: {self.input_file}")
        
        if self.is_video_input:
            # 動画から音声抽出
            print("動画から音声を抽出中...")
            video = VideoFileClip(self.input_file)
            self.base_video = video  # 元動画を保持
            
            if video.audio is None:
                raise ValueError("動画に音声トラックがありません")
            
            # 一時音声ファイルに保存
            temp_audio = 'temp_audio.wav'
            video.audio.write_audiofile(temp_audio, logger=None)
            
            # librosaで読み込み
            y, sr = librosa.load(temp_audio, sr=None)
            
            # 一時ファイル削除
            os.remove(temp_audio)
            
            print(f"✓ 音声抽出成功 (sr={sr}Hz)")
            return y, sr
        else:
            # 音声ファイル直接読み込み
            y, sr = librosa.load(self.input_file, sr=None)
            print(f"✓ 音声読み込み成功 (sr={sr}Hz)")
            return y, sr
    
    def set_display_flags(self, spectrogram=True, bars=True, grid=True, time_marker=True):
        """
        表示要素のフラグを設定
        
        Parameters:
        -----------
        spectrogram : bool
            スペクトログラムを表示
        bars : bool
            周波数バー（棒グラフ）を表示
        grid : bool
            周波数グリッドを表示
        time_marker : bool
            現在時刻マーカーを表示
        """
        self.show_spectrogram = spectrogram
        self.show_bars = bars
        self.show_grid = grid
        self.show_time_marker = time_marker
        
        print(f"表示設定: スペクトログラム={spectrogram}, バー={bars}, "
              f"グリッド={grid}, 時刻マーカー={time_marker}")
    
    def set_overlay_opacity(self, overlay=0.7, background=0.3):
        """
        オーバーレイの不透明度を設定
        
        Parameters:
        -----------
        overlay : float
            オーバーレイ全体の不透明度 (0.0-1.0)
        background : float
            背景の不透明度 (0.0-1.0)
        """
        self.overlay_opacity = np.clip(overlay, 0.0, 1.0)
        self.background_opacity = np.clip(background, 0.0, 1.0)
        print(f"不透明度設定: オーバーレイ={self.overlay_opacity:.1f}, "
              f"背景={self.background_opacity:.1f}")
    
    def create_animation(self, y, sr):
        """アニメーション作成"""
        # CQT計算
        print("CQT分析中...")
        CQT = librosa.cqt(y, sr=sr, hop_length=self.target_HOP, 
                         fmin=librosa.note_to_hz('C2'))
        CQT_db = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)
        
        # 基本情報
        audio_duration = len(y) / sr
        total_frames = CQT_db.shape[1]
        frame_duration = self.target_HOP / sr
        
        print(f"音声の長さ: {audio_duration:.2f}秒")
        print(f"CQTフレーム数: {total_frames}")
        print(f"フレーム間隔: {frame_duration:.4f}秒")
        
        # 動画フレーム設定
        times = np.arange(0, audio_duration, 1/self.target_fps)
        frames_per_sec = sr / self.target_HOP
        actual_frames = np.round(times * frames_per_sec).astype(int)
        actual_frames = np.clip(actual_frames, 0, total_frames - 1)
        
        # 周波数軸
        freqs = librosa.cqt_frequencies(CQT_db.shape[0], 
                                       fmin=librosa.note_to_hz('C2'))
        
        # 図の設定（透明背景対応）
        if self.is_video_input:
            # 動画オーバーレイ用：透明背景
            fig, ax = plt.subplots(figsize=(16, 9))
            fig.patch.set_alpha(0.0)  # 図全体を透明に
            ax.patch.set_alpha(0.0)   # 軸領域も透明に
        else:
            # 音声のみ：通常の黒背景
            fig, ax = plt.subplots(figsize=(16, 9))
            fig.patch.set_facecolor('#0a0a0a')
            ax.set_facecolor('#0a0a0a')
        
        def animate(frame_idx):
            """アニメーション関数"""
            if frame_idx >= len(actual_frames):
                return []
            
            current_frame = actual_frames[frame_idx]
            current_time = librosa.frames_to_time(current_frame, sr=sr, 
                                                 hop_length=self.target_HOP)
            
            # 画面クリア
            ax.clear()
            
            # 動画オーバーレイ時の背景設定
            if self.is_video_input:
                ax.patch.set_alpha(0.0)
                # 半透明の黒背景を追加（オプション）
                if self.background_opacity > 0:
                    ax.add_patch(plt.Rectangle((0, 0), 1, 1, 
                                              transform=ax.transAxes,
                                              facecolor='black',
                                              alpha=self.background_opacity,
                                              zorder=-1))
            else:
                ax.set_facecolor('#0a0a0a')
            
            # 時間ウィンドウの計算
            time_frames = int(self.time_window * sr / self.target_HOP)
            start_frame = max(0, current_frame - time_frames // 2)
            end_frame = min(total_frames, start_frame + time_frames)
            
            if end_frame > start_frame:
                display_data = CQT_db[:, start_frame:end_frame]
                times_display = librosa.frames_to_time(np.arange(start_frame, end_frame), 
                                                       sr=sr, hop_length=self.target_HOP)
                
                # スペクトログラム表示
                if self.show_spectrogram:
                    X, Y = np.meshgrid(freqs, times_display)
                    mesh = ax.pcolormesh(X, Y, display_data.T, 
                                        cmap=self.colormap, 
                                        vmin=-60, vmax=0,
                                        alpha=self.overlay_opacity if self.is_video_input else 1.0)
                
                # 軸の設定
                ax.set_xlim(self.freq_range)
                ax.set_ylim([times_display[0], times_display[-1]])
                ax.set_xscale('log')
                
                # 周波数バー表示
                if self.show_bars and current_frame < CQT_db.shape[1]:
                    spectrum = CQT_db[:, current_frame]
                    spectrum_normalized = (spectrum + 80) / 85
                    spectrum_normalized = np.clip(spectrum_normalized, 0, 1)
                    
                    bottom_edge = times_display[0] + (times_display[-1] - times_display[0]) * 0.02
                    spectrum_height = spectrum_normalized * (times_display[-1] - times_display[0]) * 0.15
                    
                    for freq, amp, height in zip(freqs, spectrum, spectrum_height):
                        if self.freq_range[0] <= freq <= self.freq_range[1] and amp > -70:
                            ax.bar(freq, height, width=freq*0.1, 
                                  bottom=bottom_edge, 
                                  color=self.bar_color, 
                                  alpha=0.8 * self.overlay_opacity if self.is_video_input else 0.8,
                                  edgecolor='white', linewidth=0.5)
                
                # 周波数グリッド表示
                if self.show_grid:
                    for octave in range(2, 6):
                        freq = librosa.note_to_hz(f'C{octave}')
                        if self.freq_range[0] <= freq <= self.freq_range[1]:
                            ax.axvline(x=freq, color=self.grid_color, 
                                      alpha=0.4 * self.overlay_opacity if self.is_video_input else 0.4,
                                      linewidth=1)
                            ax.text(freq * 1.05, times_display[0] + 0.5, f'C{octave}', 
                                   color='white', fontsize=11, ha='left', weight='bold',
                                   alpha=self.overlay_opacity if self.is_video_input else 1.0,
                                   bbox=dict(boxstyle="round,pad=0.2", 
                                           facecolor='black', 
                                           alpha=0.8 * self.overlay_opacity if self.is_video_input else 0.8))
                
                # 現在時刻マーカー表示
                if self.show_time_marker:
                    ax.axhline(y=current_time, color=self.marker_color, 
                              linewidth=4, 
                              alpha=0.9 * self.overlay_opacity if self.is_video_input else 0.9)
                    
                    ax.text(self.freq_range[0] * 1.02, 
                           current_time + (times_display[-1] - times_display[0]) * 0.02, 
                           f'{current_time:.1f}s', 
                           color=self.marker_color, fontsize=14, weight='bold', va='bottom',
                           alpha=self.overlay_opacity if self.is_video_input else 1.0,
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='white', 
                                   alpha=0.9 * self.overlay_opacity if self.is_video_input else 0.9))
                
                # ラベル設定
                text_alpha = self.overlay_opacity if self.is_video_input else 1.0
                ax.set_xlabel('周波数 (Hz)', color='white', fontsize=14, alpha=text_alpha)
                ax.set_ylabel('時間 (秒)', color='white', fontsize=14, alpha=text_alpha)
                ax.set_title('CQTスペクトログラム + リアルタイム周波数断面', 
                           color='white', fontsize=16, pad=20, weight='bold', alpha=text_alpha)
                ax.tick_params(colors='white', labelsize=12)
                
                # 軸ラベルの透明度設定
                if self.is_video_input:
                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        label.set_alpha(self.overlay_opacity)
            
            # 進捗表示
            progress = (frame_idx + 1) / len(actual_frames) * 100
            fig.suptitle(f'音声スペクトログラム解析 - {current_time:.1f}s / {audio_duration:.1f}s ({progress:.1f}%)', 
                        color='white', fontsize=18, y=0.95,
                        alpha=self.overlay_opacity if self.is_video_input else 1.0)
            
            plt.tight_layout()
            return []
        
        print("アニメーション作成中...")
        anim = animation.FuncAnimation(fig, animate, 
                                      frames=len(actual_frames),
                                      interval=1000/self.target_fps, 
                                      blit=False)
        
        return anim, audio_duration
    
    def process(self):
        """メイン処理"""
        # 入力タイプ判定
        self.detect_input_type()
        
        # 音声読み込み
        y, sr = self.load_audio()
        
        # アニメーション作成
        anim, audio_duration = self.create_animation(y, sr)
        
        # 一時映像ファイル保存
        temp_video = 'temp_cqt_video.mp4'
        print("映像を保存中...")
        
        if self.is_video_input:
            # 透明背景で保存（codec='png'またはアルファチャンネル対応codec使用）
            # 注意: MP4はアルファチャンネル非対応なので、合成時に処理
            anim.save(temp_video, writer='ffmpeg', fps=self.target_fps, 
                     dpi=100, savefig_kwargs={'transparent': True})
        else:
            anim.save(temp_video, writer='ffmpeg', fps=self.target_fps, dpi=100)
        
        plt.close()
        
        # 映像と音声の合成
        print("映像と音声を合成中...")
        
        if self.is_video_input:
            # 元動画とオーバーレイ
            overlay_video = VideoFileClip(temp_video)
            
            # サイズを元動画に合わせる
            overlay_video = overlay_video.resized(height=self.base_video.h)
            if overlay_video.w > self.base_video.w:
                overlay_video = overlay_video.resized(width=self.base_video.w)
            
            # センタリング
            overlay_video = overlay_video.with_position('center')
            
            # 透明度設定（全体的な不透明度）
            overlay_video = overlay_video.with_opacity(self.overlay_opacity)
            
            # 合成
            final_duration = min(self.base_video.duration, audio_duration)
            base_trimmed = self.base_video.subclipped(0, final_duration)
            overlay_trimmed = overlay_video.subclipped(0, final_duration)
            
            # CompositeVideoClipで合成
            final = CompositeVideoClip([base_trimmed, overlay_trimmed])
            
            # 音声は元動画のものを使用
            final = final.with_audio(base_trimmed.audio)
            
            overlay_video.close()
        else:
            # 音声のみの場合は通常処理
            video = VideoFileClip(temp_video)
            audio = AudioFileClip(self.input_file)
            
            final_duration = min(video.duration, audio.duration)
            video_trimmed = video.subclipped(0, final_duration)
            audio_trimmed = audio.subclipped(0, final_duration)
            
            final = video_trimmed.with_audio(audio_trimmed)
            
            video.close()
            audio.close()
        
        # 最終出力
        print(f"最終動画を保存中: {self.output_file}")
        final.write_videofile(self.output_file, fps=self.target_fps, 
                            codec='libx264', audio_codec='aac')
        
        # クリーンアップ
        final.close()
        if self.base_video:
            self.base_video.close()
        os.remove(temp_video)
        
        print(f"\n✓ 完成！ {self.output_file} を保存しました！")
        print(f"動画の長さ: {final_duration:.2f}秒")


def main():
    """コマンドライン実行用"""
    parser = argparse.ArgumentParser(description='CQTスペクトログラムビジュアライザー')
    parser.add_argument('input', help='入力ファイル（音声または動画）')
    parser.add_argument('-o', '--output', default='output.mp4', 
                       help='出力ファイル名（デフォルト: output.mp4）')
    
    # 表示フラグ
    parser.add_argument('--no-spectrogram', action='store_true', 
                       help='スペクトログラムを非表示')
    parser.add_argument('--no-bars', action='store_true', 
                       help='周波数バーを非表示')
    parser.add_argument('--no-grid', action='store_true', 
                       help='周波数グリッドを非表示')
    parser.add_argument('--no-marker', action='store_true', 
                       help='時刻マーカーを非表示')
    
    # 不透明度設定
    parser.add_argument('--opacity', type=float, default=0.7, 
                       help='オーバーレイの不透明度 0.0-1.0（デフォルト: 0.7）')
    parser.add_argument('--bg-opacity', type=float, default=0.3, 
                       help='背景の不透明度 0.0-1.0（デフォルト: 0.3）')
    
    args = parser.parse_args()
    
    # ビジュアライザー初期化
    visualizer = CQTVisualizer(args.input, args.output)
    
    # 表示フラグ設定
    visualizer.set_display_flags(
        spectrogram=not args.no_spectrogram,
        bars=not args.no_bars,
        grid=not args.no_grid,
        time_marker=not args.no_marker
    )
    
    # 不透明度設定
    visualizer.set_overlay_opacity(args.opacity, args.bg_opacity)
    
    # 実行
    visualizer.process()


if __name__ == "__main__":
    # スクリプト直接実行の例
    
    # === 使用例1: 音声ファイルから作成 ===
    # visualizer = CQTVisualizer('sound.wav', 'output_audio.mp4')
    # visualizer.process()
    
    # === 使用例2: 動画にオーバーレイ（デフォルト設定） ===
    # visualizer = CQTVisualizer('input_video.mp4', 'output_overlay.mp4')
    # visualizer.process()
    
    # === 使用例3: スペクトログラムのみ表示 ===
    # visualizer = CQTVisualizer('input.mp4', 'output_spectrum_only.mp4')
    # visualizer.set_display_flags(spectrogram=True, bars=False, grid=False, time_marker=False)
    # visualizer.set_overlay_opacity(0.5, 0.1)  # 半透明
    # visualizer.process()
    
    # === 使用例4: バーグラフのみ表示 ===
    # visualizer = CQTVisualizer('input.mp4', 'output_bars_only.mp4')
    # visualizer.set_display_flags(spectrogram=False, bars=True, grid=True, time_marker=True)
    # visualizer.set_overlay_opacity(0.8, 0.0)  # 背景なし
    # visualizer.process()
    
    # コマンドライン実行
    main()