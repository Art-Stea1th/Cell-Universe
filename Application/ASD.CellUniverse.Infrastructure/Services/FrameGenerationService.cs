using System;
using System.Windows.Media;
using System.Windows.Threading;

namespace ASD.CellUniverse.Infrastructure.Services {

    using Interfaces;
    using MVVM;

    public class FrameGenerationService : BindableBase, IFrameSequenceGenerator {

        private DispatcherTimer timer;

        private double fps = 30.0;
        private DoubleCollection fpsCollection;

        private IGenerationAlgorithm algorithm;
        private byte[,] generatedData;

        public double MinFPS => 1.0;
        public double MaxFPS => 120.0;
        public double FPS {
            get => fps;
            set {
                SetProperty(ref fps, ValidFps(value));
                UpdateTimerInterval();
            }
        }
        public DoubleCollection FPSCollection => fpsCollection;

        public IGenerationAlgorithm GenerationAlgorithm {
            get => algorithm;
            set => SetProperty(ref algorithm, value);
        }

        public byte[,] GeneratedData {
            get => generatedData;
            set {
                SetProperty(ref generatedData, value);
                NextFrameReady?.Invoke(generatedData);
            }
        }

        public event Action<byte[,]> NextFrameReady;

        public void Play() => timer.Start();
        public void Pause() => timer.Stop();
        public void Resume() => timer.Start();

        public void Stop() { timer.Stop(); Reset(); }
        public void Reset() => GeneratedData = new byte[GeneratedData.GetLength(0), GeneratedData.GetLength(1)];

        public FrameGenerationService(IGenerationAlgorithm algorithm) {
            fpsCollection = new DoubleCollection { 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0, 60.0, 120.0 };
            this.algorithm = algorithm;
            InitializeTimer();
        }

        private void InitializeTimer() {
            timer = new DispatcherTimer();
            timer.Tick += GenerateNext;
            UpdateTimerInterval();
        }

        private void GenerateNext(object sender, EventArgs e) => GeneratedData = algorithm?.GenerateNextBy(generatedData);
        private double ValidFps(double fps) => fps < MinFPS ? MinFPS : fps > MaxFPS ? MaxFPS : fps;
        private void UpdateTimerInterval() => timer.Interval = TimeSpan.FromMilliseconds(1000.0 / fps);
    }
}