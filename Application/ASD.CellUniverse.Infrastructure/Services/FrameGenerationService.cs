using System;
using System.Windows.Media;
using System.Windows.Threading;

namespace ASD.CellUniverse.Infrastructure.Services {

    using Interfaces;
    using MVVM;

    public class FrameGenerationService : BindableBase, IFrameSequenceGenerator {

        private const double minFps = 1000.0 / 60000.0; // 1   frame  per minute
        private const double maxFps = 120.0;            // 120 frames per second

        private DispatcherTimer timer;

        private double fps = 60.0;
        private IGenerationAlgorithm algorithm;
        private bool[,] generatedData;

        public double FPS {
            get => fps;
            set {
                SetProperty(ref fps, ValidFps(value));
                UpdateTimerInterval();
            }
        }

        public IGenerationAlgorithm GenerationAlgorithm {
            get => algorithm;
            set => SetProperty(ref algorithm, value);
        }

        public bool[,] GeneratedData {
            get => generatedData;
            set {
                SetProperty(ref generatedData, value);
                NextFrameReady?.Invoke(generatedData);
            }
        }

        public event Action<bool[,]> NextFrameReady;

        public void Start() => timer.Start();
        public void Pause() => timer.Stop();
        public void Stop() {
            timer.Stop();
        }

        public FrameGenerationService(IGenerationAlgorithm algorithm) {
            this.algorithm = algorithm;
            InitializeTimer();
        }

        private void InitializeTimer() {
            timer = new DispatcherTimer();
            timer.Tick += GenerateNext;
            UpdateTimerInterval();
        }

        private void GenerateNext(object sender, EventArgs e) => GeneratedData = algorithm?.GenerateNextBy(generatedData);
        private double ValidFps(double fps) => fps < minFps ? minFps : fps > maxFps ? maxFps : fps;
        private void UpdateTimerInterval() => timer.Interval = TimeSpan.FromMilliseconds(1000.0 / fps);
    }
}