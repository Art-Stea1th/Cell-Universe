using System;
using System.Linq;
using System.Windows.Media;
using System.Windows.Threading;

namespace ASD.CellUniverse.Infrastructure.Services {

    using Interfaces;

    public class FPSGenerationService : IFPSGenerator {

        private DispatcherTimer timer;

        private DoubleCollection fpsCollection;
        private double MinFPS => fpsCollection.First();
        private double MaxFPS => fpsCollection.Last();
        private double fps;        
        
        public DoubleCollection FPSCollection => fpsCollection;
        public double FPS { get => fps; set { fps = ValidFps(value); UpdateTimerInterval(); } }


        public event Action NextFrameTime;
        public void Start() => timer.Start();
        public void Stop() => timer.Stop();

        internal FPSGenerationService() {
            fpsCollection = new DoubleCollection { 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0, 60.0, 120.0 };
            timer = new DispatcherTimer();
            timer.Tick += (s, e) => NextFrameTime?.Invoke();
            FPS = fpsCollection.Last();
        }

        private double ValidFps(double fps) => fps < MinFPS ? MinFPS : fps > MaxFPS ? MaxFPS : fps;
        private void UpdateTimerInterval() => timer.Interval =
            fps == fpsCollection.Last()
            ? TimeSpan.FromTicks(1)
            : TimeSpan.FromMilliseconds(1000.0 / ValidFps(fps));
    }
}