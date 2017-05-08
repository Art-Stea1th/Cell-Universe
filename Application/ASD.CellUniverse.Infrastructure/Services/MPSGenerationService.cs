using System;
using System.Linq;
using System.Windows.Media;
using System.Windows.Threading;

namespace ASD.CellUniverse.Infrastructure.Services {

    using Interfaces;

    public class MPSGenerationService : IMPSGenerator {

        private DispatcherTimer timer;

        private DoubleCollection fpsCollection;
        private double MinFPS => fpsCollection.First();
        private double MaxFPS => fpsCollection.Last();
        private double fps;        
        
        public DoubleCollection MPSCollection => fpsCollection;
        public double MPS { get => fps; set { fps = ValidFps(value); UpdateTimerInterval(); } }


        public event Action NextFrameTime;
        public void Start() => timer.Start();
        public void Stop() => timer.Stop();

        internal MPSGenerationService() {
            fpsCollection = new DoubleCollection { 1.0, 2.0, 3.0, 5.0, 15.0, 30.0, 60.0, 120.0, 125.0 }; // Last = NoLimit
            timer = new DispatcherTimer();
            timer.Tick += (s, e) => NextFrameTime?.Invoke();
            MPS = fpsCollection.TakeWhile(f => f < 120.0).Last();
        }

        private double ValidFps(double fps) => fps < MinFPS ? MinFPS : fps > MaxFPS ? MaxFPS : fps;
        private void UpdateTimerInterval() => timer.Interval =
            fps == fpsCollection.Last()                                                           // <-- if Last - No Limit
            ? TimeSpan.FromTicks(1)
            : TimeSpan.FromMilliseconds(1000.0 / ValidFps(fps));
    }
}