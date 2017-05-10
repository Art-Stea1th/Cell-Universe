using System;
using System.Linq;
using System.Windows.Media;
using System.Windows.Threading;

namespace ASD.CellUniverse.Infrastructure.Services {

    using Interfaces;

    public class EPSGenerationService : IEPSGenerator {

        private DispatcherTimer timer;

        private DoubleCollection epsCollection;
        private double MinEPS => epsCollection.First();
        private double MaxEPS => epsCollection.Last();
        private double eps;        
        
        public DoubleCollection EPSCollection => epsCollection;
        public double EPS { get => eps; set { eps = ValidEps(value); UpdateTimerInterval(); } }


        public event Action NextFrameTime;
        public void Start() => timer.Start();
        public void Stop() => timer.Stop();

        internal EPSGenerationService() {
            epsCollection = new DoubleCollection { 1.0, 2.0, 3.0, 5.0, 15.0, 30.0, 60.0, 120.0, 125.0 }; // Last = NoLimit
            timer = new DispatcherTimer();
            timer.Tick += (s, e) => NextFrameTime?.Invoke();
            EPS = epsCollection.TakeWhile(f => f < epsCollection.Last()).Last();
        }

        private double ValidEps(double eps) => eps < MinEPS ? MinEPS : eps > MaxEPS ? MaxEPS : eps;
        private void UpdateTimerInterval() => timer.Interval =
            eps == epsCollection.Last()                                                           // <-- if Last - NoLimit
            ? TimeSpan.FromTicks(1)
            : TimeSpan.FromMilliseconds(1000.0 / ValidEps(eps));
    }
}